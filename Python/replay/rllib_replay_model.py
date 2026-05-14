"""
RLlib Checkpoint 回放脚本
=========================
加载 rllib_custom_network.py 训练产生的 RLlib checkpoint，
连接到 Godot 环境进行可视化推理回放。

支持:
  - 多智能体 (PettingZoo AEC API, 每个 agent 独立策略)
  - GRU_MLP 隐藏态自动管理
  - 自动查找最新 checkpoint

用法:
    # 修改 Config 数据类默认值后直接运行:
    python Python/replay/rllib_replay_model.py

    # 或在命令行覆盖:
    python Python/replay/rllib_replay_model.py --ckpt logs/rllib/PPO_xxx --env godot-game/build/game.exe
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import ray

# ── Monkey-patch packaging.version.Version 以兼容 pickle 反序列化 ──
# packaging 26.x 的 Version 使用 __slots__ 且无 __setstate__,
# 导致 RLlib cloudpickle checkpoint 反序列化时 BUILD 操作失败:
#   _pickle.UnpicklingError: state is not a dictionary
import packaging.version as _packaging_version

def _version_setstate(self, state):
    """兼容 tuple (pickle 格式: epoch, release, pre, post, dev, local) 和 dict 状态。"""
    if isinstance(state, tuple):
        # Version(epoch, release, pre, post, dev, local) — 6 个位置参数
        self._epoch = state[0]
        self._release = state[1] if len(state) > 1 else ()
        self._pre = state[2] if len(state) > 2 else None
        self._post = state[3] if len(state) > 3 else None
        self._dev = state[4] if len(state) > 4 else None
        self._local = state[5] if len(state) > 5 else None
        self._key_cache = {}
    else:
        for k, v in state.items():
            setattr(self, k, v)

_packaging_version.Version.__setstate__ = _version_setstate

# 保证 training 目录下的模块可导入
_TRAINING_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "training")
sys.path.insert(0, _TRAINING_DIR)

import gymnasium as gym
from gymnasium import spaces

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.checkpoints import get_checkpoint_info
from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv

from rllib_custom_network import CustomSegmentedModel # type: ignore

# ── 注册自定义模型 ──────────────────────────────────────────
# 必须在 Algorithm.from_checkpoint() 之前注册, 否则反序列化失败。
ModelCatalog.register_custom_model("custom_segmented_model", CustomSegmentedModel)

# ── 占位 Godot 环境 (动态注册) ──────────────────────────────
# RLlib Algorithm.__init__ 会验证环境, checkpoint 中 env='godot' 需已注册。
# 推理时使用独立的 GDRLPettingZooEnv, 此处仅让 worker 初始化不崩溃。
# 不同 checkpoint 可能使用不同的观测维度 (取决于 valid_mask 等),
# 因此需要先从 checkpoint config 中提取真实总维度再做注册。


def _make_placeholder_env_for_dim(total_obs_dim: int) -> type:
    """给定观测总维度, 创建一个占位 MultiAgentEnv 类。

    RLlib 2.40 多智能体模式下 RolloutWorker.__init__ 会校验 env 类型,
    要求 env 必须是 MultiAgentEnv/BaseEnv/ExternalMultiAgentEnv 的子类。
    """
    class _PlaceholderGodotEnv(MultiAgentEnv):
        def __init__(self, config=None):
            super().__init__()
            self.observation_space = spaces.Dict({
                "0": spaces.Box(-1e6, 1e6, (total_obs_dim,), dtype=np.float32),
            })
            self.action_space = spaces.Dict({
                "0": spaces.Discrete(6),
            })
            self._agent_ids = {"0"}
            self._dummy_obs = self.observation_space["0"].sample()

        def reset(self, *, seed=None, options=None):
            return {"0": self._dummy_obs}, {}

        def step(self, action_dict):
            return (
                {"0": self._dummy_obs},
                {"0": 0.0},
                {"0": False, "__all__": False},
                {"0": False},
                {},
            )

    return _PlaceholderGodotEnv


def _extract_total_obs_dim(state: dict) -> int:
    """从 checkpoint state 的 custom_model_config 中提取观测总维度。

    观测维度可能来自:
      1. custom_model_config.obs_seg_dims 的 total 字段 (分段模型)
      2. observation_space 本身的 shape (非自定义模型)

    返回总维度 int。
    """
    model_cfg = state.get("config", {}).get("model", {})
    custom = model_cfg.get("custom_model_config", {}) or {}

    # 途径 1: obs_seg_dims (可能是 ObsSegmentDims 实例或 repr 字符串)
    obs_seg = custom.get("obs_seg_dims")
    if obs_seg is not None:
        if hasattr(obs_seg, "total"):
            return int(obs_seg.total)
        if isinstance(obs_seg, str):
            m = re.search(r"total\s*=\s*(\d+)", obs_seg)
            if m:
                return int(m.group(1))

    # 途径 2: observation_space shape
    obs_space = state.get("config", {}).get("observation_space")
    if obs_space is not None:
        shape = getattr(obs_space, "shape", None)
        if shape is not None:
            return int(shape[0])  # Box 的 shape 是 tuple

    # 途径 3: 推测 — 从 obs_seg_dims repr 中手动求和各段
    if isinstance(obs_seg, str):
        import re
        parts = re.findall(r"(\w+_dim)=(\d+)", obs_seg)
        total = sum(int(v) for _, v in parts)
        if total > 0:
            return total

    raise ValueError(
        f"无法从 checkpoint state 中提取观测维度。\n"
        f"obs_seg_dims = {obs_seg!r}\n"
        f"observation_space = {state.get('config', {}).get('observation_space')!r}"
    )


def _register_env_for_checkpoint(state: dict) -> int:
    """从 checkpoint state 提取观测维度并注册匹配的占位环境。返回 total_obs_dim。"""
    from ray import tune

    total_dim = _extract_total_obs_dim(state)
    env_cls = _make_placeholder_env_for_dim(total_dim)
    tune.register_env("godot", lambda cfg, cls=env_cls: cls(cfg))
    return total_dim


# ╔══════════════════════════════════════════════════════════╗
# ║                    配  置                                ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class ReplayConfig:
    """RLlib checkpoint 回放配置 — 直接修改默认值即可。"""

    # ── Checkpoint ──
    checkpoint_path: str = "logs\\rllib\\PPO_2026-05-13_14-33-36\\PPO_2026-05-13_16-57-27\\PPO_godot_c46fe_00000_0_2026-05-13_16-57-27"
    """Checkpoint 目录路径。
    支持三种格式:
      - 直接指向 checkpoint_xxx 目录
      - 指向 trial 目录 (自动找最新 checkpoint_xxx)
      - 指向实验根目录 (自动找最新 trial → 最新 checkpoint)"""

    # ── Godot 环境 ──
    env_path: Optional[str] = "curriculum_envs/s1-no-wall-for ball/build/game.exe"
    """Godot 可执行文件路径。None 时连接 Godot 编辑器 """

    show_window: bool = True
    """是否显示游戏窗口。"""

    speedup: int = 1
    """Godot 物理加速倍数 (1=正常速度)。"""

    # ── 多智能体 ──
    is_multiagent: bool = True
    """是否为多智能体环境。"""

    policy_names: tuple = ("0", "1", "2", "3")
    """多智能体各 agent 对应的策略名 (需与训练时一致)。"""

    # ── 推理控制 ──
    deterministic: bool = True
    """确定性推理 (argmax 而非采样)。"""

    max_episodes: int = 0
    """最大回放 episode 数 (0=无限, 按 Ctrl+C 停止)。"""

    # ── 其他 ──
    seed: int = 0
    """随机种子。"""


# ╔══════════════════════════════════════════════════════════╗
# ║                  Checkpoint 路径解析                      ║
# ╚══════════════════════════════════════════════════════════╝

def _resolve_checkpoint(raw_path: str) -> pathlib.Path:
    """解析 checkpoint 路径, 支持多层自动查找。

    查找优先级:
      1. 路径本身是 checkpoint 目录 (含 algorithm_state.pkl)
      2. 路径下直接包含 checkpoint_xxx 子目录 → 取最新
      3. 路径下包含 trial 目录 (PPO_xxx) → 取最新 trial → 最新 checkpoint
    """
    p = pathlib.Path(raw_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"路径不存在: {p}")

    # 1. 直接就是 checkpoint
    if _is_checkpoint(p):
        return p

    # 2. 包含 checkpoint_xxx 子目录
    ckpt_dirs = _find_checkpoint_dirs(p)
    if ckpt_dirs:
        return max(ckpt_dirs, key=lambda d: d.stat().st_mtime)

    # 3. 包含 trial 目录 (如 PPO_2026-05-13_xx)
    trial_dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("PPO_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for trial in trial_dirs:
        ckpt_dirs = _find_checkpoint_dirs(trial)
        if ckpt_dirs:
            return max(ckpt_dirs, key=lambda d: d.stat().st_mtime)

    raise FileNotFoundError(
        f"在 {p} 及其子目录中未找到任何 checkpoint_xxx 目录。\n"
        f"请确认路径指向 RLlib 训练输出目录。"
    )


def _is_checkpoint(p: pathlib.Path) -> bool:
    """判断路径是否为 RLlib checkpoint 目录。"""
    if not p.is_dir():
        return False
    markers = ("algorithm_state.pkl", "rllib_checkpoint.json", "checkpoint.pkl")
    return any((p / m).exists() for m in markers)


def _find_checkpoint_dirs(parent: pathlib.Path) -> list[pathlib.Path]:
    """在目录下查找所有 checkpoint_xxx 子目录。"""
    return sorted(
        [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: d.stat().st_mtime,
    )


# ╔══════════════════════════════════════════════════════════╗
# ║                    回放主逻辑                             ║
# ╚══════════════════════════════════════════════════════════╝

def replay_multiagent(
    env: GDRLPettingZooEnv,
    algo: Algorithm,
    policy_names: tuple,
    config: ReplayConfig,
) -> None:
    """多智能体回放 — PettingZoo ParallelEnv API。

    每个 agent 有独立策略和独立 RNN 隐藏态 (GRU_MLP 时)。
    每个环境步为所有活跃 agent 构造动作字典, 然后统一 step。
    """
    policies = {
        name: algo.get_policy(name) for name in policy_names
    }

    def policy_name_for_agent(agent_id) -> str:
        if isinstance(agent_id, int):
            return policy_names[agent_id]
        if isinstance(agent_id, str) and agent_id.isdigit():
            return policy_names[int(agent_id)]
        return str(agent_id)

    def obs_to_array(obs) -> np.ndarray:
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        return np.asarray(obs, dtype=np.float32)

    obs_dict, _ = env.reset(seed=config.seed)

    # 每 agent 实例的 RNN 隐藏态 (SEGMENTED_MLP 时为空列表)
    rnn_states = {
        agent_id: policies[policy_name_for_agent(agent_id)].get_initial_state()
        for agent_id in env.agents
    }

    episode_count = 0
    episode_rewards: dict[int, float] = {}
    step_count = 0

    while True:
        actions = {}
        for agent_id, obs in obs_dict.items():
            policy_name = policy_name_for_agent(agent_id)
            policy = policies[policy_name]

            if agent_id not in rnn_states:
                rnn_states[agent_id] = policy.get_initial_state()

            action, new_state, _ = policy.compute_single_action(
                obs_to_array(obs),
                state=rnn_states[agent_id],
                explore=not config.deterministic,
            )
            # GDRLPettingZooEnv → godot_env.from_numpy 期望 action[agent_idx][j] 可索引,
            # 但 Discrete 空间的 compute_single_action 返回标量 int → 包装为 1 维数组
            actions[agent_id] = np.array([action], dtype=np.int64)
            rnn_states[agent_id] = new_state

        obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        step_count += 1

        for agent_id, reward in rewards.items():
            episode_rewards[agent_id] = episode_rewards.get(agent_id, 0.0) + float(reward or 0.0)

        # env_is_multiagent=true 时 Godot 会在所有 AIController done 后结束 episode。
        episode_done = all(
            terminations.get(agent_id, False) or truncations.get(agent_id, False)
            for agent_id in actions
        )
        if episode_done:
            total_reward = sum(episode_rewards.values())
            episode_count += 1
            print(
                f"[Ep {episode_count:4d}] "
                f"步数={step_count:6d}  "
                f"总奖励={total_reward:+.1f}  "
                + "  ".join(
                    f"agent_{agent_id}={episode_rewards.get(agent_id, 0.0):+.1f}"
                    for agent_id in sorted(actions, key=str)
                )
            )
            step_count = 0
            episode_rewards = {}

            if 0 < config.max_episodes <= episode_count:
                print(f"[Done] 达到最大 episode 数 {config.max_episodes}, 结束回放。")
                return

            obs_dict, _ = env.reset(seed=config.seed)
            rnn_states = {
                agent_id: policies[policy_name_for_agent(agent_id)].get_initial_state()
                for agent_id in env.agents
            }


def _parse_args() -> argparse.Namespace:
    """解析命令行参数, 覆盖 ReplayConfig 默认值。"""
    parser = argparse.ArgumentParser(
        description="RLlib Checkpoint 回放脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python Python/replay/rllib_replay_model.py
  python Python/replay/rllib_replay_model.py --ckpt logs/rllib/PPO_2026-05-13_xx/checkpoint_000100
  python Python/replay/rllib_replay_model.py --ckpt logs/rllib --env godot-game/build/game.exe --speedup 8
        """,
    )
    parser.add_argument("--ckpt", default=None, type=str,
                        help="Checkpoint 路径 (覆盖 Config.checkpoint_path)")
    parser.add_argument("--env", default=None, type=str,
                        help="Godot 可执行文件路径 (覆盖 Config.env_path)")
    parser.add_argument("--speedup", default=None, type=int,
                        help="物理加速倍数")
    parser.add_argument("--no-window", action="store_true",
                        help="不显示游戏窗口")
    parser.add_argument("--stochastic", action="store_true",
                        help="随机动作采样 (默认确定性)")
    parser.add_argument("--max-episodes", default=None, type=int,
                        help="最大回放 episode 数")
    parser.add_argument("--single-agent", action="store_true",
                        help="单智能体模式")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ReplayConfig()

    # 命令行覆盖
    if args.ckpt:
        config.checkpoint_path = args.ckpt
    if args.env:
        config.env_path = args.env
    if args.speedup is not None:
        config.speedup = args.speedup
    if args.no_window:
        config.show_window = False
    if args.stochastic:
        config.deterministic = False
    if args.max_episodes is not None:
        config.max_episodes = args.max_episodes
    if args.single_agent:
        config.is_multiagent = False

    # ── 1. 解析 checkpoint 路径 ──
    ckpt_dir = _resolve_checkpoint(config.checkpoint_path)
    print(f"[Checkpoint] {ckpt_dir}")

    # ── 2. 初始化 Ray (Algorithm.from_checkpoint 需要) ──
    # Ray worker 进程需要 PYTHONPATH 以导入 godot_env_wrapper 等本地模块
    ray.init(
        ignore_reinit_error=True,
        logging_level="ERROR",
        runtime_env={"env_vars": {"PYTHONPATH": _TRAINING_DIR}},
    )

    # ── 3. 加载算法 ──
    # 推理模式不需要 rollout workers, 但 Algorithm.__init__ 中就会尝试创建。
    # 必须先修改 config 中的 num_workers, 再创建算法实例。
    # 同时需要从 checkpoint config 提取真实观测维度, 注册匹配的占位环境,
    # 否则自定义模型 (CustomSegmentedModel) 的 obs_space 校验会失败。
    print("[Load] 从 checkpoint 恢复策略...")
    checkpoint_info = get_checkpoint_info(str(ckpt_dir))
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    # 禁用远程 workers 和多余 env runner, 避免 Godot 进程被拉起
    # RLlib 2.40 使用 num_env_runners (非 num_workers)
    state["config"]["num_env_runners"] = 0
    state["config"]["num_envs_per_env_runner"] = 1
    # 推理模式下不需要真正的 multi-agent env, 但 RLlib RolloutWorker 会校验类型。
    # 占位环境已继承 MultiAgentEnv, 此配置项配合确保校验通过。
    state["config"]["disable_env_checking"] = True

    # 动态注册匹配的占位环境 (在 Algorithm.from_state 之前!)
    total_obs_dim = _register_env_for_checkpoint(state)
    print(f"[Load] 观测维度: {total_obs_dim}")

    algo = Algorithm.from_state(state)
    print("[Load] 策略恢复完成")

    # ── 4. 创建 Godot 环境 ──
    print("[Env] 创建 Godot 环境...")
    env_config = {
        "env_path": config.env_path,
        "action_repeat": None,
        "show_window": config.show_window,
        "speedup": config.speedup,
    }

    if config.is_multiagent:
        env = GDRLPettingZooEnv(
            config=env_config,
            show_window=config.show_window,
            seed=config.seed,
        )
        # 从环境获取 policy_names (与训练时一致)
        actual_policy_names = tuple(env.agent_policy_names)
        print(f"[Env] 多智能体模式, agent 策略名: {actual_policy_names}")
    else:
        env = GodotEnv(
            env_path=config.env_path,
            show_window=config.show_window,
            speedup=config.speedup,
        )
        actual_policy_names = None
        print("[Env] 单智能体模式")

    print(f"[Env] show_window={config.show_window} speedup={config.speedup}")
    print("[Replay] 开始回放... 按 Ctrl+C 停止。")

    # ── 5. 推理回放循环 ──
    try:
        if config.is_multiagent:
            replay_multiagent(env, algo, actual_policy_names, config)
        else:
            _replay_singleagent(env, algo, config)
    except KeyboardInterrupt:
        print("\n[Replay] 回放被用户中断。")
    finally:
        env.close()
        algo.stop()
        ray.shutdown()
        print("[Done] 资源已清理。")


def _replay_singleagent(
    env: GodotEnv,
    algo: Algorithm,
    config: ReplayConfig,
) -> None:
    """单智能体回放循环。"""
    policy = algo.get_policy()
    rnn_state = policy.get_initial_state()

    obs, _ = env.reset(seed=config.seed)
    obs = np.asarray(obs, dtype=np.float32)

    episode_count = 0
    episode_reward = 0.0
    step_count = 0

    try:
        while True:
            action, rnn_state, _ = policy.compute_single_action(
                obs,
                state=rnn_state,
                explore=not config.deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(
                {"action": np.asarray(action)}
            )
            obs = np.asarray(obs, dtype=np.float32)
            episode_reward += float(reward or 0.0)
            step_count += 1

            if terminated or truncated:
                episode_count += 1
                print(
                    f"[Ep {episode_count:4d}] "
                    f"步数={step_count:6d}  总奖励={episode_reward:+.1f}"
                )
                episode_reward = 0.0
                step_count = 0
                rnn_state = policy.get_initial_state()

                if 0 < config.max_episodes <= episode_count:
                    print(f"[Done] 达到最大 episode 数 {config.max_episodes}。")
                    return
    except KeyboardInterrupt:
        raise  # 交由上层 finally 处理


# ╔══════════════════════════════════════════════════════════╗
# ║                      入口                                ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    main()
