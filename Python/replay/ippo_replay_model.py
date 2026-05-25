"""
IPPO 模型推理回放脚本
======================
加载 custom_ippo.py 训练产生的 per-agent .pt 模型文件,
连接到 Godot 环境进行可视化推理回放。

支持:
  - MLP / SegmentedMLP / GRU_MLP 三种网络架构
  - 多智能体: 从 checkpoint 的 agent_configs 重建每个 agent
  - GRU_MLP 自动维护每个 agent 的 RNN 隐藏态
  - --agent_ids 指定使用哪些 agent 的策略 (其余随机动作)

保存格式: save_ippo_model 将 4 个 agent 分别存为独立文件:
  {base}_agent0.pt, {base}_agent1.pt, {base}_agent2.pt, {base}_agent3.pt
回放时 model_path 指向基础路径 (不含 _agentN 后缀), 自动加载全部 agent。
"""
import pathlib
import sys
from dataclasses import dataclass, fields as dataclass_fields
from typing import Optional

import numpy as np
import torch

# 保证 training 目录下的模块可导入
_TRAINING_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "training")
sys.path.insert(0, _TRAINING_DIR)

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    ObsSegmentDims,
)
from custom_ppo_dataclass import NetworkType, AgentConfig, IppoArgs
from ppo_networks import SegmentedObsHelper
from custom_ippo import IPPOAgent


# ╔══════════════════════════════════════════════════════════╗
# ║                    配  置                                ║
# ╚══════════════════════════════════════════════════════════╝
@dataclass
class ReplayConfig:
    """IPPO 模型回放配置 — 直接修改默认值即可。"""

    model_path: str = "saved_models/ippo_bootstrap_agent.pt"
    """IPPO 模型基础路径 (不含 _agentN 后缀)。
    自动加载 {stem}_agent0.pt ~ {stem}_agent3.pt。
    当 model_paths 不为空时此字段被忽略。"""

    model_paths: Optional[list[str]] = None
    """分别指定 agent 模型文件路径列表。
    示例: ["saved_models/agent0.pt", "saved_models/agent1.pt", ...]
    设置后优先使用此字段, model_path 被忽略。"""
        
    def __post_init__(self):
        if self.model_paths is None:
            self.model_paths = [
                "saved_models/agent0_extra_pool_step102400_agent0.pt",
                "saved_models/ippo_bootstrap_agent1.pt",
                "saved_models/ippo_bootstrap_agent2.pt",
                "saved_models/ippo_bootstrap_agent3.pt"
            ]

    env_path: Optional[str] = "godot-game\\build-multiagent\\game.exe"
    """Godot 可执行文件路径 (None 连接编辑器)。"""

    config_path: str = "godot-game/configs/game_config.tres"
    """game_config.tres 路径, 用于读取观测维度配置。"""

    speedup: int = 1
    """物理引擎加速倍数 (1=正常速度)。"""

    show_window: bool = True
    """显示游戏窗口。"""

    deterministic: bool = True
    """确定性推理 (argmax 而非采样)。"""

    cuda: bool = False
    """回放使用 GPU (默认 CPU)。"""

    max_episodes: int = 0
    """最大回放 episode 数 (0=无限, 按 Ctrl+C 停止)。"""

    seed: int = 0
    """随机种子。"""

    agent_ids: Optional[str] = None
    """指定使用的 agent ID, 逗号分隔 (如 "0,1"). None=全部使用训练策略。"""


# ╔══════════════════════════════════════════════════════════╗
# ║                  Checkpoint 加载                          ║
# ╚══════════════════════════════════════════════════════════╝


def _make_agent_model_path(save_path: str, agent_id: int) -> pathlib.Path:
    """从基础路径推导 per-agent 文件名: {stem}_agent{id}.pt"""
    save_path = pathlib.Path(save_path).with_suffix(".pt")
    return save_path.with_name(f"{save_path.stem}_agent{agent_id}{save_path.suffix}")


def load_agent_checkpoints(model_base_path: pathlib.Path, device: torch.device) -> dict:
    """加载 save_ippo_model() 保存的 per-agent .pt 文件。

    每个 agent 独立存储为 {base}_agentN.pt，格式:
      {"args": {...}, "agent_id": N, "agent_state_dict": ...}

    返回统一格式: {"agent_0_state_dict": ..., "agent_1_state_dict": ..., ...,
                   ...args 字段...}
    """
    model_base_path = model_base_path.with_suffix(".pt")

    # 先加载 agent_0 获取 args, 确定 agent 数量
    agent0_path = _make_agent_model_path(str(model_base_path), 0)
    if not agent0_path.exists():
        raise FileNotFoundError(f"找不到 agent_0 checkpoint: {agent0_path}")

    agent0_ckpt = torch.load(str(agent0_path), map_location=device, weights_only=False)
    raw_args = agent0_ckpt.get("args", {})
    agent_configs = raw_args.get("agent_configs", [])
    n_agents = len(agent_configs) if agent_configs else 4

    result = dict(raw_args)

    for agent_id in range(n_agents):
        agent_path = _make_agent_model_path(str(model_base_path), agent_id)
        if not agent_path.exists():
            print(f"[Warn] 缺少 agent_{agent_id} 文件: {agent_path}, 跳过。")
            continue

        ckpt = (agent0_ckpt if agent_id == 0
                else torch.load(str(agent_path), map_location=device, weights_only=False))
        result[f"agent_{agent_id}_state_dict"] = ckpt["agent_state_dict"]

    return result


def _ensure_network_type(value) -> NetworkType:
    """将 checkpoint 中的 network_type 转为 NetworkType 枚举。"""
    if isinstance(value, NetworkType):
        return value
    return NetworkType(str(value).lower())


def load_agent_checkpoints_from_paths(
    model_paths: list[str], device: torch.device
) -> dict:
    """从分别指定的 per-agent .pt 文件路径加载 checkpoint。

    model_paths: ["path/agent0.pt", "path/agent1.pt", ...]
    每个文件的格式: {"args": {...}, "agent_id": N, "agent_state_dict": ...}

    返回统一格式: {"agent_0_state_dict": ..., "agent_1_state_dict": ..., ...args 字段...}
    """
    if len(model_paths) == 0:
        raise ValueError("model_paths 不能为空")

    # 加载第一个获取 args
    path0 = pathlib.Path(model_paths[0])
    if not path0.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {path0}")

    ckpt0 = torch.load(str(path0), map_location=device, weights_only=False)
    raw_args = ckpt0.get("args", {})
    result = dict(raw_args)

    for i, p in enumerate(model_paths):
        p = pathlib.Path(p)
        if not p.exists():
            print(f"[Warn] 缺少文件: {p}, 跳过。")
            continue
        ckpt = ckpt0 if i == 0 else torch.load(str(p), map_location=device, weights_only=False)
        # 使用 checkpoint 中记录的 agent_id, 回退到索引
        agent_id = ckpt.get("agent_id", i)
        result[f"agent_{agent_id}_state_dict"] = ckpt["agent_state_dict"]

    return result


def build_replay_args_and_agents(checkpoint_data: dict) -> tuple[IppoArgs, list[AgentConfig]]:
    """从 checkpoint 数据重建 IppoArgs 和 agent_configs。

    save_ippo_model 使用 vars(args) 直接 pickle 保存,
    Enum 等对象保持原样反序列化。
    """
    # ── agent_configs ──
    raw_cfgs = checkpoint_data.get("agent_configs")
    if raw_cfgs is None:
        raise KeyError("checkpoint['args'] 缺少 'agent_configs' 字段.")

    agent_configs: list[AgentConfig] = []
    for i, raw_cfg in enumerate(raw_cfgs):
        if isinstance(raw_cfg, AgentConfig):
            agent_configs.append(raw_cfg)
        elif isinstance(raw_cfg, dict):
            # 从 dict 重建 AgentConfig
            cfg_dict = dict(raw_cfg)
            nt = cfg_dict.get("network_type", NetworkType.SEGMENTED_MLP)
            cfg_dict["network_type"] = _ensure_network_type(nt)
            agent_configs.append(AgentConfig(**cfg_dict))
        else:
            raise TypeError(f"agent_configs[{i}] 类型不支持: {type(raw_cfg)}")

    # ── 仅使用 IppoArgs 中存在的字段重建 ──
    ipo_fields = {f.name for f in dataclass_fields(IppoArgs) if f.init}
    ckpt_args = {k: v for k, v in checkpoint_data.items()
                 if k in ipo_fields and k != "agent_configs"}

    args = IppoArgs(agent_configs=agent_configs, **ckpt_args)
    return args, agent_configs


# ╔══════════════════════════════════════════════════════════╗
# ║                    主回放逻辑                             ║
# ╚══════════════════════════════════════════════════════════╝
def main():
    replay_cfg = ReplayConfig()

    device = torch.device("cuda" if torch.cuda.is_available() and replay_cfg.cuda else "cpu")

    # ── 1. 加载 checkpoint ──
    if replay_cfg.model_paths:
        print(f"加载模型 (分别指定路径, 共 {len(replay_cfg.model_paths)} 个):")
        for p in replay_cfg.model_paths:
            print(f"  {p}")
        checkpoint_data = load_agent_checkpoints_from_paths(replay_cfg.model_paths, device)
    else:
        model_path = pathlib.Path(replay_cfg.model_path).resolve()
        print(f"加载模型 (基础路径): {model_path}")
        checkpoint_data = load_agent_checkpoints(model_path, device)

    # ── 2. 重建配置 + agents ──
    args, agent_configs = build_replay_args_and_agents(checkpoint_data)
    n_agents = len(agent_configs)
    n_actions = 6  # Godot 游戏离散动作空间固定为 6

    # 解析 --agent_ids
    active_ids: set[int] = set()
    if replay_cfg.agent_ids is not None:
        try:
            active_ids = {int(x.strip()) for x in replay_cfg.agent_ids.split(",")}
        except ValueError:
            raise ValueError(f"--agent_ids 格式错误: {replay_cfg.agent_ids!r}. 示例: '0,1'")
    else:
        active_ids = {cfg.agent_id for cfg in agent_configs}

    # 确认 agent_id 合法
    ckpt_ids = {cfg.agent_id for cfg in agent_configs}
    invalid = active_ids - ckpt_ids
    if invalid:
        raise ValueError(f"checkpoint 中不存在 agent_id={invalid}, 可用: {sorted(ckpt_ids)}")

    # 重建观测段维度
    config_path = args.config_path or replay_cfg.config_path
    seg = ObsSegmentDims.from_config(config_path)
    print(f"[Obs] segments: self={seg.self_dim} player={seg.player_dim} "
          f"ball={seg.ball_dim} enemy={seg.enemy_dim} map={seg.map_dim}")

    # ── 3. 重建所有 agent 网络 ──
    agents: list[IPPOAgent] = []
    rnn_states: list[Optional[torch.Tensor]] = []

    for cfg in agent_configs:
        aid = cfg.agent_id
        state_key = f"agent_{aid}_state_dict"

        if state_key not in checkpoint_data:
            print(f"[Warn] checkpoint 中缺少 {state_key}, agent_id={aid} 使用随机动作。")
            agents.append(None)  # placeholder
            rnn_states.append(None)
            continue

        agent = IPPOAgent(n_actions, seg, cfg).to(device)
        agent.load_state_dict(checkpoint_data[state_key])
        agent.eval()
        agents.append(agent)

        if agent.is_recurrent:
            rnn_states.append(agent.get_initial_state(1, device))
            print(f"[Agent {aid}] network_type={cfg.network_type}, "
                  f"params={agent.num_params():,}, recurrent=True")
        else:
            rnn_states.append(None)
            print(f"[Agent {aid}] network_type={cfg.network_type}, "
                  f"params={agent.num_params():,}, recurrent=False")

    # ── 4. 初始化环境 ──
    print("初始化 Godot 环境...")
    envs = GodotDiscreteEnvWrapper(
        env_path=replay_cfg.env_path,
        show_window=replay_cfg.show_window,
        speedup=replay_cfg.speedup,
        seed=replay_cfg.seed,
        n_parallel=1,
    )
    num_envs = envs.num_envs

    # 验证: env 槽位数应与 agent 数量匹配
    if num_envs != n_agents:
        print(f"[Warn] 环境槽位数(num_envs={num_envs}) != checkpoint agent数(n_agents={n_agents}). "
              f"将只取前 {num_envs} 个 agent 的推理结果。")
        # 截断到实际 env 槽位数
        agents = agents[:num_envs]
        rnn_states = rnn_states[:num_envs]
        agent_configs = agent_configs[:num_envs]
        active_ids = {cfg.agent_id for cfg in agent_configs}
        n_agents = num_envs

    # ── 5. 推理回放循环 ──
    print("开始回放... 按 Ctrl+C 停止。")
    next_obs, _ = envs.reset(seed=replay_cfg.seed)
    next_obs = np.array(next_obs, dtype=np.float32)

    episode_count = 0
    episode_rewards = np.zeros(n_agents, dtype=np.float64)
    step_count = 0

    try:
        while True:
            actions = []

            with torch.no_grad():
                obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)

                for i in range(n_agents):
                    agent = agents[i]
                    aid = agent_configs[i].agent_id

                    if agent is None or aid not in active_ids:
                        # 未加载模型的 agent: 随机动作
                        actions.append(np.random.randint(0, n_actions))
                        continue

                    rnn_state = rnn_states[i]
                    result = agent.get_action_and_value(
                        obs_t[i].unsqueeze(0),
                        rnn_state=rnn_state,
                        return_state=agent.is_recurrent,
                    )
                    action = result[0]
                    if agent.is_recurrent:
                        rnn_states[i] = result[4]

                    if replay_cfg.deterministic:
                        features = (agent.encoder(obs_t[i].unsqueeze(0), rnn_state)[0]
                                    if agent.is_recurrent
                                    else agent.encoder(obs_t[i].unsqueeze(0))[0])
                        logits = agent.actor(features)
                        action = logits.argmax(dim=1)

                    actions.append(int(action.item()))

            next_obs, rewards, terms, truncs, infos = envs.step(
                np.array(actions, dtype=np.int64)
            )
            next_obs = np.array(next_obs, dtype=np.float32)

            rewards_arr = np.asarray(rewards, dtype=np.float32)
            dones_arr = np.logical_or(
                np.asarray(terms, dtype=bool),
                np.asarray(truncs, dtype=bool),
            )
            episode_rewards += rewards_arr
            step_count += 1

            if dones_arr.any():
                episode_count += 1
                per_agent = " ".join(
                    f"agent_{agent_configs[i].agent_id}={episode_rewards[i]:+.1f}"
                    for i in range(n_agents)
                )
                print(
                    f"[Ep {episode_count:4d}] "
                    f"步数={step_count:6d}  "
                    f"{per_agent}"
                )
                step_count = 0
                episode_rewards[:] = 0.0

                if 0 < replay_cfg.max_episodes <= episode_count:
                    print(f"[Done] 达到最大 episode 数 {replay_cfg.max_episodes}, 结束回放。")
                    break

                # 新 episode 重置每个 agent 的 GRU 隐藏态
                for i in range(n_agents):
                    agent = agents[i]
                    if agent is not None and agent.is_recurrent:
                        rnn_states[i] = agent.get_initial_state(1, device)

    except KeyboardInterrupt:
        print("\n回放被用户中断。")
    finally:
        envs.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
