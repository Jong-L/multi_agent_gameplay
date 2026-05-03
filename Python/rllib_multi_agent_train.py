"""RLlib multi-agent PPO training for the Godot arena.

This script wraps the Godot RL Agents socket protocol as an RLlib MultiAgentEnv.
It keeps all four Godot players in the simulation, while allowing any subset of
players to be controlled by random actions for later evaluation or ablation.

配置方式: 直接修改下方 Config 数据类的默认值后运行
  python Python/rllib_multi_agent_train.py
"""

from __future__ import annotations

import inspect
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from godot_rl.core.godot_env import GodotEnv

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _RLLibMultiAgentEnv
except ImportError:
    class _RLLibMultiAgentEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs


ENV_NAME = "godot_multi_agent_arena"
DEFAULT_AGENT_PREFIX = "player"


def parse_agent_set(raw: str) -> Set[str]:
    if not raw:
        return set()
    result: Set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            result.add(f"{DEFAULT_AGENT_PREFIX}_{item}")
        else:
            result.add(item)
    return result


@dataclass
class Config:
    """RLlib 多智能体 PPO 训练配置 — 修改默认值即可调整参数。"""

    # ---- 环境 ----
    env_path: Optional[str] = "godot-game/build/game.exe"
    """Godot 可执行文件路径 (None 连接编辑器)。"""
    seed: int = 0
    """随机种子。"""
    port: int = GodotEnv.DEFAULT_PORT
    """通信端口。"""
    viz: bool = False
    """显示游戏窗口。"""
    speedup: int = 10
    """物理引擎加速倍数。"""
    action_repeat: int = 8
    """动作重复帧数。"""

    # ---- 实验 ----
    run_name: str = "rllib_multi_agent_ppo"
    """运行名称。"""
    log_dir: str = "logs/rllib"
    """日志目录。"""
    checkpoint_dir: str = "savedmodels/rllib"
    """检查点保存目录。"""
    checkpoint_freq: int = 10
    """每 N 次迭代保存一次检查点。"""
    restore: Optional[str] = None
    """恢复训练的检查点路径 (None 从头训练)。"""

    # ---- RLlib 配置 ----
    framework: str = "torch"
    """计算框架 (torch 或 tf2)。"""
    enable_new_api_stack: bool = False
    """启用 RLlib 新版 API 栈。"""
    num_gpus: float = 0
    """GPU 数量。"""
    num_env_runners: int = 0
    """并行环境 runner 数量。"""
    rollout_fragment_length: int = 128
    """每次 rollout 的片段长度。"""
    train_batch_size: int = 4096
    """训练批次大小。"""
    minibatch_size: int = 512
    """小批量大小。"""
    num_epochs: int = 10
    """每次更新的 epoch 数。"""
    lr: float = 3e-4
    """学习率。"""
    gamma: float = 0.99
    """折扣因子。"""
    lambda_: float = 0.95
    """GAE λ 参数。"""
    clip_param: float = 0.2
    """PPO 裁剪系数。"""
    entropy_coeff: float = 0.001
    """熵系数。"""
    vf_loss_coeff: float = 0.5
    """价值函数损失系数。"""
    grad_clip: float = 0.5
    """梯度裁剪范数。"""
    stop_iters: int = 200
    """停止训练的迭代次数上限。"""
    stop_timesteps: int = 1_000_000
    """停止训练的时间步上限。"""

    # ---- 多智能体 ----
    policy_mode: str = "shared"
    """策略模式: shared (共享策略) 或 independent (独立策略)。"""
    random_agents: str = ""
    """逗号分隔的随机策略智能体 ID, 如 '1,2' 或 'player_1,player_2'。"""
    keep_random_rewards: bool = False
    """是否在结果中包含随机智能体的奖励。"""
    eval: bool = False
    """评估模式 (需设置 restore)。"""
    extra_godot_args: List[str] = field(default_factory=list)
    """传递给 Godot 的额外参数, 格式 ['key=value', ...]。"""


def parse_extra_godot_args(items: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"extra_godot_args 格式须为 key=value, 得到: {item}")
        key, value = item.split("=", 1)
        result[key.strip().lstrip("-")] = value.strip()
    return result


def agent_id(index: int) -> str:
    return f"{DEFAULT_AGENT_PREFIX}_{index}"


class GodotRLLibMultiAgentEnv(_RLLibMultiAgentEnv):
    """Expose one Godot process as one simultaneous multi-agent RLlib env.

    GodotEnv already treats each AIController as a slot in one socket-connected
    environment. RLlib MultiAgentEnv expects dictionaries keyed by agent ID, so
    this adapter converts between:

    - Godot: List[obs/action/reward/done] ordered by AIController index
    - RLlib: Dict["player_i", obs/action/reward/done]

    Random agents are kept inside the Godot simulation but hidden from RLlib's
    observation dict. On each step, this adapter samples their action space.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()

        config = dict(config or {})
        self._closed = False
        self.random_agents: Set[str] = parse_agent_set(config.get("random_agents", ""))
        self.keep_random_rewards = bool(config.get("keep_random_rewards", False))

        env_config = {
            "env_path": none_or_str(config.get("env_path")),
            "port": int(config.get("port", GodotEnv.DEFAULT_PORT)),
            "seed": int(config.get("seed", 0)),
            "show_window": bool(config.get("show_window", False)),
            "action_repeat": config.get("action_repeat"),
            "speedup": config.get("speedup"),
        }
        extra_godot_args = dict(config.get("extra_godot_args", {}))
        env_config.update(extra_godot_args)

        self._env = GodotEnv(**env_config)
        self.possible_agents = [agent_id(i) for i in range(self._env.num_envs)]
        unknown_random = self.random_agents.difference(self.possible_agents)
        if unknown_random:
            raise ValueError(
                f"random_agents contains unknown agent ids: {sorted(unknown_random)}; "
                f"available ids are {self.possible_agents}"
            )

        self.controlled_agents = [
            aid for aid in self.possible_agents if aid not in self.random_agents
        ]
        if not self.controlled_agents:
            raise ValueError("At least one agent must be controlled by RLlib.")

        self.agents = list(self.controlled_agents)

        self.observation_spaces = {
            agent_id(i): space for i, space in enumerate(self._env.observation_spaces)
        }
        # GodotEnv's public action_space is a Tuple of action heads. Using Tuple
        # avoids GodotEnv.from_numpy() receiving dict actions with integer lookup.
        self.action_spaces = {
            agent_id(i): space for i, space in enumerate(self._env.tuple_action_spaces)
        }

        self.observation_space = self.observation_spaces[self.controlled_agents[0]]
        self.action_space = self.action_spaces[self.controlled_agents[0]]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, dict]]:
        del seed, options
        obs_list, info_list = self._env.reset()
        self.agents = list(self.controlled_agents)
        return self._obs_to_agent_dict(obs_list), self._to_agent_dict(info_list)

    def step(
        self,
        action_dict: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        ordered_actions = []
        for index, aid in enumerate(self.possible_agents):
            if aid in self.random_agents:
                action = self.action_spaces[aid].sample()
            else:
                if aid not in action_dict:
                    action = self.action_spaces[aid].sample()
                else:
                    action = action_dict[aid]
            ordered_actions.append(self._normalise_action(action, index))

        obs_list, reward_list, terminated_list, truncated_list, info_list = self._env.step(
            np.array(ordered_actions, dtype=object),
            order_ij=True,
        )

        # The current Godot controller resets the full scene from player_0.
        # Treat any per-agent done as an episode boundary for RLlib as well.
        episode_done = bool(any(terminated_list) or any(truncated_list))
        observations = self._obs_to_agent_dict(obs_list)
        rewards = self._to_agent_dict(reward_list)
        infos = self._to_agent_dict(info_list)

        if self.keep_random_rewards:
            reward_agents: Iterable[str] = self.possible_agents
        else:
            reward_agents = self.controlled_agents
        rewards = {
            aid: float(reward_list[self._agent_index(aid)])
            for aid in reward_agents
            if aid in self.possible_agents
        }

        terminations = {aid: episode_done for aid in self.controlled_agents}
        truncations = {aid: episode_done for aid in self.controlled_agents}
        terminations["__all__"] = episode_done
        truncations["__all__"] = episode_done
        if episode_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        except OSError as exc:
            print(f"[Godot] socket already closed during env.close(): {exc}")

    def get_observation_space(self, aid: str):
        return self.observation_spaces[aid]

    def get_action_space(self, aid: str):
        return self.action_spaces[aid]

    def _to_agent_dict(self, values: Sequence[Any]) -> Dict[str, Any]:
        return {
            aid: values[self._agent_index(aid)]
            for aid in self.controlled_agents
        }

    def _obs_to_agent_dict(self, values: Sequence[Any]) -> Dict[str, Any]:
        return {
            aid: self._coerce_to_space(
                values[self._agent_index(aid)],
                self.observation_spaces[aid],
            )
            for aid in self.controlled_agents
        }

    def _coerce_to_space(self, value: Any, space: Any) -> Any:
        from gymnasium import spaces

        if isinstance(space, spaces.Dict):
            return {
                key: self._coerce_to_space(value[key], subspace)
                for key, subspace in space.spaces.items()
            }
        if isinstance(space, spaces.Box):
            return np.asarray(value, dtype=space.dtype).reshape(space.shape)
        if isinstance(space, spaces.Discrete):
            return int(value)
        if isinstance(space, spaces.Tuple):
            return tuple(
                self._coerce_to_space(sub_value, subspace)
                for sub_value, subspace in zip(value, space.spaces)
            )
        return value

    @staticmethod
    def _agent_index(aid: str) -> int:
        return int(aid.rsplit("_", 1)[1])

    def _normalise_action(self, action: Any, index: int) -> List[Any]:
        action_space = self._env.action_spaces[index]
        action_keys = list(action_space.spaces.keys())

        if isinstance(action, Mapping):
            return [self._to_python_scalar(action[key]) for key in action_keys]
        if isinstance(action, tuple):
            return [self._to_python_scalar(value) for value in action]
        if isinstance(action, list):
            return [self._to_python_scalar(value) for value in action]
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return [self._to_python_scalar(action.item())]
            return [self._to_python_scalar(value) for value in action.tolist()]
        return [self._to_python_scalar(action)]

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value


def register_env(args: Config) -> None:
    from ray.tune.registry import register_env as tune_register_env

    def creator(config: Mapping[str, Any]) -> GodotRLLibMultiAgentEnv:
        merged = dict(config)
        # Offset ports/seeds when RLlib creates multiple env runners.
        worker_index = int(getattr(config, "worker_index", 0) or 0)
        vector_index = int(getattr(config, "vector_index", 0) or 0)
        merged["port"] = int(merged.get("port", args.port)) + worker_index * 100 + vector_index
        merged["seed"] = int(merged.get("seed", args.seed)) + worker_index * 1000 + vector_index
        return GodotRLLibMultiAgentEnv(merged)

    tune_register_env(ENV_NAME, creator)


def build_probe_env(args: Config) -> GodotRLLibMultiAgentEnv:
    config = env_config(args)
    return GodotRLLibMultiAgentEnv(config)


def env_config(args: Config) -> Dict[str, Any]:
    return {
        "env_path": args.env_path,
        "port": args.port,
        "seed": args.seed,
        "show_window": args.viz,
        "speedup": args.speedup,
        "action_repeat": args.action_repeat,
        "random_agents": args.random_agents,
        "keep_random_rewards": args.keep_random_rewards,
        "extra_godot_args": parse_extra_godot_args(args.extra_godot_args),
    }


def build_policy_specs(
    args: Config,
    probe_env: GodotRLLibMultiAgentEnv,
) -> Tuple[Dict[str, Any], List[str]]:
    from ray.rllib.policy.policy import PolicySpec

    random_agents = parse_agent_set(args.random_agents)
    trainable_agents = [
        aid for aid in probe_env.possible_agents if aid not in random_agents
    ]
    if args.policy_mode == "shared":
        first = trainable_agents[0]
        policies = {
            "shared_policy": PolicySpec(
                observation_space=probe_env.observation_spaces[first],
                action_space=probe_env.action_spaces[first],
            )
        }
        policies_to_train = ["shared_policy"]
    else:
        policies = {
            f"{aid}_policy": PolicySpec(
                observation_space=probe_env.observation_spaces[aid],
                action_space=probe_env.action_spaces[aid],
            )
            for aid in trainable_agents
        }
        policies_to_train = sorted(policies.keys())
    return policies, policies_to_train


def make_policy_mapping_fn(policy_mode: str):
    def mapping(agent: str, episode: Any = None, worker: Any = None, **kwargs: Any) -> str:
        del episode, worker, kwargs
        if policy_mode == "shared":
            return "shared_policy"
        return f"{agent}_policy"

    return mapping


def call_config(config: Any, method_name: str, **kwargs: Any) -> Any:
    method = getattr(config, method_name, None)
    if method is None:
        return config
    try:
        return method(**kwargs)
    except TypeError:
        signature = inspect.signature(method)
        allowed = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        if not allowed:
            return config
        return method(**allowed)


def configure_rllib(
    args: Config,
    policies: Mapping[str, Any],
    policies_to_train: Sequence[str],
) -> Any:
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig()
    if hasattr(config, "api_stack"):
        config = config.api_stack(
            enable_rl_module_and_learner=args.enable_new_api_stack,
            enable_env_runner_and_connector_v2=args.enable_new_api_stack,
        )
    config = call_config(
        config,
        "environment",
        env=ENV_NAME,
        env_config=env_config(args),
        disable_env_checking=True,
    )
    config = call_config(config, "framework", framework=args.framework)
    config = call_config(config, "resources", num_gpus=args.num_gpus)
    config = call_config(
        config,
        "multi_agent",
        policies=dict(policies),
        policy_mapping_fn=make_policy_mapping_fn(args.policy_mode),
        policies_to_train=list(policies_to_train),
    )

    config = call_config(
        config,
        "training",
        lr=args.lr,
        gamma=args.gamma,
        lambda_=args.lambda_,
        clip_param=args.clip_param,
        entropy_coeff=args.entropy_coeff,
        vf_loss_coeff=args.vf_loss_coeff,
        grad_clip=args.grad_clip,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        sgd_minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        num_sgd_iter=args.num_epochs,
    )

    if hasattr(config, "env_runners"):
        config = call_config(
            config,
            "env_runners",
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=args.rollout_fragment_length,
        )
    else:
        config = call_config(
            config,
            "rollouts",
            num_rollout_workers=args.num_env_runners,
            num_envs_per_worker=1,
            rollout_fragment_length=args.rollout_fragment_length,
        )

    if args.eval:
        config = call_config(config, "exploration", explore=False)
        config = call_config(config, "evaluation", evaluation_interval=1)
    return config


def result_timesteps(result: Mapping[str, Any]) -> int:
    for key in (
        "num_env_steps_sampled_lifetime",
        "num_agent_steps_sampled_lifetime",
        "timesteps_total",
        "agent_timesteps_total",
    ):
        value = result.get(key)
        if value is not None:
            return int(value)
    counters = result.get("counters", {})
    for key in ("num_env_steps_sampled", "num_agent_steps_sampled"):
        if key in counters:
            return int(counters[key])
    return 0


def maybe_checkpoint(algo: Any, checkpoint_root: pathlib.Path, iteration: int) -> Optional[str]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = algo.save(str(checkpoint_root / f"iter_{iteration:05d}"))
    if hasattr(checkpoint, "checkpoint"):
        return str(checkpoint.checkpoint.path)
    return str(checkpoint)


def print_space_summary(env: GodotRLLibMultiAgentEnv) -> None:
    print("[Godot] agents:", env.possible_agents)
    print("[Godot] random agents:", sorted(env.random_agents))
    for aid in env.possible_agents:
        print(f"[Space] {aid} obs={env.observation_spaces[aid]} action={env.action_spaces[aid]}")


def main() -> None:
    args = Config()
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import ray
    except ImportError as exc:
        raise SystemExit(
            "Ray/RLlib is not installed in this Python environment. "
            "Install a Ray build that matches your Python version, e.g. `pip install \"ray[rllib]\"`."
        ) from exc

    register_env(args)
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    checkpoint_root = pathlib.Path(args.checkpoint_dir) / args.run_name
    probe_env = build_probe_env(args)
    try:
        print_space_summary(probe_env)
        policies, policies_to_train = build_policy_specs(args, probe_env)
    finally:
        probe_env.close()

    config = configure_rllib(args, policies, policies_to_train)
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
    if args.restore:
        print(f"[Restore] {args.restore}")
        algo.restore(args.restore)

    latest_checkpoint: Optional[str] = None
    try:
        for iteration in range(1, args.stop_iters + 1):
            result = algo.train()
            sampled = result_timesteps(result)
            reward_mean = result.get("episode_reward_mean", None)
            print(
                f"[Train] iter={iteration} sampled={sampled} "
                f"episode_reward_mean={reward_mean}"
            )

            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                latest_checkpoint = maybe_checkpoint(algo, checkpoint_root, iteration)
                print(f"[Checkpoint] {latest_checkpoint}")

            if args.stop_timesteps > 0 and sampled >= args.stop_timesteps:
                break
    except (KeyboardInterrupt, ConnectionError, ConnectionResetError) as exc:
        print(f"[Stop] interrupted by {type(exc).__name__}; saving checkpoint.")
    finally:
        latest_checkpoint = maybe_checkpoint(algo, checkpoint_root, 99999)
        print(f"[Final checkpoint] {latest_checkpoint}")
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
