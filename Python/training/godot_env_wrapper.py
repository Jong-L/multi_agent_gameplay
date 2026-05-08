"""
Godot 强化学习环境包装器
  - GodotDiscreteEnvWrapper : MultiDiscrete → Discrete 动作空间转换
  - ObsSegmentDims         : 从 game_config.tres + VisionSensor 常量计算观测各段维度
  - parse_godot_tres       : Godot .tres 配置文件解析
  - layer_init             : 正交权重初始化
"""
import pathlib
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv


# 环境包装器
class GodotDiscreteEnvWrapper:
    """Godot 离散动作环境包装器
    Godot 的 godot_rl_agents 插件将单离散动作空间报告为 MultiDiscrete([n]),
    本包装器在 Python 侧完成转换。
    """
    def __init__(self, env_path: Optional[str] = None, n_parallel: int = 1,
                 seed: int = 0, **kwargs):
        self._env = CleanRLGodotEnv(
            env_path=env_path, n_parallel=n_parallel, seed=seed, **kwargs
        )
        raw_act = self._env.envs[0].action_space
        if isinstance(raw_act, gym.spaces.MultiDiscrete) and len(raw_act.nvec) == 1:
            self._act_space = gym.spaces.Discrete(int(raw_act.nvec[0]))
        else:
            self._act_space = raw_act

    # 属性代理
    @property
    def single_observation_space(self):
        return self._env.single_observation_space

    @property
    def single_action_space(self):
        return self._act_space

    @property
    def num_envs(self):
        return self._env.num_envs

    #环境接口
    def close(self):
        self._env.close()

    def reset(self, seed=None):
        return self._env.reset(seed=seed)

    def step(self, actions):
        if isinstance(self._act_space, gym.spaces.Discrete):
            if isinstance(self._env.envs[0].action_space, gym.spaces.MultiDiscrete):
                actions = np.asarray(actions,dtype=np.int64).reshape(-1, 1)
        return self._env.step(actions)

#godot tres 配置解析
def parse_godot_tres(path: str) -> dict:
    """解析 Godot .tres 文本配置文件, 提取键值对。
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("["):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            result[key] = value
    return result

# 观测维度分段
@dataclass
class ObsSegmentDims:
    self_dim: int
    player_dim: int
    ball_dim: int
    enemy_dim: int
    map_dim: int
    total: int = 0

    def __post_init__(self):
        self.total = (
            self.self_dim + self.player_dim + self.ball_dim
            + self.enemy_dim + self.map_dim
        )

    @classmethod
    def from_config(cls, config_path: str = "godot-game/configs/game_config.tres"):
        """从 Godot 配置文件 + VisionSensor 常量计算各段维度。
        常量与 res://scripts/scene_scripts/vision_sensor.gd 保持同步。
        """
        # VisionSensor 常量 — 与 vision_sensor.gd 保持一致
        SELF = 6
        SLOT_DIMS = (9, 4, 9)      # player, ball, enemy 每槽维度 (含速度)
        SLOT_COUNTS = (3, 8, 5)    # 每类实体槽位数
        VELOCITY_DIMS = 2

        ray = 36
        valid = 0
        use_vel = True
        try:
            cfg = parse_godot_tres(config_path)
            ray = cfg.get("ray_count", 36)
            if cfg.get("use_observation_valid_mask", True):
                valid = 1
            use_vel = cfg.get("use_velocity_obs", True)
        except (FileNotFoundError, OSError):
            print(f"[Warning] 无法读取 {config_path}, 使用默认值 ray=36 valid=0")

        vel_sub = VELOCITY_DIMS if not use_vel else 0
        return cls(
            self_dim=SELF,
            player_dim=SLOT_COUNTS[0] * (SLOT_DIMS[0] - vel_sub + valid),
            ball_dim=SLOT_COUNTS[1] * (SLOT_DIMS[1] + valid),
            enemy_dim=SLOT_COUNTS[2] * (SLOT_DIMS[2] - vel_sub + valid),
            map_dim=ray,
        )


def layer_init(layer: nn.Module, std: float = np.sqrt(2),
               bias_const: float = 0.0) -> nn.Module:
    """正交初始化
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



#  奖励归一化器 (Welford 算法
class RewardNormalizer:
    """Running reward normalizer — 将奖励近似归一化到零均值单位方差。

    使用 Welford 在线算法维护 running mean/variance，
    避免 per-batch 归一化导致的跨 batch 不一致。

    用法:
        norm = RewardNormalizer(clip=10.0)
        r_norm = norm.normalize(r)   # 归一化 (clip optional)
        norm.update(r)               # 更新统计量
    """

    def __init__(self, clip: Optional[float] = 10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4             # 小初始值, 避免前期方差为 0
        self.clip = clip

    def normalize(self, reward: float) -> float:
        """对单个奖励值做 z-score 归一化 (可选裁剪)。"""
        r = (reward - self.mean) / (np.sqrt(self.var) + 1e-8)
        if self.clip is not None:
            r = np.clip(r, -self.clip, self.clip)
        return float(r)

    def normalize_array(self, rewards: np.ndarray) -> np.ndarray:
        """对奖励数组做批量 z-score 归一化 (可选裁剪)。"""
        r = (rewards - self.mean) / (np.sqrt(self.var) + 1e-8)
        if self.clip is not None:
            r = np.clip(r, -self.clip, self.clip)
        return r.astype(np.float32)

    def update(self, reward: float):
        """用单个奖励值更新 running statistics (Welford 单步)。"""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / max(self.count, 1.0)

    def update_array(self, rewards: np.ndarray):
        """用批量奖励更新 running statistics (Welford 并行)。"""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Chan et al. (1979) 并行合并公式。"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        self.mean = float(new_mean)
        self.count = float(tot_count)
        self.var = float(M2 / tot_count) if tot_count > 0 else 1.0

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


#  训练初始化工具
def init_training_setup(args):
    """初始化 wandb、TensorBoard、随机种子、设备、环境和观测分段。
    """
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # Weights & Biases
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TensorBoard
    writer = SummaryWriter(f"{args.experiment_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 计算设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # Godot 环境
    envs = GodotDiscreteEnvWrapper(
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
        seed=args.seed,
        n_parallel=args.n_parallel,
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "只支持 Discrete 动作空间"

    # 观测维度分段
    seg = ObsSegmentDims.from_config(args.config_path)

    return writer, device, envs, seg, run_name


def save_pt_model(save_path: str, state_dicts: dict, args) -> None:
    """保存 PyTorch 模型检查点。

    Args:
        save_path: 保存路径
        state_dicts: 状态字典映射, 如 {"agent_state_dict": agent.state_dict()}
        args: 训练配置, 将 vars(args) 一并保存以便恢复
    """
    save_path = pathlib.Path(save_path).with_suffix(".pt")
    torch.save(
        {"args": vars(args), **state_dicts},
        str(save_path),
    )
    print(f"[Save] Model saved to {save_path}")

