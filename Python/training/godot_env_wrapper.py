"""
Godot 强化学习环境包装器 & 共享工具
====================================
提供 DQN / PPO / PQN 等训练脚本共用的基础设施:
  - GodotDiscreteEnvWrapper : MultiDiscrete → Discrete 动作空间转换
  - ObsSegmentDims         : 从 game_config.tres + VisionSensor 常量计算观测各段维度
  - parse_godot_tres       : Godot .tres 配置文件解析
  - layer_init             : 正交权重初始化
"""
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

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
                actions = np.asarray(actions).reshape(-1, 1)
        return self._env.step(actions)

#godot tres 配置解析
def parse_godot_tres(path: str) -> dict:
    """解析 Godot .tres 文本配置文件, 提取键值对。

    仅解析顶层 [resource] 段的 key=value 行, 自动推断 bool/int/float 类型。
    跳过 section header ([...]) 和不含 '=' 的行。
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
    """正交初始化 (Orthogonal initialization)。

    权重使用正交矩阵初始化 (gain=std), 偏置初始化为常数。
    适用于 ReLU / Tanh 激活的隐藏层; 输出层建议使用较小的 std。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
