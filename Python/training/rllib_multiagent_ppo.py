"""
RLlib PPO 多智能体训练入口脚本

功能概述：
    - 将 Godot 游戏引擎封装为 RLlib 兼容的多智能体环境
    - 支持独立策略（每个玩家单独策略）和共享策略两种模式
    - 通过命令行参数灵活配置训练过程
    - 集成 Godot RL Agents 插件进行通信

架构设计：
    - GodotMultiAgentEnv: 环境适配层，将 GodotEnv 转换为 MultiAgentEnv
    - 配置系统: 从 Godot .tres 文件读取配置，自动推断观测空间维度
    - 策略管理: 支持 per-agent 策略或共享策略
    - 训练循环: 基于 RLlib 的 PPO 实现，支持检查点保存和恢复

使用方法：
    基本训练:
        conda run -n gdrl python Python/training/rllib_multiagent_ppo.py
    
    从检查点恢复:
        python Python/training/rllib_multiagent_ppo.py --restore logs/checkpoint_000100
    
    使用共享策略:
        python Python/training/rllib_multiagent_ppo.py --shared-policy

观测空间设计（142维）:
    - self_state: 6维 (位置x,y, 血量比例, 朝向, 攻击动画, 技能冷却)
    - nearby_players: 27维 (3个附近玩家 × 9维/玩家)
    - nearby_balls: 32维 (8个奖励球 × 4维/球)
    - nearby_enemies: 45维 (5个敌人 × 9维/敌人)
    - map_state: 32维 (射线检测)

"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from godot_rl.core.godot_env import GodotEnv


AGENT_PREFIX = "player_"  # 智能体ID前缀，用于生成 player_0, player_1, ...
ENV_NAME = "godot_multiagent_independent"  # 在 RLlib 注册表中注册的环境名称


def str_to_optional_path(value: Optional[str]) -> Optional[str]:
    """
    将命令行参数中的路径字符串转换为可选路径对象
    
    设计目的：
        Godot 环境路径可能是可执行文件路径、编辑器模式标识或 None。
        此函数统一处理各种输入格式，返回 None（编辑器模式）或有效路径。
    
    参数：
        value: 用户输入的路径字符串，可能值包括：
            - None: 未指定
            - "": 空字符串
            - "none"/"null": 明确表示使用编辑器
            - "editor": 使用 Godot 编辑器
            - 实际路径: 如 "godot-game/build/game.exe"
    
    返回：
        Optional[str]: None 表示使用 Godot 编辑器，否则返回路径字符串
    """
    if value is None:
        return None
    if value.lower() in {"", "none", "null", "editor"}:
        return None
    return value


def parse_godot_tres(path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    config_path = pathlib.Path(path)
    if not config_path.exists():
        return result

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 跳过空行、节头（[section]）和无等号行
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()

        # 类型自动推断：先尝试 bool，然后 int，然后 float，最后作为字符串
        if raw_value.lower() == "true":
            value: Any = True
        elif raw_value.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip('"')
        result[key] = value
    return result


def infer_flat_obs_dim(config_path: str) -> int:
    """
    根据 Godot 配置文件推断观测空间的维度
    
    设计目的：
        RLlib 需要预先知道观测空间的维度来构建策略网络。
        此函数读取 Godot 配置，根据配置的动态参数（如射线数量、是否使用速度观测等）
        计算出总的观测维度，避免硬编码。
        
    观测空间组成（总计 142 维）：
        - self_state: 6维 (pos_x, pos_y, hp_ratio, flip_h, is_attack_animating, skill_cooldown_ratio)
        - nearby_players: 3个 × (9维 + valid_mask)
        - nearby_balls: 8个 × (4维 + valid_mask)
        - nearby_enemies: 5个 × (9维 + valid_mask)
        - map_state: ray_count 维（射线检测）
    
    参数：
        config_path: Godot 配置文件路径
    
    返回：
        int: 观测空间总维度
    """
    cfg = parse_godot_tres(config_path)
    
    # 从配置文件读取动态参数，使用默认值作为后备
    ray_count = int(cfg.get("ray_count", 36))  # 射线数量，默认 36
    use_valid_mask = bool(cfg.get("use_observation_valid_mask", False))  # 是否使用有效掩码
    use_velocity_obs = bool(cfg.get("use_velocity_obs", True))  # 是否包含速度观测

    # 基础维度定义
    self_dim = 6  # 自身状态维度
    player_slot_dim = 9  # 每个附近玩家的观测维度
    ball_slot_dim = 4  # 每个奖励球的观测维度
    enemy_slot_dim = 9  # 每个敌人的观测维度
    velocity_dims = 2  # 速度维度 (vel_x, vel_y)
    valid_dims = 1 if use_valid_mask else 0  # 有效掩码维度

    # 如果不使用速度观测，从玩家和敌人维度中减去速度维度
    if not use_velocity_obs:
        player_slot_dim -= velocity_dims
        enemy_slot_dim -= velocity_dims

    # 计算总维度
    return (
        self_dim
        + 3 * (player_slot_dim + valid_dims)  # 3个附近玩家
        + 8 * (ball_slot_dim + valid_dims)    # 8个奖励球
        + 5 * (enemy_slot_dim + valid_dims)    # 5个敌人
        + ray_count                            # 射线检测
    )


def default_policy_spaces(config_path: str) -> Tuple[gym.Space, gym.Space]:
    """
    创建默认的观测空间和动作空间
    
    设计目的：
        RLlib 需要为每個策略定义观测空间和动作空间。
        此函数基于配置文件推断观测维度，并创建相应的 Gym 空间对象。
        
    观测空间设计：
        - 使用 Box 空间，范围 [-1, 1]，所有观测都已归一化
        - 维度由 infer_flat_obs_dim() 动态计算
        - 数据类型为 np.float32（RLlib 标准）
    
    动作空间设计：
        - 使用 Discrete(6) 表示 6 个离散动作
        - 动作映射：0=上, 1=下, 2=左, 3=右, 4=攻击, 5=待机
    
    参数：
        config_path: Godot 配置文件路径，用于推断观测维度
    
    返回：
        Tuple[gym.Space, gym.Space]: (observation_space, action_space)
    """
    obs_dim = infer_flat_obs_dim(config_path)
    observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(obs_dim,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(6)
    return observation_space, action_space


def _worker_index(config: Mapping[str, Any]) -> int:
    """
    从 RLlib 环境配置中提取工作器索引
    
    设计目的：
        RLlib 使用多个环境工作器并行收集数据。
        每个工作器需要不同的端口来避免冲突。
        此函数从配置中提取工作器索引，用于计算唯一的端口号。
        
    参数：
        config: RLlib 环境配置，可能包含 worker_index 属性或键
    
    返回：
        int: 工作器索引（从 0 开始）
    """
    return int(getattr(config, "worker_index", config.get("worker_index", 0)))


def _vector_index(config: Mapping[str, Any]) -> int:
    """
    从 RLlib 环境配置中提取向量化环境索引
    
    设计目的：
        每个工作器可以运行多个环境实例（向量化）。
        此函数提取向量化索引，用于进一步区分端口。
        
    参数：
        config: RLlib 环境配置
    
    返回：
        int: 向量化环境索引（从 0 开始）
    """
    return int(getattr(config, "vector_index", config.get("vector_index", 0)))


def _unwrap_single_obs_space(space: gym.Space) -> gym.Space:
    """
    从 Dict 观测空间中提取单个观测空间
    
    设计目的：
        Godot RL Agents 可能将观测空间包装为 Dict({"obs": space})。
        RLlib 多智能体环境需要每个智能体的独立观测空间。
        此函数解包嵌套的 Dict 空间，返回实际的观测空间。
        
    示例：
        输入: Dict("obs": Box(-1, 1, (142,)))
        输出: Box(-1, 1, (142,))
    
    参数：
        space: Gym 空间对象，可能是 Dict 或其他空间
    
    返回：
        gym.Space: 解包后的观测空间
    """
    if isinstance(space, gym.spaces.Dict) and set(space.spaces.keys()) == {"obs"}:
        return space.spaces["obs"]
    return space


def _unwrap_single_action_space(space: gym.Space) -> gym.Space:
    """
    从 Dict/Tuple 动作空间中提取单个动作空间
    
    设计目的：
        Godot RL Agents 可能将动作空间包装为 Dict 或 Tuple。
        此函数处理这些包装，返回实际的动作空间。
        
    处理逻辑：
        - Dict 且只有一个键：返回该键对应的空间
        - Tuple 且只有一个元素：返回该元素
        - 其他：返回原空间
    
    参数：
        space: Gym 空间对象
    
    返回：
        gym.Space: 解包后的动作空间
    """
    if isinstance(space, gym.spaces.Dict) and len(space.spaces) == 1:
        return next(iter(space.spaces.values()))
    if isinstance(space, gym.spaces.Tuple) and len(space.spaces) == 1:
        return space.spaces[0]
    return space


def _flatten_obs(raw_obs: Any) -> Any:
    """
    将可能的 Dict 格式观测转换为扁平数组
    
    设计目的：
        Godot RL Agents 可能返回 {"obs": np.array} 格式的观测。
        RLlib 策略网络通常需要直接的数组输入。
        此函数统一观测格式，确保输出为扁平的 numpy 数组。
        
    参数：
        raw_obs: 原始观测，可能是 {"obs": array} 或直接的 array
    
    返回：
        np.ndarray: 扁平的观测数组，dtype 为 float32
    """
    if isinstance(raw_obs, Mapping) and set(raw_obs.keys()) == {"obs"}:
        return np.asarray(raw_obs["obs"], dtype=np.float32)
    return raw_obs


def _normalise_single_action(raw_action: Any, original_action_space: gym.Space) -> List[Any]:
    """
    将 RLlib 策略输出的动作转换为 Godot 环境期望的格式
    
    设计目的：
        RLlib 策略输出的是离散动作的索引（int 或 np.ndarray）。
        Godot RL Agents 的 GodotEnv 期望动作是列表格式：[[action1], [action2], ...]。
        此函数处理各种可能的输入格式，确保输出符合 GodotEnv 的要求。
        
    格式转换示例：
        - Discrete 动作: 3 → [3]
        - np.ndarray (scalar): array(3) → [3]
        - np.ndarray (vector): array([1,2,3]) → [1, 2, 3]
        
    参数：
        raw_action: RLlib 策略输出的原始动作
        original_action_space: GodotEnv 的原始动作空间（用于判断格式）
    
    返回：
        List[Any]: GodotEnv 期望的动作列表格式
    """
    # 处理 Dict 动作空间（多动作空间）
    if isinstance(original_action_space, gym.spaces.Dict):
        keys = list(original_action_space.spaces.keys())
        if isinstance(raw_action, Mapping):
            # raw_action 是字典：{"action_key": action_value}
            return [raw_action[key] for key in keys]
        if len(keys) == 1:
            # 只有一个动作键：将标量动作包装为列表
            return [int(np.asarray(raw_action).item())]

    # 处理 Tuple 动作空间
    if isinstance(original_action_space, gym.spaces.Tuple):
        if len(original_action_space.spaces) == 1:
            # 只有一个动作：返回 [action]
            return [int(np.asarray(raw_action).item())]
        if isinstance(raw_action, np.ndarray):
            # 多个动作：转换为列表
            return raw_action.tolist()
        return list(raw_action)

    # 处理普通 Discrete 动作空间
    if isinstance(raw_action, np.ndarray) and raw_action.shape == ():
        return [int(raw_action.item())]
    if np.isscalar(raw_action):
        return [int(raw_action)]
    return list(raw_action)


class GodotMultiAgentEnv(MultiAgentEnv):
    """
    Godot 游戏环境与 RLlib 多智能体框架的适配层
    
    核心功能：
        将 Godot 游戏引擎封装为 RLlib 兼容的多智能体环境，使 RLlib 能够
        训练 Godot 中的多个 AIController 智能体。
        
    设计思路：
        1. 每个 Godot 进程可包含 N 个 AIController（对应 N 个智能体）
        2. GodotEnv 返回 list 格式数据：[obs_for_agent_0, obs_for_agent_1, ...]
        3. RLlib MultiAgentEnv 需要 dict 格式：{"player_0": obs, ...}
        4. 此类负责两种格式之间的转换
        
    端口分配策略（支持并行训练）：
        - 每个工作器（worker）需要独立的 Godot 进程
        - 每个 Godot 进程需要独立的通信端口
        - 端口计算公式：base_port + worker_index * port_stride + vector_index
        - 默认：base_port=11008, port_stride=100
        - 示例：worker=0, vector=0 → 端口 11008；worker=1, vector=0 → 端口 11108
        
    数据流向：
        RLlib策略 → action_dict → _normalise_single_action → GodotEnv.step() 
        → obs_list → _to_obs_dict → obs_dict → RLlib策略
        
    回合结束模式：
        - "any": 任一智能体结束 → 所有智能体结束（默认，适用于同步场景）
        - "all": 所有智能体都结束 → 才结束（严格同步）
        - "individual": 每个智能体独立结束（异步训练）
        
    属性说明：
        _env (GodotEnv): 管理与 Godot 进程的通信
        _num_godot_agents (int): Godot 中的智能体数量
        possible_agents (List[str]): 所有可能的智能体 ID（如 ["player_0", ...]）
        agents (List[str]): 当前存活的智能体 ID 列表
        _agent_to_index (Dict): 智能体 ID → GodotEnv 索引的映射
        _index_to_agent (Dict): GodotEnv 索引 → 智能体 ID 的映射
        observation_spaces (Dict): 每个智能体的观测空间
        action_spaces (Dict): 每个智能体的动作空间
        episode_done_mode (str): 回合结束模式
        max_episode_steps (int): 回合最大步数（防止无限循环）
        _episode_steps (int): 当前回合已执行的步数
    """

    def __init__(self, config: Optional[EnvContext] = None):
        """
        初始化 Godot 多智能体环境
        
        工作流程：
            1. 计算此环境实例的唯一端口号（避免多个实例冲突）
            2. 创建 GodotEnv 实例，启动 Godot 进程
            3. 读取 Godot 中的智能体数量
            4. 构建智能体 ID 与 GodotEnv 索引的双向映射
            5. 为每个智能体解包观测空间和动作空间
            
        参数：
            config: RLlib 环境配置（EnvContext），包含：
                - env_path (str): Godot 可执行文件路径（None 表示编辑器模式）
                - base_port (int): 基础端口号
                - port_stride (int): 端口步长
                - seed (int): 随机种子
                - show_window (bool): 是否显示 Godot 窗口
                - framerate (int): 帧率限制
                - action_repeat (int): 动作重复次数
                - speedup (int): 游戏速度倍数
                - episode_done_mode (str): 回合结束模式
                - max_episode_steps (int): 最大回合步数
        """
        super().__init__()
        config = config or {}

        # ===== 端口分配：确保每个环境实例使用唯一端口 =====
        # 计算公式：端口 = 基础端口 + 工作器偏移 + 向量化偏移
        # 这样不同的工作器和向量化环境实例不会端口冲突
        base_port = int(config.get("base_port", GodotEnv.DEFAULT_PORT))
        port_stride = int(config.get("port_stride", 100))
        port = base_port + _worker_index(config) * port_stride + _vector_index(config)
        
        # 为每个工作器分配不同的随机种子（确保训练多样性）
        # 不同工作器看到不同的随机序列，提高数据多样性
        seed = int(config.get("seed", 0)) + _worker_index(config)

        # ===== 回合管理参数 =====
        self.episode_done_mode = str(config.get("episode_done_mode", "any"))
        self.max_episode_steps = int(config.get("max_episode_steps", 512))
        self._episode_steps = 0  # 当前回合步数计数器
        
        # ===== 创建 GodotEnv 实例 =====
        # GodotEnv 是 godot-rl 库提供的环境接口，负责与 Godot 进程通信
        # convert_action_space=False: 保留原始动作空间，后续手动处理格式转换
        self._env = GodotEnv(
            env_path=str_to_optional_path(config.get("env_path")),
            port=port,
            seed=seed,
            show_window=bool(config.get("show_window", False)),
            framerate=config.get("framerate"),
            action_repeat=config.get("action_repeat"),
            speedup=config.get("speedup"),
            convert_action_space=False,  # 不自动转换动作空间，保持灵活性
        )

        # ===== 智能体管理 =====
        # GodotEnv.num_envs 表示 Godot 场景中的 AIController 数量
        self._num_godot_agents = int(self._env.num_envs)
        
        # 生成智能体 ID 列表：["player_0", "player_1", ...]
        # AGENT_PREFIX = "player_"，与 Godot 中的节点命名保持一致
        self.possible_agents = [f"{AGENT_PREFIX}{idx}" for idx in range(self._num_godot_agents)]
        self.agents = list(self.possible_agents)  # 当前存活的智能体（初始为全部）

        # ===== 构建双向映射 =====
        # 用途：在 GodotEnv 索引（0,1,2,3）和智能体 ID（player_0,...）之间转换
        self._agent_to_index = {agent_id: idx for idx, agent_id in enumerate(self.possible_agents)}
        self._index_to_agent = {idx: agent_id for agent_id, idx in self._agent_to_index.items()}

        # ===== 解包观测空间和动作空间 =====
        # GodotEnv 可能返回包装后的空间（Dict/Tuple），需要解包为 RLlib 期望的格式
        self.observation_spaces = {
            agent_id: _unwrap_single_obs_space(self._env.observation_spaces[idx])
            for agent_id, idx in self._agent_to_index.items()
        }
        self.action_spaces = {
            agent_id: _unwrap_single_action_space(self._env.action_spaces[idx])
            for agent_id, idx in self._agent_to_index.items()
        }

        # ===== 兼容 RLlib 单智能体接口 =====
        # 假设所有智能体共享相同的空间维度（Tiny Swords 确实是这样）
        self.observation_space = self.observation_spaces[self.possible_agents[0]]
        self.action_space = self.action_spaces[self.possible_agents[0]]

        # 打印环境初始化信息（用于调试和确认配置）
        print(
            "[GodotMultiAgentEnv] "
            f"port={port} seed={seed} agents={self.possible_agents} "
            f"policy_names={getattr(self._env, 'agent_policy_names', None)}"
        )

    def get_agent_ids(self) -> set:
        """
        返回所有可能的智能体 ID（RLlib MultiAgentEnv 接口要求）
        
        返回：
            set: 所有可能的智能体 ID 集合（如 {"player_0", "player_1", ...}）
            
        注：
            RLlib 使用此方法了解环境中存在哪些智能体。
            与 self.agents 不同，此方法返回的是"可能"的智能体，而不是"当前存活"的。
        """
        return set(self.possible_agents)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, dict]]:
        """
        重置环境，开始新的回合
        
        工作流程：
            1. 重置存活智能体列表（所有智能体重新参与）
            2. 重置步数计数器
            3. 调用 GodotEnv.reset() 重置 Godot 游戏状态
            4. 将 list 格式转换为 dict 格式
            
        参数：
            seed: 随机种子（未使用，Godot 的种子在 __init__ 中设置）
            options: 额外选项（未使用，RLlib 接口要求）
        
        返回：
            Tuple[obs_dict, info_dict]:
                - obs_dict: {agent_id: observation} 字典
                - info_dict: {agent_id: info} 字典（附加信息，如调试数据）
        """
        del seed, options  # RLlib 接口要求，但此处未使用
        self.agents = list(self.possible_agents)  # 重置存活智能体列表
        self._episode_steps = 0  # 重置步数计数器
        obs_list, info_list = self._env.reset()  # 调用 GodotEnv.reset()
        return self._to_obs_dict(obs_list), self._to_info_dict(info_list)

    def step(
        self,
        action_dict: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Any],      # obs: 下一步的观测
        Dict[str, float],     # rewards: 此步的奖励
        Dict[str, bool],     # terminated: 是否终止（如死亡）
        Dict[str, bool],     # truncated: 是否截断（如超时）
        Dict[str, dict],     # info: 附加信息
    ]:
        """
        执行一步环境仿真（核心方法）
        
        工作流程：
            1. 验证 action_dict 包含所有存活智能体的动作
            2. 将 RLlib 的 action_dict 转换为 GodotEnv 期望的 ordered_actions
            3. 调用 GodotEnv.step() 执行仿真
            4. 将返回数据从 list 格式转换为 dict 格式
            5. 根据 episode_done_mode 处理回合结束逻辑
            6. 检查最大步数限制
            7. 返回多智能体格式的结果
            
        参数：
            action_dict: {agent_id: action} 字典，包含每个智能体的动作
            
        返回：
            Tuple[obs_dict, reward_dict, term_dict, trunc_dict, info_dict]
                - obs_dict: 下一步的观测
                - reward_dict: 此步的奖励
                - term_dict: 是否终止（episode 自然结束，如智能体死亡）
                - trunc_dict: 是否截断（episode 被外部截断，如达到最大步数）
                - info_dict: 附加信息
                
        注：
            RLlib 使用 terminated/truncated 区分两种 episode 结束方式：
            - terminated: 环境自然结束（如智能体死亡、任务完成）
            - truncated: 外部截断（如达到最大步数、人为中断）
        """
        ordered_actions = []
        # 按 possible_agents 顺序构建动作列表（确保与 GodotEnv 索引对齐）
        for agent_id in self.possible_agents:
            if agent_id not in action_dict:
                raise KeyError(f"Missing action for live agent {agent_id}")
            idx = self._agent_to_index[agent_id]
            # 将 RLlib 策略输出的动作转换为 GodotEnv 期望的格式
            ordered_actions.append(
                _normalise_single_action(action_dict[agent_id], self._env.action_spaces[idx])
            )

        # ===== 调用 Godot 环境执行一步仿真 =====
        # order_ij=True: 动作按 [agent_idx][action] 组织
        # 返回：obs_list, rewards, terms, truncs, infos（均为 list 格式）
        obs_list, rewards, terms, truncs, infos = self._env.step(
            np.asarray(ordered_actions, dtype=object),
            order_ij=True,
        )
        self._episode_steps += 1

        # ===== 将 list 格式转换为 dict 格式 =====
        obs = self._to_obs_dict(obs_list)
        reward_dict = {
            self._index_to_agent[idx]: float(rewards[idx])
            for idx in range(self._num_godot_agents)
        }
        term_dict = {
            self._index_to_agent[idx]: bool(terms[idx])
            for idx in range(self._num_godot_agents)
        }
        trunc_dict = {
            self._index_to_agent[idx]: bool(truncs[idx])
            for idx in range(self._num_godot_agents)
        }

        # ===== 回合结束逻辑 =====
        # 根据 episode_done_mode 处理回合结束信号
        if self.episode_done_mode == "any":
            # 模式 "any": 任一智能体结束 → 所有智能体结束
            # 用途：集中式终止，适用于所有智能体必须同时结束的场景
            # 示例：4人混战，任一死亡则重置整个场景
            done_all = any(term_dict.values()) or any(trunc_dict.values())
            if done_all:
                # 强制所有智能体结束
                term_dict = {agent_id: True for agent_id in self.possible_agents}
                trunc_dict = {agent_id: False for agent_id in self.possible_agents}
            terminated_all = done_all
            truncated_all = False
            
        elif self.episode_done_mode == "all":
            # 模式 "all": 所有智能体都结束 → 才结束
            # 用途：严格同步，所有智能体必须完成各自的 episode
            terminated_all = all(term_dict.values())
            truncated_all = all(trunc_dict.values())
            done_all = terminated_all or truncated_all
            
        elif self.episode_done_mode == "individual":
            # 模式 "individual": 每个智能体独立结束
            # 用途：异步训练，智能体可以有不同的 episode 长度
            # 注意：此模式需要 RLlib 支持异步 episode（较新版本支持）
            terminated_all = all(term_dict.values())
            truncated_all = all(trunc_dict.values())
            done_all = terminated_all or truncated_all
        else:
            raise ValueError(
                "--episode-done-mode must be one of: any, all, individual"
            )

        # ===== 检查最大步数限制 =====
        # 这是一个安全机制，防止智能体永远不结束 episode（如卡在角落）
        if not done_all and self.max_episode_steps > 0 and self._episode_steps >= self.max_episode_steps:
            done_all = True
            terminated_all = False  # 不是自然终止
            truncated_all = True   # 而是被截断（超时）
            trunc_dict = {agent_id: True for agent_id in self.possible_agents}

        # ===== RLlib 特殊键 "__all__" =====
        # RLlib 使用 "__all__" 键来表示整个回合的状态
        term_dict["__all__"] = bool(terminated_all)
        trunc_dict["__all__"] = bool(truncated_all)
        
        # 如果回合结束，清空存活智能体列表
        if done_all:
            self.agents = []

        return obs, reward_dict, term_dict, trunc_dict, self._to_info_dict(infos)

    def close(self) -> None:
        """
        关闭环境，释放 Godot 进程资源
        
        用途：
            当训练结束或需要重启环境时，调用此方法来：
            1. 关闭 Godot 进程
            2. 释放网络通信端口
            3. 清理临时文件（如果有）
            
        注：
            此方法由 RLlib 自动调用，也可以手动调用。
            如果不调用此方法，Godot 进程可能成为僵尸进程。
        """
        self._env.close()

    def _to_obs_dict(self, obs_list: Iterable[Any]) -> Dict[str, Any]:
        """
        将 GodotEnv 的观测列表转换为 RLlib 的观测字典
        
        格式转换：
            GodotEnv 返回: [obs_for_agent_0, obs_for_agent_1, ...] (list)
            RLlib 期望: {"player_0": obs, "player_1": obs, ...} (dict)
            
        参数：
            obs_list: 观测列表，每个元素对应一个智能体的观测
            
        返回：
            Dict[str, Any]: {"player_0": obs, "player_1": obs, ...}
            
        注：
            此方法还会调用 _flatten_obs() 来统一观测格式，
            确保输出为扁平的 numpy 数组（dtype=float32）。
        """
        return {
            self._index_to_agent[idx]: _flatten_obs(obs)
            for idx, obs in enumerate(obs_list)
        }

    def _to_info_dict(self, info_list: Iterable[dict]) -> Dict[str, dict]:
        """
        将 GodotEnv 的信息列表转换为 RLlib 的信息字典
        
        用途：
            info 字典包含附加信息，如：
            - 调试数据
            - 奖励分解
            - 内部状态信息
            
        参数：
            info_list: 信息列表，每个元素对应一个智能体的信息字典
            
        返回：
            Dict[str, dict]: {"player_0": info, "player_1": info, ...}
            
        注：
            如果某个智能体的 info 为 None，则使用空字典 {}。
        """
        return {
            self._index_to_agent[idx]: dict(info or {})
            for idx, info in enumerate(info_list)
        }


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器
    """
    parser = argparse.ArgumentParser(
        description="Train independent multi-agent PPO policies with RLlib and Godot RL Agents."
    )
    # ===== 环境相关参数 =====
    # Godot 可执行文件路径（None 表示使用编辑器模式）
    parser.add_argument("--env-path", type=str_to_optional_path, default="godot-game/build/game.exe")
    # Godot 配置文件路径（用于推断观测空间维度）
    parser.add_argument("--config-path", type=str, default="godot-game/configs/game_config.tres")
    # 实验输出目录
    parser.add_argument("--experiment-dir", type=str, default="logs/rllib_multiagent_ppo")
    # 实验名称（用于区分不同的训练运行）
    parser.add_argument("--experiment-name", type=str, default=f"ppo_ma_{time.strftime('%m%d-%H%M')}")
    # 从哪个检查点恢复训练（用于继续之前的训练）
    parser.add_argument("--restore", type=str, default=None)

    # ===== 多智能体相关参数 =====
    # 智能体数量（对应 Godot 场景中的 AIController 数量）
    parser.add_argument("--num-agents", type=int, default=4)
    # 是否使用共享策略（所有智能体共享同一个策略网络）
    parser.add_argument("--shared-policy", action="store_true")
    # 回合结束模式（any/all/individual）
    parser.add_argument("--episode-done-mode", choices=["any", "all", "individual"], default="any")
    # 每个回合的最大步数（防止无限循环）
    parser.add_argument("--max-episode-steps", type=int, default=512)

    # ===== 环境运行参数 =====
    # 随机种子（用于复现实验结果）
    parser.add_argument("--seed", type=int, default=1)
    # Godot 通信基础端口
    parser.add_argument("--base-port", type=int, default=GodotEnv.DEFAULT_PORT)
    # 端口步长（用于并行训练时分配不同端口）
    parser.add_argument("--port-stride", type=int, default=100)
    # 是否显示 Godot 窗口（调试时有用，训练时建议关闭）
    parser.add_argument("--show-window", action="store_true", default=True)
    # 游戏速度倍数（加速训练）
    parser.add_argument("--speedup", type=int, default=8)
    # 帧率限制（None 表示不限制）
    parser.add_argument("--framerate", type=int, default=None)
    # 动作重复次数（每个动作执行多少步）
    parser.add_argument("--action-repeat", type=int, default=None)

    # ===== 训练参数 =====
    # 训练迭代次数
    parser.add_argument("--iterations", type=int, default=200)
    # 停止训练的总步数（None 表示不限制）
    parser.add_argument("--stop-timesteps", type=int, default=None)
    # 检查点保存频率（每 N 次迭代保存一次）
    parser.add_argument("--checkpoint-freq", type=int, default=10)

    # ===== 环境运行器参数 =====
    # 环境运行器数量（并行环境数量）
    parser.add_argument("--num-env-runners", type=int, default=1)
    # 每个运行器的环境数量
    parser.add_argument("--num-envs-per-env-runner", type=int, default=1)
    # GPU 使用数量（0 表示使用 CPU）
    parser.add_argument("--num-gpus", type=float, default=0.0)
    # 深度学习框架（torch 或 tf2）
    parser.add_argument("--framework", choices=["torch", "tf2"], default="torch")

    # ===== PPO 超参数 =====
    # 学习率
    parser.add_argument("--lr", type=float, default=3e-4)
    # 折扣因子（gamma）
    parser.add_argument("--gamma", type=float, default=0.99)
    # GAE lambda 参数
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    # PPO 裁剪参数（epsilon）
    parser.add_argument("--clip-param", type=float, default=0.2)
    # 熵系数（鼓励探索）
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    # 价值函数损失系数
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    # 训练批次大小
    parser.add_argument("--train-batch-size", type=int, default=4096)
    # 小批次大小（用于 SGD 更新）
    parser.add_argument("--minibatch-size", type=int, default=256)
    # 每个更新的训练轮数（epoch）
    parser.add_argument("--num-epochs", type=int, default=10)
    # 收集片段长度
    parser.add_argument("--rollout-fragment-length", type=int, default=128)
    # 全连接隐藏层维度
    parser.add_argument("--fcnet-hiddens", type=int, nargs="+", default=[256, 256])
    # 禁用环境检查（在使用 Godot 时建议启用）
    parser.add_argument("--disable-env-checking", action="store_true")
    # 是否使用新的 API 栈（RLlib 2.0+）
    parser.add_argument("--use-new-api-stack", action="store_true")

    return parser


def make_policy_specs(args: argparse.Namespace) -> Dict[str, PolicySpec]:
    """
    创建策略规格字典（PolicySpec）
    
    功能：
        根据命令行参数创建策略规格，定义每个策略的：
        - policy_class: 策略类（None 表示使用默认 PPO 策略）
        - observation_space: 观测空间
        - action_space: 动作空间
        - config: 策略特定配置（此处为空）
        
    策略模式：
        - 独立策略模式（默认）：每个智能体有独立的策略
            {"player_0_policy": PolicySpec(...), "player_1_policy": ...}
        - 共享策略模式（--shared-policy）：所有智能体共享一个策略
            {"shared_policy": PolicySpec(...)}
            
    参数：
        args: 命令行参数（包含 num_agents, shared_policy, config_path 等）
        
    返回：
        Dict[str, PolicySpec]: 策略 ID → 策略规格的映射
        
    注：
        所有策略共享相同的观测空间和动作空间（因为所有智能体是同质的）。
    """
    # 推断观测空间和动作空间（基于配置文件）
    observation_space, action_space = default_policy_spaces(args.config_path)
    
    # 根据策略模式决定策略 ID 列表
    policy_ids = (
        ["shared_policy"]
        if args.shared_policy
        else [f"{AGENT_PREFIX}{idx}_policy" for idx in range(args.num_agents)]
    )
    
    # 创建策略规格字典
    return {
        policy_id: PolicySpec(
            policy_class=None,  # None 表示使用 PPO 默认策略类
            observation_space=observation_space,
            action_space=action_space,
            config={},  # 策略特定配置（此处不需要）
        )
        for policy_id in policy_ids
    }


def make_policy_mapping_fn(shared_policy: bool):
    """
    创建策略映射函数
    
    功能：
        返回一个函数，用于将智能体 ID 映射到策略 ID。
        这是 RLlib 多智能体训练的核心：决定哪个智能体使用哪个策略。
        
    映射逻辑：
        - 共享策略模式：所有智能体 → "shared_policy"
        - 独立策略模式：player_0 → "player_0_policy", ...
        
    参数：
        shared_policy (bool): 是否使用共享策略
        
    返回：
        Callable: 策略映射函数 policy_mapping_fn(agent_id) -> policy_id
        
    示例：
        # 独立策略模式
        fn = make_policy_mapping_fn(shared_policy=False)
        fn("player_0")  # 返回 "player_0_policy"
        
        # 共享策略模式
        fn = make_policy_mapping_fn(shared_policy=True)
        fn("player_0")  # 返回 "shared_policy"
        fn("player_1")  # 返回 "shared_policy"
    """
    def policy_mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
        """
        策略映射函数（实际执行映射）
        
        参数：
            agent_id (str): 智能体 ID（如 "player_0"）
            *args, **kwargs: RLlib 可能传递的额外参数（忽略）
            
        返回：
            str: 策略 ID（如 "player_0_policy" 或 "shared_policy"）
        """
        del args, kwargs  # RLlib 接口要求，此处未使用
        if shared_policy:
            return "shared_policy"
        return f"{agent_id}_policy"

    return policy_mapping_fn


def build_config(args: argparse.Namespace) -> PPOConfig:
    """
    构建 RLlib PPO 训练配置
    
    功能：
        根据命令行参数构建完整的 PPO 训练配置，包括：
        1. 环境配置（Godot 环境）
        2. 框架配置（PyTorch/TensorFlow）
        3. 资源配置（GPU 分配）
        4. 环境运行器配置（并行环境）
        5. 训练超参数配置
        6. 多智能体配置（策略定义和映射）
        
    配置流程：
        1. 注册环境（将 ENV_NAME 映射到 GodotMultiAgentEnv）
        2. 使用 PPOConfig() 流式 API 构建配置
        3. 根据 args.use_new_api_stack 决定是否使用新 API 栈
        
    参数：
        args: 命令行参数（包含所有训练配置）
        
    返回：
        PPOConfig: 配置好的 PPO 训练配置
        
    注：
        - PPOConfig 是 RLlib 2.0+ 的新配置方式（流式 API）
        - 旧版使用 dict 配置，现已弃用
        - api_stack() 控制是否使用新的 RL Module 和 Learner API
    """
    # ===== 注册环境 =====
    # 将环境名称 "godot_multiagent_independent" 映射到 GodotMultiAgentEnv 类
    # RLlib 会根据需要创建环境实例
    register_env(ENV_NAME, lambda env_config: GodotMultiAgentEnv(env_config))

    # ===== 构建 PPO 配置（流式 API）=====
    config = (
        PPOConfig()
        
        # --- 环境配置 ---
        .environment(
            env=ENV_NAME,  # 使用注册的环境名称
            env_config={  # 传递给 GodotMultiAgentEnv 的配置
                "env_path": args.env_path,  # Godot 可执行文件路径
                "seed": args.seed,  # 随机种子
                "base_port": args.base_port,  # 基础端口
                "port_stride": args.port_stride,  # 端口步长
                "show_window": args.show_window,  # 是否显示窗口
                "speedup": args.speedup,  # 游戏速度倍数
                "framerate": args.framerate,  # 帧率限制
                "action_repeat": args.action_repeat,  # 动作重复次数
                "episode_done_mode": args.episode_done_mode,  # 回合结束模式
                "max_episode_steps": args.max_episode_steps,  # 最大回合步数
            },
            disable_env_checking=args.disable_env_checking,  # 禁用环境检查（Godot 需要）
        )
        
        # --- 框架配置 ---
        .framework(args.framework)  # "torch" 或 "tf2"
        
        # --- 资源配置 ---
        .resources(num_gpus=args.num_gpus)  # GPU 数量（0 表示使用 CPU）
        
        # --- 环境运行器配置 ---
        .env_runners(
            num_env_runners=args.num_env_runners,  # 环境运行器数量
            num_envs_per_env_runner=args.num_envs_per_env_runner,  # 每个运行器的环境数量
            rollout_fragment_length=args.rollout_fragment_length,  # 收集片段长度
            batch_mode="truncate_episodes",  # 批次模式：截断 episode
        )
        
        # --- 训练超参数 ---
        .training(
            lr=args.lr,  # 学习率
            gamma=args.gamma,  # 折扣因子
            lambda_=args.gae_lambda,  # GAE lambda
            clip_param=args.clip_param,  # PPO 裁剪参数
            entropy_coeff=args.entropy_coeff,  # 熵系数
            vf_loss_coeff=args.vf_loss_coeff,  # 价值函数损失系数
            train_batch_size=args.train_batch_size,  # 训练批次大小
            minibatch_size=args.minibatch_size,  # 小批次大小
            num_epochs=args.num_epochs,  # 每个更新的训练轮数
            model={  # 策略网络模型配置
                "fcnet_hiddens": args.fcnet_hiddens,  # 隐藏层维度
                "fcnet_activation": "tanh",  # 激活函数
            },
        )
        
        # --- 多智能体配置 ---
        .multi_agent(
            policies=make_policy_specs(args),  # 策略规格字典
            policy_mapping_fn=make_policy_mapping_fn(args.shared_policy),  # 策略映射函数
            count_steps_by="agent_steps",  # 步数统计方式（按智能体步数）
        )
    )

    # ===== 新/旧 API 栈切换 =====
    # RLlib 2.0 引入了新 API 栈（RL Module + Learner）
    # 如果不需要新特性，可以禁用以兼容旧代码
    if not args.use_new_api_stack:
        config = config.api_stack(
            enable_rl_module_and_learner=False,  # 禁用 RL Module
            enable_env_runner_and_connector_v2=False,  # 禁用新环境运行器
        )

    return config


def result_metric(result: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """
    从嵌套字典中安全提取指标值
    
    功能：
        从 RLlib 的训练结果字典中按路径提取指标值。
        RLlib 的结果字典是嵌套的（如 result["env_runners"]["episode_return_mean"]）。
        此函数安全地遍历路径，如果任何中间键不存在则返回默认值。
        
    参数：
        result (Mapping): RLlib 的训练结果字典
        *keys (str): 键路径（如 "env_runners", "episode_return_mean"）
        default: 默认值（如果路径不存在）
        
    返回：
        Any: 提取的值，或默认值
        
    示例：
        result = {"env_runners": {"episode_return_mean": 10.5}}
        result_metric(result, "env_runners", "episode_return_mean")  # 返回 10.5
        result_metric(result, "foo", "bar")  # 返回 None（路径不存在）
        
    用途：
        兼容不同版本的 RLlib，因为指标路径可能变化。
        例如：
        - 旧版：result["sampler_results"]["episode_reward_mean"]
        - 新版：result["env_runners"]["episode_return_mean"]
    """
    cursor: Any = result
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def print_iteration(iteration: int, result: Mapping[str, Any]) -> None:
    """
    打印训练迭代的摘要信息
    
    功能：
        从 RLlib 的训练结果中提取关键指标并打印，包括：
        - 平均奖励（reward_mean）
        - 环境步数（env_steps）
        - 智能体步数（agent_steps）
        - 总回合数（episodes）
        
        此函数兼容不同版本的 RLlib，通过尝试多个可能的指标路径。
        
    参数：
        iteration (int): 当前迭代次数
        result (Mapping): RLlib 的训练结果字典
        
    打印格式：
        [Iter 0010] reward_mean=10.5 env_steps=10000 agent_steps=40000 episodes=50
        
    注：
        如果训练刚开始还没有完成的 episode，reward_mean 会是 NaN，
        此时会显示为 "nan(no completed episodes yet)"。
    """
    # ===== 提取平均奖励 =====
    # 尝试多个可能的路径（兼容不同版本的 RLlib）
    reward_mean = (
        result.get("episode_reward_mean")  # 常见路径 1
        or result_metric(result, "env_runners", "episode_return_mean")  # 新版路径
        or result_metric(result, "sampler_results", "episode_reward_mean")  # 旧版路径
    )
    
    # ===== 提取环境步数 =====
    timesteps = (
        result.get("timesteps_total")  # 常见路径 1
        or result.get("num_env_steps_sampled_lifetime")  # 旧版路径
        or result_metric(result, "env_runners", "num_env_steps_sampled_lifetime")  # 新版路径
    )
    
    # ===== 提取智能体步数 =====
    agent_steps = (
        result.get("agent_timesteps_total")  # 常见路径 1
        or result.get("num_agent_steps_sampled_lifetime")  # 旧版路径
        or result_metric(result, "env_runners", "num_agent_steps_sampled_lifetime")  # 新版路径
    )
    
    # ===== 提取总回合数 =====
    episodes_total = (
        result.get("episodes_total")  # 常见路径 1
        or result_metric(result, "env_runners", "num_episodes_lifetime")  # 新版路径
    )
    
    # ===== 处理 NaN =====
    # 如果还没有完成的 episode，reward_mean 会是 NaN
    if isinstance(reward_mean, float) and math.isnan(reward_mean):
        reward_mean = "nan(no completed episodes yet)"
    
    # ===== 打印摘要 =====
    print(
        f"[Iter {iteration:04d}] "
        f"reward_mean={reward_mean} env_steps={timesteps} "
        f"agent_steps={agent_steps} episodes={episodes_total}"
    )


def main() -> None:
    """
    训练脚本的主入口点
    错误处理：
        - KeyboardInterrupt: 用户按 Ctrl+C，保存当前检查点后退出
        - 其他异常: 由 Python 默认异常处理
        
    参数修正逻辑：
        - num_env_runners <= 0 → 强制设为 1（Godot 不支持 0 个环境）
        - show_window + num_env_runners > 1 → 强制 num_env_runners=1（避免多个窗口）
        - show_window → 自动启用 disable_env_checking（避免额外验证环境）
        
    检查点策略：
        - 每 N 次迭代保存一次（由 --checkpoint-freq 控制）
        - 训练结束后保存最终检查点
        - 支持从检查点恢复（--restore）
        
    示例用法：
        # 基本训练
        python rllib_multiagent_ppo.py --iterations 100
        
        # 从检查点恢复
        python rllib_multiagent_ppo.py --restore logs/checkpoint_000100
        
        # 使用共享策略
        python rllib_multiagent_ppo.py --shared-policy
    """
    # ===== 1. 解析命令行参数 =====
    args = build_arg_parser().parse_args()
    
    # ===== 2. 验证和修正参数 =====
    if args.num_env_runners <= 0:
        print("[Config] --num-env-runners must be >= 1 for this Godot/RLlib setup; using 1.")
        args.num_env_runners = 1
        
    # Godot 不支持同时显示多个窗口，强制单环境
    if args.show_window and args.num_env_runners > 1:
        print(
            "[Config] --show-window is limited to one Godot instance; "
            "forcing --num-env-runners=1 to avoid duplicate windows."
        )
        args.num_env_runners = 1
        
    # 显示窗口时，禁用环境检查（避免 RLlib 创建额外的验证环境）
    if args.show_window and not args.disable_env_checking:
        print("[Config] --show-window enables --disable-env-checking to avoid extra validation envs.")
        args.disable_env_checking = True
    
    # 创建实验目录（如果不存在）
    pathlib.Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)

    # ===== 3. 打印配置摘要 =====
    print("[Config] independent policies:", not args.shared_policy)
    print("[Config] inferred obs dim:", infer_flat_obs_dim(args.config_path))
    print("[Config] env path:", args.env_path or "Godot editor")

    # ===== 4. 初始化 Ray =====
    # ignore_reinit_error=True: 如果 Ray 已初始化，不报错
    # include_dashboard=False: 不启动 Ray Dashboard（节省资源）
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    algorithm = None
    last_checkpoint: Optional[str] = None
    
    try:
        # ===== 5. 构建 PPO 配置 =====
        config = build_config(args)
        
        # ===== 6. 创建训练算法实例 =====
        algorithm = config.build()

        # ===== 7. （可选）从检查点恢复 =====
        if args.restore:
            print("[Restore]", os.path.abspath(args.restore))
            algorithm.restore(args.restore)

        # ===== 8. 训练循环 =====
        for iteration in range(1, args.iterations + 1):
            # 执行一次训练迭代
            result = algorithm.train()
            
            # 打印迭代摘要
            print_iteration(iteration, result)

            # 提取总步数（用于判断是否达到停止条件）
            timesteps_total = (
                result.get("timesteps_total")
                or result.get("num_env_steps_sampled_lifetime")
                or 0
            )
            
            # ===== 9. 定期保存检查点 =====
            should_checkpoint = (
                args.checkpoint_freq > 0
                and iteration % args.checkpoint_freq == 0
            )
            if should_checkpoint:
                last_checkpoint = algorithm.save(args.experiment_dir)
                print("[Checkpoint]", last_checkpoint)

            # ===== 10. 检查是否达到停止条件 =====
            if args.stop_timesteps is not None and timesteps_total >= args.stop_timesteps:
                print(f"[Stop] reached --stop-timesteps={args.stop_timesteps}")
                break

        # ===== 11. 保存最终检查点 =====
        final_checkpoint = algorithm.save(args.experiment_dir)
        print("[Final checkpoint]", final_checkpoint)
        if last_checkpoint and last_checkpoint != final_checkpoint:
            print("[Previous checkpoint]", last_checkpoint)
            
    except KeyboardInterrupt:
        # ===== 用户中断处理（Ctrl+C）=====
        print("[Interrupted] Saving checkpoint before shutdown.")
        if algorithm is not None:
            checkpoint = algorithm.save(args.experiment_dir)
            print("[Interrupted checkpoint]", checkpoint)
    finally:
        # ===== 清理资源 =====
        if algorithm is not None:
            algorithm.stop()  # 停止训练算法
        ray.shutdown()  # 关闭 Ray


if __name__ == "__main__":
    main()
