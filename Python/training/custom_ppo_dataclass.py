from dataclasses import dataclass, field
from enum import Enum
from math import fabs
from typing import Optional, Union

import numpy as np
import torch


class NetworkType(str, Enum):
    """Supported feature extractors for PPO/IPPO policies."""
    SEGMENTED_MLP = "segmented_mlp"
    MLP = "mlp"
    GRU_MLP = "gru_mlp"

@dataclass
class PPOArgs:
    """Single-agent PPO training config."""
    # Environment
    env_path: Optional[str] = "curriculum_envs/s3-enemy-and-ball/build3/game.exe"
    """游戏环境路径（Godot 可执行文件）"""
    config_path: str = "godot-game/configs/game_config.tres"
    """游戏配置文件路径（.tres）"""
    n_parallel: int = 4
    """并行环境（智能体）数量"""
    seed: int = 1
    """随机种子"""
    show_window: bool = False
    """是否显示游戏窗口"""
    speedup: int = 16
    """游戏物理加速倍数"""
    port_offset: int = 0
    """Godot 通信端口偏移量。基础端口 11008，实际端口 = 11008 + port_offset。
    多进程并行训练时，每个进程设置不同偏移量避免端口冲突（如 0, 100, 200, 300）。"""

    # Logging/checkpointing
    exp_name: str = "custom_ppo"
    """实验名称（用于 TensorBoard / WandB 显示）"""
    experiment_dir: str = "logs/cleanrl_ppo"
    """实验日志根目录"""
    save_model_path: Optional[str] = "saved_models/mlp_p1_s1"
    """最终模型保存路径（.pt），设为 None 则不保存最终模型"""
    save_checkpoint: bool = True
    """是否在训练中周期性保存中断点"""
    resume_from: Optional[str] = None
    # resume_from:Optional[str]="saved_models/mlp_p1_s1_episode280.pt"
    """从中断点恢复训练的路径（设为 None 不恢复）"""
    load_model_path: Optional[str] = None
    """加载已有模型权重但不恢复优化器/计数器"""
    save_every_n_episodes: int = 10
    """每 N 个 episode 保存一次中断点"""
    max_checkpoints: int = 3
    """最多保留多少个中断点文件，超出则删除最旧的"""
    track: bool = False
    """是否记录到 WandB"""
    wandb_project_name: str = "cleanRL_ppo"
    """WandB 项目名称"""
    wandb_entity: Optional[str] = "lunjijiang-rl"
    """WandB 用户 / 团队名"""

    # PPO hyperparameters
    total_timesteps: int = 1_000_000
    """训练总时间步数,多环境并行时加速消耗"""
    learning_rate: float = 0.0007221131969041414
    """学习率"""
    num_steps: int = 256
    """每次 rollout 采集的步数（每个并行环境）"""
    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE 的 lambda 参数"""
    num_minibatches: int = 4
    """将 rollout 数据分成多少个小批量"""
    update_epochs: int = 5
    """每个 rollout 的更新轮数"""
    recurrent_seq_len: int = 128
    """循环网络（GRU）的序列截断长度"""
    clip_coef: float = 0.2
    """PPO 裁剪系数 epsilon"""
    ent_coef: float = 0.002
    """策略熵损失系数"""
    vf_coef: float = 0.5
    """价值函数损失系数"""
    max_grad_norm: float = 4.0
    """梯度裁剪阈值"""
    norm_adv: bool = True
    """是否对优势函数标准化"""
    clip_vloss: bool = True
    """是否裁剪价值函数损失"""
    anneal_lr: bool = True
    """是否对学习率退火（随训练进度线性衰减）"""
    target_kl: Optional[float] = None
    """早停目标 KL 散度阈值（None 则禁用早停）"""
    torch_deterministic: bool = True
    """是否使用确定性算法（保证复现性）"""
    cuda: bool = True
    """是否使用 CUDA 加速"""
    reward_norm: bool = True
    """是否启用奖励归一化"""
    reward_clip: float = 1.0
    """Reward normalization clipping range."""

    # Optuna hyperparameter tuning
    enable_optuna: bool = False
    """Enable Optuna hyperparameter search instead of one normal training run."""
    optuna_trials: int = 100
    """Number of Optuna trials to run."""
    optuna_timesteps: int = 500_000
    """Training timesteps used by each trial."""
    optuna_study_name: str = "custom_ppo_optuna"
    """Optuna study name."""
    optuna_storage: Optional[str] = "sqlite:///logs/optuna/custom_ppo.db"
    # optuna_storage: Optional[str] = None
    """Optuna storage URI (sqlite for persistence across crashes)."""
    optuna_best_params_path: Optional[str] = "logs/optuna/custom_ppo_best_params.json"
    """Where to write the best trial parameters after tuning."""
    optuna_prune: bool = True
    """Allow Optuna to prune clearly weak trials during training."""

    # Network
    network_type: NetworkType = NetworkType.MLP
    """网络类型"""

    #seg mlp
    self_hidden: int = 32
    """SELF 段 MLP 隐藏层宽度"""
    player_hidden: int = 64
    """PLAYER 段 MLP 隐藏层宽度"""
    ball_hidden: int = 64
    """BALL 段 MLP 隐藏层宽度"""
    enemy_hidden: int = 64
    """ENEMY 段 MLP 隐藏层宽度"""
    map_hidden: int = 64
    """MAP 段 MLP 隐藏层宽度"""
    seg_trunk_hiddens: tuple = (196, 64)
    """SEGMENTED_MLP 躯干层宽度（各段输出拼接后 -> 躯干 -> 融合特征）"""

    gru_trunk_hiddens: tuple = (128, 64)
    """GRU_MLP 躯干层宽度（GRU 输出 + BALL 特征 融合后 -> 躯干 -> 融合特征）"""

    # MLP
    mlp_hiddens: tuple = (256,256,128)
    """MLP 主干网络各层宽度（直接拼接观测 -> MLP 输出）"""

    # GRU
    gru_hidden: int = 128
    """GRU 隐藏单元数"""
    gru_num_layers: int = 1
    """GRU 层数（gru_mlp 用，>=1）"""
    gru_input_layernorm: bool = True
    """GRU 输入前是否加 LayerNorm"""

    # Runtime-derived values
    num_envs: int = 0
    """并行环境（智能体）数"""
    batch_size: int = 0
    """每次采样时总的样本数"""
    minibatch_size: int = 0
    """小批量样本数"""

@dataclass
class AgentConfig:
    agent_id: int
    """智能体编号（0~num_agents-1）"""
    train: bool = True
    """是否训练该智能体"""
    act_when_not_training: bool = False
    """train=False 时是否仍使用该 agent 的策略动作；False 则使用随机动作"""

    network_type: NetworkType = NetworkType.MLP
    """网络类型"""

    # Segmented MLP
    self_hidden: int = 32
    """SELF 段隐藏层宽度"""
    player_hidden: int = 64
    """PLAYER 段隐藏层宽度"""
    ball_hidden: int = 64
    """BALL 段隐藏层宽度"""
    enemy_hidden: int = 64
    """ENEMY 段隐藏层宽度"""
    map_hidden: int = 64
    """MAP 段隐藏层宽度"""
    seg_trunk_hiddens: tuple = (196, 64)
    """SEGMENTED_MLP 躯干层宽度"""

    # MLP
    mlp_hiddens: tuple = (256, 256, 128)
    """主干 MLP 各层宽度"""

    # GRU
    gru_hidden: int = 128
    """GRU 隐藏单元数"""
    gru_num_layers: int = 1
    """GRU 层数"""
    gru_input_layernorm: bool = True
    """GRU 输入前是否加 LayerNorm"""
    gru_trunk_hiddens: tuple = (128, 64)
    """GRU_MLP 躯干层宽度"""

    learning_rate: float = 0.0007221131969041414
    """学习率"""

    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_coef: float = 0.16557490394051463
    """PPO 裁剪系数"""
    ent_coef: float = 0.0018160124315864958
    """熵系数"""
    vf_coef: float = 0.5
    """价值函数损失系数"""
    max_grad_norm: float = 4.0
    """梯度裁剪阈值"""

    reward_norm: bool = True
    """是否启用奖励归一化"""
    reward_clip: float = 1.0
    """奖励归一化裁剪范围"""

@dataclass
class IppoArgs:
    """Multi-agent IPPO training config."""
    # Environment
    env_path: Optional[str] = "curriculum_envs/s4-enemy-only/build/game.exe"
    """游戏可执行文件路径"""
    config_path: str = "godot-game/configs/game_config.tres"
    """Godot 游戏配置文件路径（.tres）"""
    n_parallel: int = 3
    """并行 Godot 进程数"""
    seed: int = 1
    """随机种子"""
    show_window: bool = False
    """是否显示 Godot 窗口"""
    speedup: int = 10
    """游戏物理加速倍数"""

    # Training
    total_timesteps: int = 5_000_000
    """训练总时间步数"""
    count_steps_by: str = "env_steps"
    """步数统计维度：env_steps / agent_steps"""
    num_steps: int = 256
    """每个 rollout 的步数"""
    num_minibatches: int = 4
    """将 rollout 数据分成多少个小批量"""
    update_epochs: int = 8
    """每次 rollout 更新的 epoch 数"""
    recurrent_seq_len: int = 128
    """循环网络序列截断长度"""
    norm_adv: bool = True
    """是否标准化优势函数"""
    clip_vloss: bool = True
    """是否裁剪价值函数损失"""
    anneal_lr: bool = True
    """是否线性退火学习率"""
    target_kl: Optional[float] = None
    """KL 散度早停阈值（None=禁用）"""
    torch_deterministic: bool = True
    """是否启用确定性算法（确保可复现）"""
    cuda: bool = True
    """是否使用 CUDA"""

    #智能体配置
    agent_configs: list[AgentConfig] = field(default_factory=lambda: [
        AgentConfig(agent_id=0, train=True),
        AgentConfig(agent_id=1, train=False),
        AgentConfig(agent_id=2, train=False),
        AgentConfig(agent_id=3, train=False),
    ])

    #从ppo中训练好的模型
    # ppo_model_paths: list[Optional[str]] = field(default_factory=lambda: [
    # "saved-models/ppo_agent0.pt",
    # "saved-models/ppo_agent1.pt",
    # "saved-models/ppo_agent2.pt",
    # "saved-models/ppo_agent3.pt",])

    # Logging/checkpointing
    exp_name: str = "custom_ippo"
    """实验名称"""
    experiment_dir: str = "logs/cleanrl_ippo"
    """实验日志根目录"""
    save_model_path: Optional[str] = "saved-models/clean_rl_ippo"
    """最终模型保存路径（.pt）"""
    track: bool = False
    """是否记录到 WandB"""
    save_checkpoint: bool = True
    """是否在训练中周期性保存中断点"""
    resume_from: Optional[str] = None
    """从中断点恢复训练的路径（设置为 None 不恢复）"""
    load_model_path: Optional[str] = None
    """加载已有模型权重但不恢复优化器、计数器"""
    ppo_model_paths: list[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    """按 agent 下标加载 PPO 预训练模型权重，None 表示不加载"""
    save_every_n_episodes: int = 10
    """每 N 个 episode 保存一次中断点"""
    max_checkpoints: int = 3
    """最多保留多少个中断点文件，超出则删除最旧的"""
    wandb_project_name: str = "cleanRL"
    """WandB 项目名称"""
    wandb_entity: Optional[str] = "lunjijiang-rl"
    """WandB 用户 / 团队名"""

    # Opponent Pool (PFSP)
    use_opponent_pool: bool = True
    """是否启用 PFSP 对手池训练模式（每局只更新一个智能体，其余从池中采样）"""
    pool_max_size: int = 40
    """对手池最大容量（~10/agent）"""
    pool_slots_per_agent: int = 20
    """每个 agent slot 的队列容量，对应 opponent_pool[agent_id][slot_index]"""
    pool_initial_keep_per_agent: int = 10
    """从初始 IPPO 中断点目录中为每个 agent 载入最近多少个 checkpoint"""
    pool_phase_timesteps: int = 800_000
    """轮回对手池训练中，每个主 agent phase 的训练步数"""
    pool_rounds: int = 3
    """完整轮回次数；每轮按 pool_main_agent_order 依次训练"""
    pool_main_agent_order: tuple = (0, 1, 2, 3)
    """轮回训练顺序"""
    pool_final_agent_id: int = 0
    """轮回结束后额外长训练的主 agent"""
    pool_final_timesteps: int = 3_000_000
    """轮回结束后对 pool_final_agent_id 的额外训练步数"""
    pool_save_interval: int = 50_000
    """全局步数间隔：每隔多少步把当前四智能体快照加入对手池"""
    pool_checkpoint_dir: Optional[str] = None
    """对手池 checkpoint 文件目录。None 则使用 {experiment_dir}/pool_checkpoints"""
    pool_pfsp_temperature: float = 0.5
    """PFSP softmax 温度 (<1 聚焦有挑战的对手, >1 更均匀)"""
    pool_epsilon: float = 0.1
    """均匀采样概率（避免胜率估计过时）"""
    pool_win_rate_ema: float = 0.95
    """胜率 EMA 衰减系数"""
    pool_reward_ema: float = 0.9
    """按主 agent 回合平均奖励更新对手难度的 EMA 衰减系数"""
    pool_default_reward_score: float = 0.0
    """没有交手记录的对手组默认回合平均奖励；采样时奖励越低概率越高"""
    pool_elo_k_factor: float = 32.0
    """ELO 评分 K 因子"""
    pool_use_recency_bias: bool = True
    """是否对新近快照添加采样偏置"""
    pool_recency_scale: float = 1_000_000
    """新近偏置尺度因子"""
    pool_training_agent_selection: str = "round_robin"
    """训练 agent 选择策略: 'round_robin' / 'random'"""
    pool_initial_checkpoint_dir: Optional[str] ="saved_models/ippo_bootstrap"
    """ippo联合训练后的 checkpoint 目录，用于初始化对手池"""
    pool_log_every_n_updates: int = 10
    """每 N 个 update 打印一次对手池详细统计"""
    pool_delete_replaced_checkpoints: bool = False
    """队列挤出旧条目时是否删除对应 checkpoint 文件"""
    pool_bootstrap_checkpoint_timesteps: int = 1_000_000
    """IPPO 直接训练阶段：保存中断点的训练步数"""
    pool_bootstrap_extra_timesteps: int = 8_000_000
    """IPPO 直接训练阶段：不保存中断点的额外训练步数"""
    pool_final_save_agent_ids: tuple = (0,)
    """IPPO 直接训练或对手池最终保存时，需要保存的 agent id"""
    pool_eval_groups: int = 4
    """对比评价时随机抽取多少组对手 checkpoint"""
    pool_eval_episodes_per_group: int = 20
    """每组对手评价多少个 episode"""
    pool_eval_output_path: Optional[str] = "logs/ippo_pool_eval.csv"
    """对比评价 CSV 输出路径"""

    # Runtime-derived values
    num_agents: int = 0
    """智能体数量"""
    num_envs: int = 0
    """并行环境数量"""
    num_game_envs: int = 0
    """Godot 游戏进程数量"""
    batch_size: int = 0
    """每次更新的总样本数"""
    minibatch_size: int = 0
    """小批量大小"""

@dataclass
class RolloutData:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_obs: torch.Tensor
    next_done: torch.Tensor
    rnn_states: Optional[torch.Tensor] = None
    next_rnn_state: Optional[torch.Tensor] = None
    next_value: Optional[Union[torch.Tensor, float]] = None


@dataclass
class PoolEntry:
    """对手池中的一个策略快照条目。"""
    checkpoint_path: str
    """单 agent 权重 .pt 文件路径"""
    agent_id: int
    """归属哪个 agent slot（0~3）"""
    global_step: int
    """快照时的训练全局步数"""
    slot_index: int = 0
    """该条目在对应 agent 队列中的位置"""
    source: str = ""
    """条目来源，例如 bootstrap / pool_phase"""
    elo_rating: float = 1200.0
    """ELO 评分"""
    win_rate: float = 0.5
    """训练 agent 对该对手的 EMA 胜率 (0=全败, 1=全胜)"""
    main_reward_ema: float = 0.0
    """训练主 agent 面对该条目所在对手组时的回合平均奖励 EMA"""
    n_games: int = 0
    """该对手参与的游戏局数"""
    age: int = 0
    """自加入池以来经过的训练步数"""

@dataclass
class OpponentPool:
    """PFSP 对手池，管理所有 agent slot 的历史策略快照。"""
    entries: list[PoolEntry] = field(default_factory=list)
    """所有池条目"""
    entries_by_agent: dict[int, list[PoolEntry]] = field(default_factory=dict)
    """按 agent_id 分组后的队列，形如 opponent_pool[agent_id][slot_index]"""
    max_size: int = 40
    """池总容量（~10/agent）"""
    per_agent_max_size: int = 20
    """每个 agent slot 的队列容量"""
    save_interval: int = 50_000
    """快照保存间隔（全局步数）"""
    last_save_step: int = 0
    """上一次保存快照的全局步数"""
    elo_k_factor: float = 32.0
    """ELO K 因子"""
    win_rate_ema: float = 0.95
    """胜率 EMA 衰减系数"""
    epsilon: float = 0.1
    """均匀采样概率"""
    temperature: float = 0.5
    """PFSP softmax 温度"""
    use_recency_bias: bool = True
    """是否对新近快照添加采样偏置"""
    recency_scale: float = 1_000_000
    """新近偏置尺度因子"""
    training_agent_elo: float = 1200.0
    """当前训练 agent 的总 ELO（各 slot 共享一个追踪值）"""
