from dataclasses import dataclass, field
from enum import Enum
from math import fabs
from typing import Optional, Union

import torch


class NetworkType(str, Enum):
    """Supported feature extractors for PPO/IPPO policies."""

    SEGMENTED_MLP = "segmented_mlp"
    MLP = "mlp"
    GRU_MLP = "gru_mlp"


@dataclass
class Args:
    """Single-agent PPO training config."""
    # Environment
    env_path: Optional[str] = "curriculum_envs\\s4-enemy-only\\build\\game.exe"
    config_path: str = "curriculum_envs\\s4-enemy-only\\configs\\game_config.tres"
    n_parallel: int = 4
    seed: int = 1
    show_window: bool = False
    speedup: int = 16

    # Logging/checkpointing
    exp_name: str = "custom_ppo"
    experiment_dir: str = "logs/cleanrl_ppo"
    save_model_path: Optional[str] = None
    resume_from: Optional[str] = None
    load_model_path: Optional[str] = None
    save_every_n_episodes: int = 10
    max_checkpoints: int = 3
    track: bool = False
    wandb_project_name: str = "cleanRL_ppo"
    wandb_entity: Optional[str] = "lunjijiang-rl"

    # PPO hyperparameters
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 8
    recurrent_seq_len: int = 128
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 4.0
    norm_adv: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = False
    target_kl: Optional[float] = None
    torch_deterministic: bool = True
    cuda: bool = True
    reward_norm: bool = True
    reward_clip: float = 5.0

    # Network
    network_type: NetworkType = NetworkType.MLP
    self_hidden: int = 32
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hiddens: tuple = (128, 64)
    mlp_hiddens: tuple = (256, 128, 64)
    gru_hidden: int = 128
    gru_num_layers: int = 1
    gru_input_layernorm: bool = True

    # Runtime-derived values
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0

@dataclass
class AgentConfig:
    agent_id: int
    train: bool = True

    network_type: NetworkType = NetworkType.MLP

    # Segmented MLP
    self_hidden: int = 32
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hiddens: tuple = (256, 128)

    # MLP
    mlp_hiddens: tuple = (400, 150)

    # GRU
    gru_hidden: int = 128
    gru_num_layers: int = 1
    gru_input_layernorm: bool = True

    learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 4.0

    reward_norm: bool = True
    reward_clip: float = 5.0


@dataclass
class IppoArgs:
    """Multi-agent IPPO training config."""
    # Environment
    env_path: Optional[str] = "godot-game/build/game.exe"
    config_path: str = "godot-game/configs/game_config.tres"
    n_parallel: int = 1
    seed: int = 1
    show_window: bool = False
    speedup: int = 10

    # Training
    total_timesteps: int = 5_000_000
    count_steps_by: str = "env_steps"
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 8
    recurrent_seq_len: int = 128
    norm_adv: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = False
    target_kl: Optional[float] = None
    torch_deterministic: bool = True
    cuda: bool = True

    # Multi-agent config. These are ignored when is_multi_agent=False.
    agent_configs: list[AgentConfig] = field(default_factory=lambda: [
        AgentConfig(agent_id=0, train=True),
        AgentConfig(agent_id=1, train=True),
        AgentConfig(agent_id=2, train=False),
        AgentConfig(agent_id=3, train=True),
    ])

    # Logging/checkpointing
    exp_name: str = "custom_ippo"
    experiment_dir: str = "logs/cleanrl_ippo"
    save_model_path: Optional[str] = "saved-models/clean_rl_ippo"
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = "lunjijiang-rl"

    # Runtime-derived values
    num_agents: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0

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
