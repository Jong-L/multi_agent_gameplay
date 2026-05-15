"""
Independent PPO (IPPO),离散动作空间
每个智能体拥有独立的 PPO 网络、优化器和超参数配置，
在同一个 Godot 环境中同时训练多个智能体。
该脚本只支持一个godot进程，n_parallel为1
"""

import os
import pathlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    ObsSegmentDims,
    RewardNormalizer,
    init_training_setup,
    save_pt_model,
    layer_init,
)




class NetworkType(str, Enum):
    """Supported feature extractors for each IPPO policy."""

    SEGMENTED_MLP = "segmented_mlp"
    MLP = "mlp"
    GRU_MLP = "gru_mlp"

def as_hidden_tuple(value, default: tuple) -> tuple:
    """Normalize None/int/iterable hidden-size config into a tuple[int, ...]."""
    if value is None:
        value = default
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


def make_mlp(input_dim: int, hidden_sizes: tuple) -> tuple[nn.Module, int]:
    """Build Linear+ReLU layers and return both module and output size."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(layer_init(nn.Linear(in_dim, hidden_size)))
        layers.append(nn.ReLU())
        in_dim = hidden_size
    return (nn.Sequential(*layers) if layers else nn.Identity(), in_dim)


def init_gru_weights(gru: nn.GRU) -> nn.GRU:
    """Use stable GRU initialization matching rllib_custom_network.py."""
    for name, param in gru.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(param)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0.0)
    return gru


class SegmentedObsHelper:
    """Small helper for the shared observation layout."""

    def __init__(self, seg: ObsSegmentDims):
        self.seg = seg

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        i = 0
        s = obs[:, i: i + self.seg.self_dim];   i += self.seg.self_dim
        p = obs[:, i: i + self.seg.player_dim]; i += self.seg.player_dim
        b = obs[:, i: i + self.seg.ball_dim];   i += self.seg.ball_dim
        e = obs[:, i: i + self.seg.enemy_dim];  i += self.seg.enemy_dim
        m = obs[:, i: i + self.seg.map_dim]
        return s, p, b, e, m


class FlatMlpEncoder(nn.Module):
    def __init__(self, obs_dim: int, mlp_hiddens: tuple):
        super().__init__()
        self.trunk, self.output_dim = make_mlp(obs_dim, mlp_hiddens)
        self.recurrent_state_size = 0

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.trunk(obs), None


class SegmentedMlpEncoder(nn.Module):
    def __init__(
        self,
        obs_helper: SegmentedObsHelper,
        self_hiddens: tuple,
        player_hiddens: tuple,
        ball_hiddens: tuple,
        enemy_hiddens: tuple,
        map_hiddens: tuple,
        trunk_hiddens: tuple,
    ):
        super().__init__()
        seg = obs_helper.seg
        self.obs = obs_helper
        self.self_net, self_out = make_mlp(seg.self_dim, self_hiddens)
        self.player_net, player_out = make_mlp(seg.player_dim, player_hiddens)
        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        self.enemy_net, enemy_out = make_mlp(seg.enemy_dim, enemy_hiddens)
        self.map_net, map_out = make_mlp(seg.map_dim, map_hiddens)

        fused_dim = self_out + player_out + ball_out + enemy_out + map_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)
        self.recurrent_state_size = 0

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        s, p, b, e, m = self.obs.split(obs)
        fused = torch.cat([
            self.self_net(s),
            self.player_net(p),
            self.ball_net(b),
            self.enemy_net(e),
            self.map_net(m),
        ], dim=1)
        return self.trunk(fused), None


class GruMlpEncoder(nn.Module):
    """GRU-MLP encoder mirroring rllib_custom_network.py."""

    def __init__(
        self,
        obs_helper: SegmentedObsHelper,
        ball_hiddens: tuple,
        trunk_hiddens: tuple,
        gru_hidden: int,
        gru_num_layers: int,
        gru_input_layernorm: bool,
    ):
        super().__init__()
        seg = obs_helper.seg
        self.obs = obs_helper
        self.gru_hidden = int(gru_hidden)
        self.gru_num_layers = int(gru_num_layers)
        self.recurrent_state_size = self.gru_hidden * self.gru_num_layers

        gru_input_dim = seg.self_dim + seg.player_dim + seg.enemy_dim + seg.map_dim
        self.gru_ln = nn.LayerNorm(gru_input_dim) if gru_input_layernorm else nn.Identity()
        self.gru = init_gru_weights(
            nn.GRU(
                input_size=gru_input_dim,
                hidden_size=self.gru_hidden,
                num_layers=self.gru_num_layers,
                batch_first=True,
            )
        )

        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        fused_dim = self.gru_hidden + ball_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.recurrent_state_size, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        if rnn_state is None:
            rnn_state = self.initial_state(batch_size, obs.device)
        if rnn_state.dim() == 1:
            rnn_state = rnn_state.unsqueeze(0)

        s, p, b, e, m = self.obs.split(obs)
        gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(1)
        gru_input = self.gru_ln(gru_input)

        h0 = rnn_state.view(batch_size, self.gru_num_layers, self.gru_hidden)
        h0 = h0.transpose(0, 1).contiguous()
        gru_out, h_new = self.gru(gru_input, h0)

        ball_features = self.ball_net(b)
        fused = torch.cat([gru_out[:, -1, :], ball_features], dim=1)
        features = self.trunk(fused)
        h_new = h_new.transpose(0, 1).reshape(batch_size, self.recurrent_state_size)
        return features, h_new


@dataclass
class AgentConfig:
    agent_id: int
    train: bool = True

    network_type: NetworkType = NetworkType.MLP

    # segmented mlp
    self_hidden: int = 32
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hidden: tuple = (256, 128)
    self_hiddens: Optional[tuple] = None
    player_hiddens: Optional[tuple] = None
    ball_hiddens: Optional[tuple] = None
    enemy_hiddens: Optional[tuple] = None
    map_hiddens: Optional[tuple] = None
    trunk_hiddens: Optional[tuple] = None

    # mlp
    mlp_hiddens: tuple = (400, 150)

    # gru
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
    """全局训练配置。
    """

    #环境
    # env_path: Optional[str] = None
    env_path: Optional[str] = "godot-game/build/game.exe"
    config_path: str = "godot-game/configs/game_config.tres"
    n_parallel: int = 1 #只能为1
    seed: int = 1
    show_window: bool = True
    speedup: int = 10

    # 训练
    total_timesteps: int = 5_000_000
    count_steps_by: str = "agent_steps"
    """训练步数计数方式: "agent_steps" (智能体步数) 或 "env_steps" (环境步数)"""
    batch_size: int = 1024
    """每个智能体每个 rollout 的步数（也是每个 rollout 的环境步数）"""
    num_minibatches: int = 4
    update_epochs: int = 8
    recurrent_seq_len: int = 64
    """GRU_MLP 训练时的连续序列长度，用于 truncated BPTT。"""
    norm_adv: bool = True # 是否归一化优势函数
    clip_vloss: bool = True
    anneal_lr: bool = False
    target_kl: Optional[float] = None
    torch_deterministic: bool = True
    cuda: bool = True

    #智能体配置
    agent_configs: list[AgentConfig] = field(default_factory=lambda: [
        AgentConfig(agent_id=0, train=True),
        AgentConfig(agent_id=1,train=True),
        AgentConfig(agent_id=2,train=False),
        AgentConfig(agent_id=3,train=True),
    ])

    #日志 + 保存
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    experiment_dir: str = "logs/cleanrl_ippo"
    save_model_path: Optional[str] = "saved-models/clean_rl_ippo"
    # save_model_path: Optional[str] = None
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None

    #运行时衍生数据
    num_agents: int = 0
    """智能体数量"""
    num_envs: int = 0
    """环境数量"""
    minibatch_size: int = 0
    """每个智能体每个小批量的样本数量 (batch_size // num_minibatches)"""

class IPPOAgent(nn.Module):
    """Policy/value module with pluggable MLP, segmented MLP, or GRU-MLP encoder."""

    def __init__(
        self,
        n_actions: int,
        seg: ObsSegmentDims,
        cfg: AgentConfig,
    ):
        super().__init__()

        self.network_type = cfg.network_type
        self.seg = seg
        obs_helper = SegmentedObsHelper(seg)

        self_hiddens = as_hidden_tuple(cfg.self_hiddens, (cfg.self_hidden,))
        player_hiddens = as_hidden_tuple(cfg.player_hiddens, (cfg.player_hidden,))
        ball_hiddens = as_hidden_tuple(cfg.ball_hiddens, (cfg.ball_hidden,))
        enemy_hiddens = as_hidden_tuple(cfg.enemy_hiddens, (cfg.enemy_hidden,))
        map_hiddens = as_hidden_tuple(cfg.map_hiddens, (cfg.map_hidden,))
        trunk_hiddens = as_hidden_tuple(cfg.trunk_hiddens, cfg.trunk_hidden)
        mlp_hiddens = as_hidden_tuple(cfg.mlp_hiddens, ())

        if self.network_type == NetworkType.SEGMENTED_MLP:
            self.encoder = SegmentedMlpEncoder(
                obs_helper,
                self_hiddens,
                player_hiddens,
                ball_hiddens,
                enemy_hiddens,
                map_hiddens,
                trunk_hiddens,
            )
        elif self.network_type == NetworkType.MLP:
            self.encoder = FlatMlpEncoder(seg.total, mlp_hiddens)
        elif self.network_type == NetworkType.GRU_MLP:
            self.encoder = GruMlpEncoder(
                obs_helper,
                ball_hiddens=ball_hiddens,
                trunk_hiddens=trunk_hiddens,
                gru_hidden=cfg.gru_hidden,
                gru_num_layers=cfg.gru_num_layers,
                gru_input_layernorm=cfg.gru_input_layernorm,
            )
        else:
            raise ValueError(f"Unsupported network_type={self.network_type}")

        self.actor = layer_init(nn.Linear(self.encoder.output_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(self.encoder.output_dim, 1), std=1.0)

    @property
    def is_recurrent(self) -> bool:
        return self.encoder.recurrent_state_size > 0

    @property
    def recurrent_state_size(self) -> int:
        return self.encoder.recurrent_state_size

    def get_initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.is_recurrent:
            return None
        return self.encoder.initial_state(batch_size, device)

    def _forward_features(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.encoder(obs, rnn_state) if self.is_recurrent else self.encoder(obs)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_value(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features, _ = self._forward_features(obs, rnn_state)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        rnn_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        features, next_rnn_state = self._forward_features(obs, rnn_state)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        result = (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(features),
        )
        if return_state:
            return (*result, next_rnn_state)
        return result


@dataclass
class AgentRolloutData:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_obs: torch.Tensor
    next_done: torch.Tensor
    next_value: Optional[float] = None
    rnn_states: Optional[torch.Tensor] = None
    next_rnn_state: Optional[torch.Tensor] = None


#  GAE
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,#最后一步(idx:num_steps)状态价值
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (GAE)。"""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = (
            rewards[t]
            + gamma * nextvalues * nextnonterminal
            - values[t]
        )
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    return advantages, advantages + values


def compute_actor_loss(
    new_logprob: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    norm_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """裁剪 PPO actor loss。"""
    logratio = new_logprob - old_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return pg_loss, approx_kl, clipfrac


def compute_critic_loss(
    new_value: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    clip_coef: float,
    clip_vloss: bool = True,
) -> torch.Tensor:
    """裁剪 PPO critic loss。"""
    if clip_vloss:
        v_loss_unclipped = (new_value - returns) ** 2
        v_clipped = old_values + torch.clamp(
            new_value - old_values, -clip_coef, clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()
    return v_loss


def evaluate_recurrent_sequences(
    agent: IPPOAgent,
    rollout: AgentRolloutData,
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate contiguous rollout chunks while preserving GRU state flow."""
    if rollout.rnn_states is None:
        raise ValueError("Recurrent agent update requires rollout.rnn_states.")

    indices = []
    logprobs = []
    entropies = []
    values = []

    for start, end in zip(seq_starts, seq_ends):
        state = rollout.rnn_states[int(start)].unsqueeze(0).detach()
        for t in range(int(start), int(end)):
            _, logprob, entropy, value, state = agent.get_action_and_value(
                rollout.obs[t].unsqueeze(0),
                rollout.actions[t].unsqueeze(0),
                rnn_state=state,
                return_state=True,
            )
            indices.append(t)
            logprobs.append(logprob)
            entropies.append(entropy)
            values.append(value.view(-1))

            if t + 1 < int(end):
                state = state * (1.0 - rollout.dones[t + 1]).view(1, 1).to(device)

    return (
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.cat(logprobs, dim=0),
        torch.cat(entropies, dim=0),
        torch.cat(values, dim=0),
    )

#  多智能体 Rollout 收集
def collect_rollout_ippo(
    agents_cfg: list[AgentConfig],
    agents: list[IPPOAgent],
    envs: GodotDiscreteEnvWrapper,
    rollout_steps: int,
    device: torch.device,
    next_obs_all: torch.Tensor,
    next_done_all: torch.Tensor,
    global_step: int,
    episode_returns: list[deque],
    accum_rewards: list,
    reward_normalizers: list[Optional[RewardNormalizer]],
    rnn_states: list[Optional[torch.Tensor]],
) -> tuple[list[AgentRolloutData], int, list[Optional[torch.Tensor]]]:
    """实际包括rollout_steps+1步"""
    n_agents = len(agents_cfg)
    obs_dim = envs.single_observation_space.shape
    #预分配 per-agent 缓冲区
    buffers: list[dict] = []
    for _ in range(n_agents):
        buffers.append({
            "obs": torch.zeros((rollout_steps, obs_dim[0])).to(device),
            "actions": torch.zeros((rollout_steps,), dtype=torch.long).to(device),
            "logprobs": torch.zeros((rollout_steps,)).to(device),
            "rewards": torch.zeros((rollout_steps,)).to(device),
            "dones": torch.zeros((rollout_steps,)).to(device),
            "values": torch.zeros((rollout_steps,)).to(device),
            "rnn_states": None,
        })

    for i, agent in enumerate(agents):
        if agent.is_recurrent:
            buffers[i]["rnn_states"] = torch.zeros(
                (rollout_steps, agent.recurrent_state_size), device=device
            )
            if rnn_states[i] is None:
                rnn_states[i] = agent.get_initial_state(1, device)

    next_obs_all = next_obs_all.clone()#所有智能体的观测
    next_done_all = next_done_all.clone()

    for step in range(rollout_steps):
        global_step += 1

        # 每个智能体独立推理动作
        actions_list = [4] * n_agents  # 默认 IDLE 

        for i in range(n_agents):
            obs_i = next_obs_all[i].unsqueeze(0)  # shape(1, obs_dim)

            # 存入缓冲区
            buffers[i]["obs"][step] = obs_i.squeeze(0)#shape(obs_dim,)
            buffers[i]["dones"][step] = next_done_all[i]

            if agents_cfg[i].train:
                with torch.no_grad():
                    if agents[i].is_recurrent and next_done_all[i].item():
                        rnn_states[i] = agents[i].get_initial_state(1, device)
                    if agents[i].is_recurrent:
                        buffers[i]["rnn_states"][step] = rnn_states[i].squeeze(0)

                    action, logprob, _, value, next_rnn_state = agents[i].get_action_and_value(
                        obs_i,
                        rnn_state=rnn_states[i],
                        return_state=True,
                    )
                    buffers[i]["actions"][step] = action.item()
                    buffers[i]["logprobs"][step] = logprob.item()
                    buffers[i]["values"][step] = value.item()
                    if agents[i].is_recurrent:
                        rnn_states[i] = next_rnn_state.detach()
                
                actions_list[i] = int(action.item())
            else:
                # 非训练智能体: 随机动作, 无价值估计
                buffers[i]["actions"][step] = np.random.randint(0, envs.single_action_space.n)
                actions_list[i] = int(buffers[i]["actions"][step].item())
                buffers[i]["logprobs"][step] = 0.0
                buffers[i]["values"][step] = 0.0

        # 所有智能体一起执行动作
        next_obs_raw, rewards_raw, terminations, truncations, infos = envs.step(np.array(actions_list,dtype=np.int64))
        dones_raw = np.logical_or(terminations, truncations)#反应的是第i+1步状态是否终止

        #转换为张量
        next_obs_tensor = torch.tensor(np.array(next_obs_raw, dtype=np.float32)).to(device)
        next_done_tensor = torch.tensor(dones_raw, dtype=torch.float32).to(device)

        for i in range(n_agents):
            r = float(rewards_raw[i])

            # 奖励归一化
            if agents_cfg[i].train and reward_normalizers[i] is not None:
                r = reward_normalizers[i].normalize(r)  # 先用旧统计量归一化
                reward_normalizers[i].update(float(rewards_raw[i]))  # 再用原始值更新统计量

            buffers[i]["rewards"][step] = r

        # 回合奖励追踪 (使用原始奖励)
        for i in range(n_agents):
            if agents_cfg[i].train:
                accum_rewards[i] += float(rewards_raw[i])
                if dones_raw[i]:
                    episode_returns[i].append(accum_rewards[i])
                    accum_rewards[i] = 0.0

        next_obs_all = next_obs_tensor
        next_done_all = next_done_tensor 

    #包装为 AgentRolloutData
    rollouts = []
    for i in range(n_agents):
        next_val = None
        next_done_i = next_done_all[i]
        if agents_cfg[i].train:
            with torch.no_grad():
                next_val = agents[i].get_value(
                    next_obs_all[i].unsqueeze(0),
                    rnn_state=rnn_states[i],
                ).item()
        rollouts.append(AgentRolloutData(
            obs=buffers[i]["obs"],
            actions=buffers[i]["actions"],
            logprobs=buffers[i]["logprobs"],
            rewards=buffers[i]["rewards"],
            dones=buffers[i]["dones"],
            values=buffers[i]["values"],
            next_obs=next_obs_all[i],
            next_done=next_done_i,
            next_value=next_val,
            rnn_states=buffers[i]["rnn_states"],
            next_rnn_state=rnn_states[i],
        ))

    return rollouts, global_step, rnn_states


#  日志系统
def log_ippo(
    writer,
    global_step: int,
    agents_cfg: list[AgentConfig],
    optimizers: list[optim.Optimizer],
    losses: list[dict],         # [{pg_loss, v_loss, entropy, approx_kl, clipfrac}]
    explained_vars: list[float],
    episode_returns: list[deque],
    start_time: float,
    update: int = -1,
    num_updates: int = -1,
) -> None:
    """将 IPPO 训练指标写入 TensorBoard 并打印终端日志。"""
    # 全局指标
    sps = int(global_step / (time.time() - start_time)) if start_time > 0 else 0 #steps per second 
    writer.add_scalar("charts/SPS", sps, global_step)

    # per-agent 指标
    for i in range(len(agents_cfg)):
        tag = f"agent_{i}"

        # 学习率
        writer.add_scalar(
            f"{tag}/learning_rate", optimizers[i].param_groups[0]["lr"], global_step
        )

        # Loss 指标
        if losses[i] is not None:
            writer.add_scalar(f"{tag}/losses/policy_loss", losses[i]["pg_loss"], global_step)
            writer.add_scalar(f"{tag}/losses/value_loss", losses[i]["v_loss"], global_step)
            writer.add_scalar(f"{tag}/losses/entropy", losses[i]["entropy"], global_step)
            writer.add_scalar(f"{tag}/losses/approx_kl", losses[i]["approx_kl"], global_step)
            writer.add_scalar(f"{tag}/losses/clipfrac", losses[i]["clipfrac"], global_step)

        # Explained variance
        writer.add_scalar(f"{tag}/losses/explained_variance", explained_vars[i], global_step)

        # Episode return
        if len(episode_returns[i]) > 0:
            mean_ret = np.mean(np.array(episode_returns[i]))
            writer.add_scalar(f"{tag}/charts/episodic_return", mean_ret, global_step)

    # 终端日志
    if update > 0 and num_updates > 0:
        return_strs = []
        for i in range(len(agents_cfg)):
            if agents_cfg[i].train and len(episode_returns[i]) > 0:
                mean_ret = np.mean(np.array(episode_returns[i]))
                return_strs.append(f"p{i}:{mean_ret:.1f}")

        returns_summary = "  ".join(return_strs)

        kl_summary = "  ".join(
            f"p{i}:{losses[i]['approx_kl']:.3f}"
            for i in range(len(agents_cfg))
            if agents_cfg[i].train and losses[i] is not None
        )

        ev_summary = "  ".join(
            f"p{i}:{explained_vars[i]:.3f}"
            for i in range(len(agents_cfg))
            if agents_cfg[i].train and losses[i] is not None
        )

        print(
            f"[Update {update:4d}/{num_updates}] "
            # f"SPS: {sps:5d}  "
            f"returns [{returns_summary}]  "
            f"kl [{kl_summary}]  "
            f"ev [{ev_summary}]"
        )


#  模型保存
def save_ippo_model(
    save_path: str,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    reward_normalizers: list[Optional[RewardNormalizer]],
    args: IppoArgs,
) -> None:
    """保存 IPPO 所有智能体的模型检查点。
    """
    save_path = pathlib.Path(save_path).with_suffix(".pt")#自动追加 .pt 后缀
    agent_states = {}
    for i, agent in enumerate(agents):
        agent_states[f"agent_{i}_state_dict"] = agent.state_dict() #网络参数
        agent_states[f"agent_{i}_optimizer"] = optimizers[i].state_dict() #优化器参数
        if reward_normalizers[i] is not None:
            agent_states[f"agent_{i}_reward_norm"] = reward_normalizers[i].state_dict() #奖励归一化器参数

    save_path.parent.mkdir(parents=True, exist_ok=True) # 创建保存目录
    torch.save({"args": vars(args), **agent_states},str(save_path))
    print(f"[Save] IPPO model saved to {save_path}")


#  Per-Agent 训练更新
def train_agent_update(
    agent: IPPOAgent,
    optimizer: optim.Optimizer,
    rollout: AgentRolloutData,
    cfg: AgentConfig,
    args: IppoArgs,
    device: torch.device,
) -> dict:
    """对单个智能体执行一次 PPO 更新 (多个 epoch + minibatch)
    """
    batch_size = rollout.obs.shape[0]  # per-agent (=args.batch_size)

    #GAE
    with torch.no_grad():
        next_value_t = torch.tensor([rollout.next_value], device=device)         # (1,)
        next_done_t = rollout.next_done.unsqueeze(0).to(device)                  # (1,)

        advantages, target_values = compute_gae(
            rollout.rewards.unsqueeze(1),     # (num_steps, 1)
            rollout.values.unsqueeze(1),       # (num_steps, 1)
            rollout.dones.unsqueeze(1),        # (num_steps, 1)
            next_value_t,                      # (1,)
            next_done_t,                       # (1,)
            cfg.gamma,
            cfg.gae_lambda,
        )
    # Flatten
    b_obs = rollout.obs              # (num_steps, obs_dim)
    b_actions = rollout.actions      # (num_steps,)
    b_logprobs = rollout.logprobs    # (num_steps,)
    b_advantages = advantages.squeeze(1)  # (num_steps,)
    b_target_values = target_values.squeeze(1)  # (num_steps,)
    b_values = rollout.values        # (num_steps,)

    clipfracs = []
    pg_losses = []
    v_losses = []
    entropies = []
    approx_kls = []

    if agent.is_recurrent:
        seq_len = max(1, min(int(args.recurrent_seq_len), batch_size))
        all_seq_starts = np.arange(0, batch_size, seq_len)
        all_seq_ends = np.minimum(all_seq_starts + seq_len, batch_size)
        seq_inds = np.arange(len(all_seq_starts))
        seqs_per_minibatch = max(
            1, (len(seq_inds) + args.num_minibatches - 1) // args.num_minibatches
        )

        for epoch in range(args.update_epochs):
            np.random.shuffle(seq_inds)
            kl_start = len(approx_kls)

            for start in range(0, len(seq_inds), seqs_per_minibatch):
                mb_seq_inds = seq_inds[start:start + seqs_per_minibatch]
                mb_inds, new_logprob, entropy, new_value = evaluate_recurrent_sequences(
                    agent,
                    rollout,
                    all_seq_starts[mb_seq_inds],
                    all_seq_ends[mb_seq_inds],
                    device,
                )

                pg_loss, approx_kl, clipfrac = compute_actor_loss(
                    new_logprob,
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    cfg.clip_coef,
                    args.norm_adv,
                )
                clipfracs.append(clipfrac)

                v_loss = compute_critic_loss(
                    new_value,
                    b_target_values[mb_inds],
                    b_values[mb_inds],
                    cfg.clip_coef,
                    args.clip_vloss,
                )

                loss = pg_loss - cfg.ent_coef * entropy.mean() + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl.item())

            if args.target_kl is not None:
                epoch_approx_kl = float(np.mean(approx_kls[kl_start:]))
                if epoch_approx_kl > args.target_kl:
                    break
    else:
        b_inds = np.arange(batch_size)#batch indices
        minibatch_size = max(1, batch_size // args.num_minibatches)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            kl_start = len(approx_kls)  # 记录本 epoch 起始位置

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                new_value = new_value.view(-1)  # 转换为一维向量(minibatch_size,)

                # Actor loss
                pg_loss, approx_kl, clipfrac = compute_actor_loss(
                    new_logprob,
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    cfg.clip_coef,
                    args.norm_adv,
                )
                clipfracs.append(clipfrac)

                # Critic loss
                v_loss = compute_critic_loss(
                    new_value,
                    b_target_values[mb_inds],
                    b_values[mb_inds],
                    cfg.clip_coef,
                    args.clip_vloss,
                )

                # Total loss
                loss = pg_loss - cfg.ent_coef * entropy.mean() + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl.item())

            # KL 早停（使用本 epoch 所有 minibatch 的平均 KL）
            if args.target_kl is not None:
                epoch_approx_kl = float(np.mean(approx_kls[kl_start:]))
                if epoch_approx_kl > args.target_kl:
                    break

    final_metrics = {
        "pg_loss": np.mean(pg_losses),
        "v_loss": np.mean(v_losses),
        "entropy": np.mean(entropies),
        "approx_kl": np.mean(approx_kls),
        "clipfrac": np.mean(clipfracs),
    }

    #Explained variance
    y_pred = b_values.cpu().numpy()
    y_true = b_target_values.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    final_metrics["explained_var"] = explained_var

    return final_metrics


#  主训练循环

def train(
    args: IppoArgs,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    envs: GodotDiscreteEnvWrapper,
    device: torch.device,
    writer,
    reward_normalizers: list[Optional[RewardNormalizer]],
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
) -> None:
    """IPPO 主训练循环"""
    n_agents = len(args.agent_configs)
    global_step = 0
    start_time = time.time()

    # 按 count_steps_by 计算总更新次数
    if args.count_steps_by == "env_steps":
        num_updates = args.total_timesteps // args.batch_size
    elif args.count_steps_by == "agent_steps":
        num_updates = args.total_timesteps // (n_agents * args.batch_size)
    else:
        raise ValueError(
            f"Unknown count_steps_by='{args.count_steps_by}'. "
            "Expected 'env_steps' or 'agent_steps'."
        )

    episode_returns = [deque(maxlen=100) for _ in range(n_agents)] #每个智能体最近100个回合的奖励
    accum_rewards = np.zeros(n_agents, dtype=np.float64)#每个智能体每回合累计奖励
    rnn_states = [agent.get_initial_state(1, device) for agent in agents] #每个智能体的初始 RNN 状态

    for update in range(1, num_updates + 1):
        # 学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            for i in range(n_agents):
                if args.agent_configs[i].train:
                    optimizers[i].param_groups[0]["lr"] = progress * args.agent_configs[i].learning_rate

        # 经验采集
        rollouts, global_step, rnn_states = collect_rollout_ippo(
            args.agent_configs, agents, envs, args.batch_size, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards, reward_normalizers,
            rnn_states,
        )

        next_obs = torch.stack([r.next_obs for r in rollouts])
        next_done = torch.stack([r.next_done for r in rollouts])

        # 独立更新
        losses = []
        explained_vars = []

        for i in range(n_agents):
            if args.agent_configs[i].train:
                metrics = train_agent_update(agents[i], optimizers[i], rollouts[i], args.agent_configs[i], args, device)
                losses.append(metrics)
                explained_vars.append(metrics.get("explained_var", 0.0))
            else:
                losses.append(None)
                explained_vars.append(0.0)

        # 日志
        log_ippo(
            writer, global_step, args.agent_configs, optimizers,
            losses, explained_vars,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
        )


#  主训练入口
def main():
    # 初始化
    args = IppoArgs()
    writer, device, envs, seg, run_name = init_training_setup(args)

    if args.n_parallel != 1:
        raise ValueError(f"this script only supports n_parallel=1, but got {args.n_parallel}.")

    n_agents = len(args.agent_configs)
    args.num_agents = n_agents
    args.num_envs = envs.num_envs
    if args.num_envs != n_agents:
        raise ValueError(
            "IPPO expects one Godot training slot per configured agent: "
            f"envs.num_envs={args.num_envs}, len(args.agent_configs)={n_agents}."
        )

    obs_shape = envs.single_observation_space.shape
    if len(obs_shape) != 1 or obs_shape[0] != seg.total:
        raise ValueError(f"Observation dimension mismatch: env observation shape={obs_shape}, configured segment total={seg.total}.")

    args.minibatch_size = max(1, args.batch_size // args.num_minibatches)

    n_actions = int(envs.single_action_space.n)

    # 创建 per-agent 网络、优化器、奖励归一化器
    agents: list[IPPOAgent] = []
    optimizers: list[optim.Optimizer] = []
    reward_normalizers: list[Optional[RewardNormalizer]] = []

    for cfg in args.agent_configs:
        agent = IPPOAgent(n_actions, seg, cfg).to(device)

        # 打印每个智能体的网络类型和参数量
        tag = f"[Agent {cfg.agent_id}]"
        print(f"{tag} network_type={cfg.network_type}, params={agent.num_params():,}")

        agents.append(agent)
        optimizers.append(optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5))

        if cfg.train and cfg.reward_norm:
            reward_normalizers.append(RewardNormalizer(clip=cfg.reward_clip))
        else:
            reward_normalizers.append(None)

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed) #shape (n_agents, obs_dim)
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device)
    if next_obs.shape[0] != n_agents:
        raise ValueError(
            "Reset observation count does not match configured agents: "
            f"next_obs.shape[0]={next_obs.shape[0]}, len(args.agent_configs)={n_agents}."
        )
    next_done = torch.zeros(n_agents).to(device)

    try:
        train(
            args, agents, optimizers, envs, device, writer,
            reward_normalizers, next_obs, next_done,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] 训练被手动中断")
        if args.save_model_path is not None:
            print(f"[Interrupt] 保存检查点到 {args.save_model_path} ...")
            save_ippo_model(args.save_model_path, agents, optimizers, reward_normalizers, args)
        return
    finally:
        envs.close()
        writer.close()

    # 正常训练结束后的保存
    if args.save_model_path is not None:
        save_ippo_model(args.save_model_path, agents, optimizers, reward_normalizers, args)


if __name__ == "__main__":
    main()
