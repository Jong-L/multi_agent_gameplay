"""
离散动作 PPO (Proximal Policy Optimization)
"""
import os
import time
from collections import deque
from typing import Optional

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
    layer_init,
    load_full_checkpoint,
    save_pt_model,
)

from custom_ppo_dataclass import Args, NetworkType, RolloutData

def as_hidden_tuple(value, default: tuple) -> tuple:
    """
    将输入的值(整数，none，可迭代对象)转换为元组
    """
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

    def forward(self,obs: torch.Tensor,) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        #分段mlp子网络
        self.self_net, self_out = make_mlp(seg.self_dim, self_hiddens)
        self.player_net, player_out = make_mlp(seg.player_dim, player_hiddens)
        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        self.enemy_net, enemy_out = make_mlp(seg.enemy_dim, enemy_hiddens)
        self.map_net, map_out = make_mlp(seg.map_dim, map_hiddens)

        fused_dim = self_out + player_out + ball_out + enemy_out + map_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)
        self.recurrent_state_size = 0

    def forward(self,obs: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        self.recurrent_state_size = self.gru_hidden * self.gru_num_layers #状态维度：(B,L*H)

        gru_input_dim = seg.self_dim + seg.player_dim + seg.enemy_dim + seg.map_dim #gru输入维度
        self.gru_ln = nn.LayerNorm(gru_input_dim) if gru_input_layernorm else nn.Identity()
        #初始化gru权重
        self.gru = init_gru_weights(
            nn.GRU(
                input_size=gru_input_dim,
                hidden_size=self.gru_hidden,
                num_layers=self.gru_num_layers,
                batch_first=True,
            )
        )
        #奖励球的mlp子网络
        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        #融合后的mlp网络
        fused_dim = self.gru_hidden + ball_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化gru隐藏状态(B, L*H)
        """
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
            rnn_state = rnn_state.unsqueeze(0)#(1, B, L*H)

        s, p, b, e, m = self.obs.split(obs)
        gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(1)#(B, 1, H)
        gru_input = self.gru_ln(gru_input)#归一化

        h0 = rnn_state.view(batch_size, self.gru_num_layers, self.gru_hidden)#(B, L, H)
        h0 = h0.transpose(0, 1).contiguous()#(L, B, H)
        #gru输出
        gru_out, h_new = self.gru(gru_input, h0)#out:(B,seq_len, H), h_new:(L, B, H)

        #奖励球输出
        ball_features = self.ball_net(b)
        #取最后一个时间步的输出进行融合特征
        fused = torch.cat([gru_out[:, -1, :], ball_features], dim=1)
        features = self.trunk(fused)
        #更新隐藏状态
        h_new = h_new.transpose(0, 1).reshape(batch_size, self.recurrent_state_size)#(B, L*H)
        return features, h_new

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,       # (seq_len, obs_dim)
        rnn_state: torch.Tensor,     # (1, L*H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """整序列前向传播：一次 GRU 调用处理全部时间步。"""
        s, p, b, e, m = self.obs.split(obs_seq)

        gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(0)#(1, seq_len, H)
        gru_input = self.gru_ln(gru_input)

        h0 = rnn_state.view(1, self.gru_num_layers, self.gru_hidden)#(1, L, H)
        h0 = h0.transpose(0, 1).contiguous()#(L, 1, H)

        gru_out, h_new = self.gru(gru_input, h0)#out:(1, seq_len, H), h_new:(L, 1, H)

        ball_features = self.ball_net(b)
        fused = torch.cat([gru_out.squeeze(0), ball_features], dim=1)
        features = self.trunk(fused)

        h_new = h_new.transpose(0, 1).reshape(1, self.recurrent_state_size)#(1, L*H)
        return features, h_new

#  训练配置
class PPOAgent(nn.Module):
    """离散 PPO 智能体
    输入: 观测向量 (obs_dim,):
          [self_state | nearby_players | nearby_balls | nearby_enemies | map_state]
    输出: 离散动作 (0~n_actions-1) + 对数概率 + 熵 + 状态价值
    """
    def __init__(
        self,
        n_actions: int,
        seg: ObsSegmentDims,
        args: Args,
    ):
        super().__init__()

        self.network_type = args.network_type
        self.seg = seg
        obs_helper = SegmentedObsHelper(seg)

        # 将隐藏层参数都用元组表示
        self_hiddens = as_hidden_tuple(args.self_hidden, (args.self_hidden,))
        player_hiddens = as_hidden_tuple(args.player_hidden, (args.player_hidden,))
        ball_hiddens = as_hidden_tuple(args.ball_hidden, (args.ball_hidden,))
        enemy_hiddens = as_hidden_tuple(args.enemy_hidden, (args.enemy_hidden,))
        map_hiddens = as_hidden_tuple(args.map_hidden, (args.map_hidden,))
        trunk_hiddens = as_hidden_tuple(args.trunk_hiddens, args.trunk_hiddens)
        mlp_hiddens = as_hidden_tuple(args.mlp_hiddens, args.mlp_hiddens)

        # 创建不同网络结构
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
                gru_hidden=args.gru_hidden,
                gru_num_layers=args.gru_num_layers,
                gru_input_layernorm=args.gru_input_layernorm,
            )
        else:
            raise ValueError(f"Unsupported network_type={self.network_type}")

        #  Actor 头
        self.actor = layer_init(nn.Linear(self.encoder.output_dim, n_actions), std=0.01)

        # Critic 头
        self.critic = layer_init(nn.Linear(self.encoder.output_dim, 1), std=1.0)

    @property
    def is_recurrent(self) -> bool:
        return self.encoder.recurrent_state_size > 0

    @property
    def recurrent_state_size(self) -> int:
        return self.encoder.recurrent_state_size

    def get_initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        '''gru隐藏层初始状态'''
        if not self.is_recurrent:
            return None
        return self.encoder.initial_state(batch_size, device)

    # 前向传播
    def _forward_features(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """将观测向量按段切片, 送入各子网络, 融合后经过共享躯干。"""
        return self.encoder(obs, rnn_state) if self.is_recurrent else self.encoder(obs)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_value(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取状态价值 V(s)。"""
        features, _ = self._forward_features(obs, rnn_state)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        rnn_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """根据观测计算当前价值、采样动作并计算相关统计量
        """
        features, next_rnn_state = self._forward_features(obs, rnn_state)
        logits = self.actor(features)
        probs = Categorical(logits=logits) #转换为动作概率分布

        if action is None:#如果未提供动作
            action = probs.sample()

        result = (
            action,
            probs.log_prob(action),# 计算动作对数概率
            probs.entropy(),# 计算策略熵
            self.critic(features), 
        )
        if return_state:
            return (*result, next_rnn_state)
        return result

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,       # (seq_len, obs_dim)
        action_seq: torch.Tensor,    # (seq_len,)
        rnn_state: torch.Tensor,     # (1, L*H)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """整段序列评估：计算每个时间步的 logprob / 熵 / 价值。"""
        features_seq, next_state = self.encoder.forward_sequence(obs_seq, rnn_state)
        logits_seq = self.actor(features_seq)
        probs = Categorical(logits=logits_seq)
        logprobs = probs.log_prob(action_seq)
        entropies = probs.entropy()
        values = self.critic(features_seq).squeeze(-1)

        return logprobs, entropies, values, next_state

#  经验收集 (rollout)
def collect_rollout(
    agent: PPOAgent,
    envs: GodotDiscreteEnvWrapper,
    num_steps: int,
    device: torch.device,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    global_step: int,
    episode_returns: deque,
    accum_rewards: np.ndarray,
    reward_normalizer: Optional[RewardNormalizer] = None,
    rnn_state: Optional[torch.Tensor] = None,
) -> tuple[RolloutData, int, list[float]]:
    """使用当前策略收集 num_steps 步经验。
    加上最后一步，实际包括num_steps+1步
    """
    num_envs = envs.num_envs
    obs_dim = envs.single_observation_space.shape

    # 预分配缓冲区
    obs = torch.zeros((num_steps, num_envs) + obs_dim).to(device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    rnn_states = None
    if agent.is_recurrent:
        rnn_states = torch.zeros((num_steps, num_envs, agent.recurrent_state_size)).to(device)
        if rnn_state is None:#初始状态
            rnn_state = agent.get_initial_state(num_envs, device)#shape(num_envs,L*H)

    for step in range(num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done #dones[t]表示s_t是否终止，dones[0]为初始值0

        # 用当前策略采样动作并用旧网络计算状态值
        with torch.no_grad():
            if agent.is_recurrent:
                rnn_state = rnn_state * (1.0 - next_done).view(-1, 1)#回合结束时重置状态
                rnn_states[step] = rnn_state#每步状态
            action, logprob, _, value, next_rnn_state = agent.get_action_and_value(
                next_obs,
                rnn_state=rnn_state,
                return_state=True,
            )
            values[step] = value.flatten()#将(1, num_envs)转换为(num_envs,)
            if agent.is_recurrent:
                rnn_state = next_rnn_state.detach()

        actions[step] = action
        logprobs[step] = logprob

        # 执行动作   next_obs_raw形状为(num_envs, obs_dim)
        next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)

        # 奖励归一化
        reward_arr = np.asarray(reward, dtype=np.float32)
        if reward_normalizer is not None:
            reward_arr = reward_normalizer.normalize_array(reward_arr)#先用旧统计量归一化
            reward_normalizer.update_array(np.asarray(reward, dtype=np.float32))

        rewards[step] = torch.tensor(reward_arr, dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs_raw, dtype=np.float32)).to(device)
        next_done = torch.tensor(done, dtype=torch.float32).to(device)

        # 追踪平均回合奖励 (使用原始奖励)
        accum_rewards += np.asarray(reward, dtype=np.float64)
        new_episode_returns: list[float] = []
        for i, d in enumerate(np.asarray(done)):
            if d:
                ep_ret = float(accum_rewards[i])
                episode_returns.append(ep_ret)
                new_episode_returns.append(ep_ret)
                accum_rewards[i] = 0.0

    with torch.no_grad():
        next_value = agent.get_value(next_obs, rnn_state).reshape(1, -1)

    return (
        RolloutData(obs, actions, logprobs, rewards, dones, values,next_obs, next_done, rnn_states, rnn_state, next_value),
        global_step,
        new_episode_returns,
    )

#  GAE 优势估计
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Generalized Advantage Estimation (GAE)。
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

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
            delta
            + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    target_values = advantages + values
    return advantages, target_values # (num_steps, num_envs)

#  PPO 损失函数
def compute_actor_loss(
    new_logprob: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    norm_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """计算裁剪actor loss。
    """
    logratio = new_logprob - old_logprobs
    ratio = logratio.exp()

    # KL 散度近似
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

    # 优势标准化
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 裁剪策略损失
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
    """计算critic loss。
    """
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
    agent: PPOAgent,
    rollout: RolloutData,
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
    seq_envs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 done 边界切分子序列，整段批量前向传播。"""
    if rollout.rnn_states is None:
        raise ValueError("Recurrent PPO update requires rollout.rnn_states.")

    num_envs = rollout.obs.shape[1]
    all_indices = []
    all_logprobs = []
    all_entropies = []
    all_values = []

    for start_t, end_t, env_i in zip(seq_starts, seq_ends, seq_envs):
        start_t = int(start_t)
        end_t = int(end_t)
        env_i = int(env_i)

        seq_len = end_t - start_t
        done_seq = rollout.dones[start_t:end_t, env_i]
        done_positions = torch.where(done_seq > 0.5)[0].cpu().tolist()
        split_points = [0] + [int(p) for p in done_positions] + [seq_len]

        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end = split_points[i + 1]
            if sub_start >= sub_end:
                continue

            abs_start = start_t + sub_start
            abs_end = start_t + sub_end

            state = rollout.rnn_states[abs_start, env_i].unsqueeze(0).detach()
            sub_obs = rollout.obs[abs_start:abs_end, env_i]
            sub_actions = rollout.actions[abs_start:abs_end, env_i]

            logprobs, entropies, values, state = agent.evaluate_sequence(
                sub_obs, sub_actions, state,
            )

            sub_indices = torch.arange(abs_start, abs_end, device=device) * num_envs + env_i
            all_indices.append(sub_indices)
            all_logprobs.append(logprobs)
            all_entropies.append(entropies)
            all_values.append(values)

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_logprobs, dim=0),
        torch.cat(all_entropies, dim=0),
        torch.cat(all_values, dim=0),
    )

#  日志 + 模型导出
def log_ppo(
    writer,
    global_step: int,
    optimizer: optim.Optimizer,
    v_loss: torch.Tensor,
    pg_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    approx_kl: torch.Tensor,
    clipfracs: list,
    explained_var: float,
    episode_returns: deque,
    start_time: float,
    update: int = -1,
    num_updates: int = -1,
    new_episode_returns: Optional[list[float]] = None,
) -> None:
    """将 PPO 训练指标写入 TensorBoard 并打印终端日志。
    """
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

    #时间统计
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    if len(episode_returns) > 0:
        sps = int(global_step / (time.time() - start_time))
        mean_return = np.mean(np.array(episode_returns))
        if update > 0 and num_updates > 0:
            print(
                f"[Update {update:4d}/{num_updates}] "
                # f"SPS: {sps:5d}  "
                f"return: {mean_return:8.2f}  "
                f"kl: {approx_kl.item():.4f}"
                f"   training time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
        writer.add_scalar("charts/SPS", sps, global_step)
        # 有新 episode 完成时用最新单个回合奖励；否则保持上一次记录值
        if new_episode_returns is not None and len(new_episode_returns) > 0:
            writer.add_scalar("charts/episodic_return", new_episode_returns[-1], global_step)
    else:
        if update > 0 and num_updates > 0:
            print(
                f"[Update {update:4d}/{num_updates}]"
                f"kl: {approx_kl.item():.4f}"
                f"   training time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                )

def _count_completed_episodes(rollout: RolloutData) -> int:
    """统计本次 rollout 中完成的 episode 数量。"""
    done_rows = [rollout.next_done.unsqueeze(0)]
    if rollout.dones.shape[0] > 1:
        done_rows.insert(0, rollout.dones[1:])
    dones = torch.cat(done_rows, dim=0)
    return int(torch.any(dones > 0.5, dim=1).sum().item())

def _build_train_state(
    global_step: int,
    update: int,
    episode_count: int,
    optimizer: optim.Optimizer,
    episode_returns: deque,
) -> dict:
    """打包当前训练状态，用于 checkpoint 保存。"""
    state: dict = dict(
        global_step=int(global_step),
        update=int(update),
        episode_count=int(episode_count),
        lr=float(optimizer.param_groups[0]["lr"]),
    )
    state["episode_returns"] = list(episode_returns)
    return state

def _load_train_state(ckpt: dict) -> tuple[int, int, int, deque]:
    """从 checkpoint dict 恢复训练运行时变量。"""
    global_step = int(ckpt.get("global_step", 0))
    update = int(ckpt.get("update", 0))
    episode_count = int(ckpt.get("episode_count", 0))
    episode_returns = deque(ckpt.get("episode_returns", []), maxlen=20)
    return global_step, update, episode_count, episode_returns

def _restore_reward_normalizer(
    ckpt: dict,
    reward_normalizer: Optional[RewardNormalizer],
) -> None:
    """从 checkpoint 恢复奖励归一化器状态。"""
    if reward_normalizer is not None and "reward_normalizer" in ckpt:
        reward_normalizer.load_state_dict(ckpt["reward_normalizer"])

def _make_checkpoint_path(base_path: Optional[str], episode_count: int) -> Optional[str]:
    """生成带 episode 编号的 checkpoint 路径。"""
    if base_path is None:
        return None
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{base}_episode{episode_count}{ext}"

def _cleanup_old_checkpoints(base_path: Optional[str], max_keep: int) -> None:
    """删除超出保留数量的旧 checkpoint。"""
    if base_path is None or max_keep <= 0:
        return
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    prefix = os.path.basename(base) + "_episode"
    dir_name = os.path.dirname(base_path) or "."

    if not os.path.isdir(dir_name):
        return

    checkpoints: list[tuple[int, str]] = []
    for f in os.listdir(dir_name):
        if f.startswith(prefix) and f.endswith(ext):
            try:
                num_str = f[len(prefix):-len(ext)]
                episode_num = int(num_str)
                checkpoints.append((episode_num, os.path.join(dir_name, f)))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, old_path = checkpoints.pop(0)
        try:
            os.remove(old_path)
        except OSError:
            pass

def _save_checkpoint(
    ckpt_path: str,
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
    reward_normalizer: Optional[RewardNormalizer],
    extra: dict,
) -> None:
    """保存模型 + optimizer + 训练状态到 checkpoint 文件。"""
    save_pt_model(
        ckpt_path,
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args,
        reward_normalizer=reward_normalizer,
        extra=extra,
    )

def load_checkpoint_if_requested(
    resume_path: Optional[str],
    is_resume: bool,
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    reward_normalizer: Optional[RewardNormalizer],
    device: torch.device,
) -> tuple[int, int, int, deque]:
    """按需恢复模型 / optimizer / 归一化器 / 训练计数。"""
    if not resume_path:
        return 0, 1, 0, deque(maxlen=20)

    print(f"[Resume] 加载 checkpoint: {resume_path}")
    ckpt = load_full_checkpoint(resume_path, device)
    agent.load_state_dict(ckpt["agent_state_dict"])

    if is_resume:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Resume] optimizer state restored")
        _restore_reward_normalizer(ckpt, reward_normalizer)
        start_global_step, start_update, start_episode_count, episode_returns = _load_train_state(ckpt)
        start_update += 1
        print(
            f"[Resume] 从 update {start_update} / step {start_global_step} "
            f"/ episode {start_episode_count} 继续"
        )
    else:
        print("[Load] 仅加载模型参数，其余从头初始化")
        return 0, 1, 0, deque(maxlen=20)

    return start_global_step, start_update, start_episode_count, episode_returns

#  主训练循环
def train(
    args: Args,
    agent: PPOAgent,
    envs: GodotDiscreteEnvWrapper,
    optimizer: optim.Optimizer,
    device: torch.device,
    writer,
    reward_normalizer: Optional[RewardNormalizer],
    next_obs: torch.Tensor,#初始观测
    next_done: torch.Tensor,#初始 done 标志
    next_rnn_state: Optional[torch.Tensor],#初始隐藏态
    start_global_step: int = 0,
    start_update: int = 1,
    start_episode_count: int = 0,
    start_episode_returns: Optional[deque] = None,
) -> dict:
    """PPO 主训练循环。"""
    global_step = start_global_step
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    episode_count = start_episode_count
    episode_returns = deque(start_episode_returns or [], maxlen=20)#最近20个回合的奖励
    accum_rewards: np.ndarray = np.zeros(args.num_envs)#每回合累计奖励
    next_checkpoint_episode = None
    if args.save_every_n_episodes > 0:
        interval = int(args.save_every_n_episodes)
        next_checkpoint_episode = ((episode_count // interval) + 1) * interval
    train.last_train_state = _build_train_state(
        global_step, start_update - 1, episode_count,
        optimizer, episode_returns,
    )

    for update in range(start_update, num_updates + 1):
        # 学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = progress * args.learning_rate

        # 收集经验
        rollout, global_step, new_episode_returns = collect_rollout(
            agent, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards,
            reward_normalizer=reward_normalizer,
            rnn_state=next_rnn_state,
        )
        next_obs = rollout.next_obs
        next_done = rollout.next_done
        next_rnn_state = rollout.next_rnn_state
        episode_count += _count_completed_episodes(rollout)

        # GAE 优势估计
        with torch.no_grad():
            advantages, target_values = compute_gae(
                rollout.rewards, rollout.values, rollout.dones,
                rollout.next_value, rollout.next_done,
                args.gamma, args.gae_lambda,
            )

        # 展平 rollout 数据,统一形状为(batch_size, *)
        b_obs = rollout.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = rollout.actions.reshape(-1)
        b_logprobs = rollout.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_target_values = target_values.reshape(-1)
        b_values = rollout.values.reshape(-1)

        clipfracs = []

        if agent.is_recurrent:# 如果是循环神经网络
            seq_len = max(1, min(int(args.recurrent_seq_len), args.num_steps))
            seq_starts = []
            seq_ends = []
            seq_envs = []
            for env_i in range(args.num_envs):#对于每个智能体
                for start_t in range(0, args.num_steps, seq_len):
                    seq_starts.append(start_t)
                    seq_ends.append(min(start_t + seq_len, args.num_steps))
                    seq_envs.append(env_i)#env_i=0,1,2,...

            seq_starts = np.asarray(seq_starts)#序列在样本中的起始位置
            seq_ends = np.asarray(seq_ends)#序列在样本中的结束位置
            seq_envs = np.asarray(seq_envs)
            seq_inds = np.arange(len(seq_starts))#[0,1,2,...,len(seq_starts)-1]
            seqs_per_minibatch = max(1, (len(seq_inds) + args.num_minibatches - 1) // args.num_minibatches)
            """"每个minibatch包含多少个序列,等价于len(seq_inds)/args.num_minibatches向上取整"""

            for epoch in range(args.update_epochs):
                np.random.shuffle(seq_inds)#打乱序列索引

                #对每个minibatch中的所有子序列
                for start in range(0, len(seq_inds), seqs_per_minibatch):
                    mb_seq_inds = seq_inds[start:start + seqs_per_minibatch]#子序列组成一个minibatch的长序列
                    mb_inds, new_logprob, entropy, new_value = evaluate_recurrent_sequences(
                        agent,
                        rollout,
                        seq_starts[mb_seq_inds],
                        seq_ends[mb_seq_inds],
                        seq_envs[mb_seq_inds],
                        device,
                    )

                    # Actor loss
                    pg_loss, approx_kl, clipfrac = compute_actor_loss(
                        new_logprob,
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        args.clip_coef,
                        args.norm_adv,
                    )
                    clipfracs.append(clipfrac)

                    # Critic loss
                    v_loss = compute_critic_loss(
                        new_value,
                        b_target_values[mb_inds],
                        b_values[mb_inds],
                        args.clip_coef,
                        args.clip_vloss,
                    )

                    # 优化 loss
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # KL散度过大时结束本轮更新
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        else:
            b_inds = np.arange(args.batch_size)#batch_indices

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)# 打乱索引

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]#切出小批量mini batch indices

                    # 用当前网络采样动作并计算价值
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                    )
                    new_value = new_value.view(-1)

                    # Actor loss
                    pg_loss, approx_kl, clipfrac = compute_actor_loss(
                        new_logprob,
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        args.clip_coef,
                        args.norm_adv,
                    )
                    clipfracs.append(clipfrac)

                    # Critic loss
                    v_loss = compute_critic_loss(
                        new_value,
                        b_target_values[mb_inds],
                        b_values[mb_inds],
                        args.clip_coef,
                        args.clip_vloss,
                    )

                    # 优化 loss
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # KL散度过大时结束本轮更新
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        # 计算解释方差
        y_pred = b_values.cpu().numpy()
        y_true = b_target_values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        # 日志
        log_ppo(
            writer, global_step, optimizer,
            v_loss, pg_loss, entropy.mean(),
            approx_kl, clipfracs, explained_var,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
            new_episode_returns=new_episode_returns,
        )

        train.last_train_state = _build_train_state(
            global_step, update, episode_count,
            optimizer, episode_returns,
        )

        if next_checkpoint_episode is not None and episode_count >= next_checkpoint_episode:
            ckpt_path = _make_checkpoint_path(args.save_model_path, episode_count)
            if ckpt_path:
                _save_checkpoint(
                    ckpt_path, agent, optimizer,
                    args, reward_normalizer, train.last_train_state,
                )
                print(
                    f"[Checkpoint] episode={episode_count}, "
                    f"update={update}, step={global_step} -> {ckpt_path}"
                )
                _cleanup_old_checkpoints(args.save_model_path, args.max_checkpoints)
            while episode_count >= next_checkpoint_episode:
                next_checkpoint_episode += int(args.save_every_n_episodes)

    return _build_train_state(
        global_step, num_updates, episode_count,
        optimizer, episode_returns,
    )

#  主训练入口
def main():
    # 初始化
    args = Args()
    writer, device, envs, seg, run_name = init_training_setup(args)

    # PPO配置
    args.num_envs = envs.num_envs
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    n_actions = int(envs.single_action_space.n)

    # 智能体 + 优化器
    agent = PPOAgent(n_actions, seg, args).to(device)
    print(f"[PPO] network_type={args.network_type}, params={agent.num_params():,}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 奖励归一化器
    reward_normalizer = None
    if args.reward_norm:
        reward_normalizer = RewardNormalizer(clip=args.reward_clip)
        print(f"[RewardNorm] enabled, clip={args.reward_clip}")

    resume_path = args.resume_from or args.load_model_path
    is_resume = bool(args.resume_from)
    start_global_step, start_update, start_episode_count, episode_returns = (
        load_checkpoint_if_requested(
            resume_path, is_resume,
            agent, optimizer, reward_normalizer,
            device,
        )
    )

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device)#(num_envs,obs_dim)
    next_done = torch.zeros(args.num_envs).to(device)#(num_envs,)
    next_rnn_state = agent.get_initial_state(args.num_envs, device)#(num_envs,rec_state_size)

    final_state = None
    try:
        final_state = train(
            args, agent, envs, optimizer, device, writer,
            reward_normalizer, next_obs, next_done, next_rnn_state,
            start_global_step, start_update, start_episode_count,
            episode_returns,
        )
    except (KeyboardInterrupt, ConnectionResetError) as e:
        print(f"\n[Interrupt] 训练中断 ({type(e).__name__})")
        if args.save_model_path is not None:
            print(f"[Interrupt] 保存检查点到 {args.save_model_path} ...")
            interrupt_state = getattr(
                train,
                "last_train_state",
                _build_train_state(0, 0, 0, optimizer, deque(maxlen=20)),
            )
            _save_checkpoint(
                args.save_model_path, agent, optimizer,
                args, reward_normalizer, interrupt_state,
            )
        return
    finally:
        # Godot 进程在 Ctrl+C 时可能已退出，close 必然失败，加防护防止二次报错
        try:
            envs.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass

    # 正常训练结束后的保存与导出
    if args.save_model_path is not None and final_state is not None:
        save_dict = {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_pt_model(
            args.save_model_path, save_dict, args,
            reward_normalizer, extra=final_state,
        )

if __name__ == "__main__":
    main()
