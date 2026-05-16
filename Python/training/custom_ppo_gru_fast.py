"""
ppo_gru_fast.py — GRU-MLP 优化版 PPO

核心改进：
1. GruMlpEncoder 新增 forward_sequence() 方法，将整段序列一次性通过 GRU
   （而非逐时间步 seq_len=1 调用），消除 ~128× 的 GPU kernel launch 开销。
2. PPOAgent 新增 evaluate_sequence() 方法，封装整段序列的 actor+critic 评估。
3. evaluate_recurrent_sequences 改为按 done 边界切分子序列，
   每段调用一次 evaluate_sequence()，彻底消除原 batch=1 的逐点循环。

进一步优化（2026-05-16）：
4. torch.compile 加速 forward_sequence / evaluate_sequence（方案1）。
5. 向量化索引收集，消去 evaluate_recurrent_sequences 中的 Python 逐元素循环（方案3）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_ppo as _ppo

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# 给 GruMlpEncoder 添加 forward_sequence() 方法
def _gru_forward_sequence(
    self: "_ppo.GruMlpEncoder",
    obs_seq: torch.Tensor,       # (seq_len, obs_dim)
    rnn_state: torch.Tensor,     # (1, L*H)
) -> "tuple[torch.Tensor, torch.Tensor]":
    """整序列前向传播：一次 GRU 调用处理全部时间步。

    Args:
        obs_seq: (seq_len, obs_dim) 一整段连续观测序列
        rnn_state: (1, L*H) 序列起始时刻的 GRU 隐藏状态

    Returns:
        features: (seq_len, output_dim) 每个时间步的特征向量
        h_new: (1, L*H) 序列末时刻的 GRU 隐藏状态
    """
    seq_len = obs_seq.shape[0]

    # 分离观测段：每段 (seq_len, seg_dim)
    s, p, b, e, m = self.obs.split(obs_seq)

    # GRU 输入：(1, seq_len, gru_input_dim)
    gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(0)
    gru_input = self.gru_ln(gru_input)

    # 初始隐藏状态：(L, 1, H)
    h0 = rnn_state.view(1, self.gru_num_layers, self.gru_hidden)# (1, L, H)
    h0 = h0.transpose(0, 1).contiguous()

    # 整段一次通过 GRU
    gru_out, h_new = self.gru(gru_input, h0)
    # gru_out: (1, seq_len, gru_hidden)
    # h_new: (L, 1, H)

    # 奖励球特征：(seq_len, ball_out)
    ball_features = self.ball_net(b)

    # 融合特征：(seq_len, gru_hidden + ball_out)
    fused = torch.cat([gru_out.squeeze(0), ball_features], dim=1)

    # 躯干网络：(seq_len, output_dim)
    features = self.trunk(fused)

    # 最终状态：(1, L*H)
    h_new = h_new.transpose(0, 1).reshape(1, self.recurrent_state_size)

    return features, h_new


# 给 PPOAgent 添加 evaluate_sequence() 方法
def _agent_evaluate_sequence(
    self: "_ppo.PPOAgent",
    obs_seq: torch.Tensor,       # (seq_len, obs_dim)
    action_seq: torch.Tensor,    # (seq_len,)  rollout 中实际执行的动作
    rnn_state: torch.Tensor,     # (1, L*H)
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """整段序列评估：计算每个时间步的 logprob / 熵 / 价值。

    Returns:
        logprobs: (seq_len,)   动作在当前策略下的对数概率
        entropies: (seq_len,)  策略熵
        values: (seq_len,)     状态价值
        next_state: (1, L*H)   序列末时刻的 GRU 状态
    """
    features_seq, next_state = self.encoder.forward_sequence(obs_seq, rnn_state)
    # features_seq: (seq_len, output_dim)

    logits_seq = self.actor(features_seq)          # (seq_len, n_actions)
    probs = Categorical(logits=logits_seq)
    logprobs = probs.log_prob(action_seq)           # (seq_len,)
    entropies = probs.entropy()                     # (seq_len,)
    values = self.critic(features_seq).squeeze(-1)  # (seq_len,)

    return logprobs, entropies, values, next_state

# 重写 evaluate_recurrent_sequences() — 整段批量评估
def evaluate_recurrent_sequences(
    agent: "_ppo.PPOAgent",
    rollout: "_ppo.RolloutData",
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
    seq_envs: np.ndarray,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """按 done 边界切分子序列，整段批量前向传播。

    - 原版：逐时间步 for t in range(start, end) → 每次 batch=1
    - 现在：找到 done 切分点，每段调用 agent.evaluate_sequence()
            -> 整段通过 GRU（seq_len 可达 128），一次调用替代 128 次
    """
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

        # 找到 done 边界：done[t]=True 的位置标记着"新回合开始"
        # rollout.rnn_states[start_t + p, env_i] 在这些位置已被 rollout 阶段置零
        done_seq = rollout.dones[start_t:end_t, env_i]  # (seq_len,)
        done_positions = torch.where(done_seq > 0.5)[0].cpu().tolist()

        # 切分子序列：[0, p1), [p1, p2), ..., [pk, seq_len)
        split_points = [0] + [int(p) for p in done_positions] + [seq_len]#[0, p1, p2, ..., pk, seq_len]

        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end = split_points[i + 1]
            if sub_start >= sub_end:
                continue

            abs_start = start_t + sub_start
            abs_end = start_t + sub_end

            # 该子序列的初始 GRU 状态
            state = rollout.rnn_states[abs_start, env_i].unsqueeze(0).detach()

            # 整段评估
            sub_obs = rollout.obs[abs_start:abs_end, env_i]           # (sub_len, obs_dim)
            sub_actions = rollout.actions[abs_start:abs_end, env_i]   # (sub_len,)

            logprobs, entropies, values, state = agent.evaluate_sequence(
                sub_obs, sub_actions, state,
            )

            # 收集结果（保持时间顺序）
            sub_indices = torch.arange(sub_start, sub_end, device=device) * num_envs + env_i
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


# Monkey-patch：将新方法注入 custom_ppo 模块的类中
_ppo.GruMlpEncoder.forward_sequence = _gru_forward_sequence
_ppo.PPOAgent.evaluate_sequence = _agent_evaluate_sequence
_ppo.evaluate_recurrent_sequences = evaluate_recurrent_sequences

# 导入所有内容
from custom_ppo import *

if __name__ == "__main__":
    main()
