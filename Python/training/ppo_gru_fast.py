"""
ppo_gru_fast.py — GRU-MLP 优化版 PPO

核心改进：
1. evaluate_recurrent_sequences 改为按时间步聚合多序列批量前向传播，
   彻底消除原来 batch=1 的逐点循环，让 GPU 真正并行起来。
2. 数据索引全部使用 PyTorch 高级索引（避免 Python list 和 torch.stack），
   进一步减少 CPU-GPU 同步开销。

推荐超参数（可在 Args 中调整）：
- recurrent_seq_len: 32 或 16（默认 128 太大，导致每个 env 只有 1 条序列）
  减小后每个 env 产生 4~8 条序列，更容易填满 minibatch。
- num_minibatches: 1 或 2（默认 4 在 env 数少时会让每个 minibatch 只有 1 条序列）
  减小后每个 minibatch 序列数翻倍，聚合批量的收益更大。
- update_epochs: 4（默认 8 对循环网络来说更新成本过高，适当降低可大幅提速）
"""

import os
import sys

# 确保能导入同目录的原始 ppo.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始文件的所有定义
from ppo import *

import numpy as np
import torch


def evaluate_recurrent_sequences(
    agent: PPOAgent,
    rollout: RolloutData,
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
    seq_envs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate contiguous rollout chunks while preserving GRU state flow.

    优化版：将同一 minibatch 中多个序列按时间步聚合，批量前向传播，
    避免 batch=1 的逐点循环，显著提升 GPU 利用率。
    """
    if rollout.rnn_states is None:
        raise ValueError("Recurrent PPO update requires rollout.rnn_states.")

    num_envs = rollout.obs.shape[1]
    max_len = int((seq_ends - seq_starts).max())

    # 收集所有序列的初始状态 (n_seqs, L*H)
    init_indices = (seq_starts * num_envs + seq_envs).astype(np.int64)
    states = rollout.rnn_states.view(-1, agent.recurrent_state_size)[init_indices].to(device).detach()

    all_indices = []
    all_logprobs = []
    all_entropies = []
    all_values = []

    for t_offset in range(max_len):
        # 该时间步仍有效的序列掩码
        valid_mask = (seq_starts + t_offset) < seq_ends
        if not valid_mask.any():
            break
        valid_idx = np.where(valid_mask)[0]

        t = seq_starts[valid_idx] + t_offset
        env_i = seq_envs[valid_idx]

        # 直接通过高级索引取批量数据（已经在 GPU 上）
        obs_batch = rollout.obs[t, env_i]
        action_batch = rollout.actions[t, env_i]
        state_batch = states[valid_idx]

        # 批量前向
        _, logprob, entropy, value, next_state = agent.get_action_and_value(
            obs_batch,
            action_batch,
            rnn_state=state_batch,
            return_state=True,
        )

        # 记录结果
        all_indices.extend((t * num_envs + env_i).tolist())
        all_logprobs.append(logprob)
        all_entropies.append(entropy)
        all_values.append(value.view(-1))

        # 根据下一时刻的 done 标志重置状态
        next_t = t + 1
        valid_next = next_t < rollout.obs.shape[0]
        done_next = torch.zeros(len(t), 1, dtype=torch.float32, device=device)
        if valid_next.any():
            done_next[valid_next] = rollout.dones[next_t[valid_next], env_i[valid_next]].view(-1, 1)
        states[valid_idx] = next_state * (1.0 - done_next)

    return (
        torch.tensor(all_indices, dtype=torch.long, device=device),
        torch.cat(all_logprobs, dim=0),
        torch.cat(all_entropies, dim=0),
        torch.cat(all_values, dim=0),
    )


if __name__ == "__main__":
    main()
