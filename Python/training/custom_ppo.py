"""
离散动作 PPO (Proximal Policy Optimization)
"""
import argparse
import os
import copy
import json
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    RewardNormalizer,
    init_training_setup,
    load_full_checkpoint,
    save_pt_model,
)

from custom_ppo_dataclass import PPOArgs, RolloutData
from ppo_networks import DiscreteActorCriticAgent, NetworkType

class PPOAgent(DiscreteActorCriticAgent):
    """Discrete PPO policy/value network."""

    pass


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

    # 追踪本次 rollout 内完成的回合奖励
    new_episode_returns: list[float] = []

    for step in range(num_steps):
        global_step += num_envs
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
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()##概率比例偏离裁剪区间的比例

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
    v_loss: float,
    pg_loss: float,
    entropy_loss: float,
    approx_kl: float,
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
    # 先写 SPS 到 TensorBoard
    sps = int(global_step / (time.time() - start_time)) if start_time > 0 else 0
    writer.add_scalar("charts/SPS", sps, global_step)

    # 写所有指标到 TensorBoard
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss, global_step)
    writer.add_scalar("losses/policy_loss", pg_loss, global_step)
    writer.add_scalar("losses/entropy", entropy_loss, global_step)
    writer.add_scalar("losses/approx_kl", approx_kl, global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

    # 有新 episode 完成时写入回合奖励均值（并行环境取平均）；否则保持上一次记录值
    if new_episode_returns is not None and len(new_episode_returns) > 0:
        writer.add_scalar("charts/episodic_return", float(np.mean(new_episode_returns)), global_step)

    # 终端日志
    #时间统计
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    if len(episode_returns) > 0:
        mean_return = np.mean(np.array(episode_returns))
        if update > 0 and num_updates > 0:
            print(
                f"[Update {update:4d}/{num_updates}] "
                f"return: {mean_return:8.2f}  "
                f"kl: {approx_kl:.4f}"
                f"   training time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
    else:
        if update > 0 and num_updates > 0:
            print(
                f"[Update {update:4d}/{num_updates}]"
                f"kl: {approx_kl:.4f}"
                f"   training time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                )

def _count_completed_episodes(rollout: RolloutData) -> int:
    """统计本次 rollout 中完成的 episode 数量。"""
    done_rows = [rollout.next_done.unsqueeze(0)]#[(1, num_envs)]
    if rollout.dones.shape[0] > 1:
        done_rows.insert(0, rollout.dones[1:])#[(num_steps-1,num_envs),(1,num_envs)]
    dones = torch.cat(done_rows, dim=0)#tensor(num_steps,num_envs)
    #所有时间步至少有1个环境done的次数；
    #一个游戏进程中多个智能体会在同一个时间步 done，应视为一个回合
    #并行环境时所有环境也会在同一个时间步 done，既可以视为一个回合，也可以视为多个回合，这里视为一个回合
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
    args: PPOArgs,
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
    args: PPOArgs,
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
    trial: Optional[Any] = None,
) -> dict:
    """PPO 主训练循环。"""
    global_step = start_global_step
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    episode_count = start_episode_count
    optuna_max_len = 20 if trial is None else 100
    # print(optuna_max_len)
    episode_returns = deque(start_episode_returns or [], maxlen=optuna_max_len)#最近20个回合的奖励
    accum_rewards: np.ndarray = np.zeros(args.num_envs)#每回合累计奖励
    next_checkpoint_episode = None
    if args.save_checkpoint and args.save_every_n_episodes > 0:
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
        pg_losses = []
        v_losses = []
        entropies = []
        approx_kls = []

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

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropies.append(entropy.mean().item())
                    approx_kls.append(approx_kl.item())

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

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropies.append(entropy.mean().item())
                    approx_kls.append(approx_kl.item())

                # KL散度过大时结束本轮更新
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        mean_pg_loss = float(np.mean(pg_losses))
        mean_v_loss = float(np.mean(v_losses))
        mean_entropy = float(np.mean(entropies))
        mean_approx_kl = float(np.mean(approx_kls))

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
            mean_v_loss, mean_pg_loss, mean_entropy,
            mean_approx_kl, clipfracs, explained_var,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
            new_episode_returns=new_episode_returns,
        )

        train.last_train_state = _build_train_state(
            global_step, update, episode_count,
            optimizer, episode_returns,
        )

        if trial is not None and len(episode_returns) > 0:
            objective_value = float(np.mean(np.array(episode_returns)))
            trial.report(objective_value, update)
            if getattr(args, "optuna_prune", False) and trial.should_prune():
                raise optuna.TrialPruned()

        if args.save_checkpoint and next_checkpoint_episode is not None and episode_count >= next_checkpoint_episode:
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

def run_training(args: PPOArgs, trial: Optional[Any] = None) -> dict:
    """Run one PPO training job and return the final training state."""
    writer = None
    envs = None
    reward_normalizer = None
    final_state = None

    try:
        writer, device, envs, seg, run_name = init_training_setup(args)

        # PPO配置
        args.num_envs = envs.num_envs
        args.batch_size = args.num_envs * args.num_steps
        args.minibatch_size = args.batch_size // args.num_minibatches
        if args.minibatch_size <= 0:
            raise ValueError(
                f"Invalid minibatch_size={args.minibatch_size}; "
                f"batch_size={args.batch_size}, num_minibatches={args.num_minibatches}"
            )
        n_actions = int(envs.single_action_space.n)

        # 智能体 + 优化器
        agent = PPOAgent(n_actions, seg, args).to(device)
        print(f"[PPO] network_type={args.network_type}, params={agent.num_params():,}")
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # 奖励归一化器
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

        final_state = train(
            args, agent, envs, optimizer, device, writer,
            reward_normalizer, next_obs, next_done, next_rnn_state,
            start_global_step, start_update, start_episode_count,
            episode_returns, trial=trial,
        )

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

        return final_state
    finally:
        if envs is not None:
            envs.close()
        if writer is not None:
            writer.close()


def _parse_hiddens(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _sample_optuna_args(base_args: PPOArgs, trial: Any) -> PPOArgs:
    args = copy.deepcopy(base_args)
    args.enable_optuna = False
    args.resume_from = None
    args.load_model_path = None
    args.save_model_path = None
    args.save_checkpoint = False
    args.track = False
    # args.seed = int(base_args.seed + trial.number)
    args.exp_name = f"{base_args.exp_name}_optuna_trial_{trial.number}"
    args.total_timesteps = int(base_args.optuna_timesteps)
    args.port_offset = base_args.port_offset + trial.number*args.n_parallel

    args.network_type = trial.suggest_categorical("network_type", ["mlp", "segmented_mlp", "gru_mlp"])
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    args.gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    args.num_minibatches = trial.suggest_categorical("num_minibatches", [2, 4, 8,16])
    args.update_epochs = trial.suggest_categorical("update_epochs", [3, 5, 8, 10])
    args.max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0, 4.0])
    args.num_steps = trial.suggest_categorical("num_steps", [64, 128, 256, 512])
    args.vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0, log=True)

    args.clip_coef = trial.suggest_float("clip_coef", 0.15, 0.3)
    args.ent_coef = trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True)
    args.reward_norm = trial.suggest_categorical("reward_norm", [True, False])
    args.anneal_lr = trial.suggest_categorical("anneal_lr", [False, True])

    if args.reward_norm:
        args.reward_clip = trial.suggest_categorical("reward_clip", [1.0, 2.0,3.0])

    if args.network_type == NetworkType.MLP:
        hidden_choice = trial.suggest_categorical(
            "mlp_hiddens",
            ["128,64", "256,128,64", "256,256,128"],
        )
        args.mlp_hiddens = _parse_hiddens(hidden_choice)

    if args.network_type == NetworkType.SEGMENTED_MLP:
        hidden_choice = trial.suggest_categorical(
            "seg_trunk_hiddens",
            ["128,64", "196,64", "256,128"],
        )
        args.seg_trunk_hiddens = _parse_hiddens(hidden_choice)

    if args.network_type == NetworkType.GRU_MLP:
        args.recurrent_seq_len = trial.suggest_categorical(
            "recurrent_seq_len", [32, 64, 128, 196, 256, 512]
        )
        args.gru_hidden = trial.suggest_categorical("gru_hidden", [64, 128, 196, 256])
        args.gru_num_layers = trial.suggest_int("gru_num_layers", 1, 2)
        hidden_choice = trial.suggest_categorical(
            "gru_trunk_hiddens",
            ["64,32", "128,64", "128,64,32"],
        )
        args.gru_trunk_hiddens = _parse_hiddens(hidden_choice)

    return args


def _objective_from_state(final_state: dict) -> float:
    episode_returns = final_state.get("episode_returns", [])
    if len(episode_returns) == 0:
        return -1e9
    return float(np.mean(np.array(episode_returns, dtype=np.float64)))


def _write_optuna_best_params(args: PPOArgs, study: Any) -> None:
    if not args.optuna_best_params_path:
        return

    dir_name = os.path.dirname(args.optuna_best_params_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    data = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    with open(args.optuna_best_params_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[Optuna] best params saved to {args.optuna_best_params_path}")


def run_optuna(args: PPOArgs):
    # 确保 optuna storage / best_params 父目录存在（避免 sqlite3.OperationalError）
    if args.optuna_storage and args.optuna_storage.startswith("sqlite:///"):
        db_path = args.optuna_storage[len("sqlite:///"):]
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    if args.optuna_best_params_path:
        os.makedirs(os.path.dirname(args.optuna_best_params_path), exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = (
        optuna.pruners.MedianPruner(
            n_warmup_steps=100,
            n_startup_trials=10,
            interval_steps=10,
        )
        if args.optuna_prune
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        trial_args = _sample_optuna_args(args, trial)
        final_state = run_training(trial_args, trial=trial)
        return _objective_from_state(final_state)

    study.optimize(objective, n_trials=args.optuna_trials, n_jobs=1)
    print(f"[Optuna] best value: {study.best_value}")
    print(f"[Optuna] best params: {study.best_params}")
    _write_optuna_best_params(args, study)
    return study

#  主训练入口
def main():
    parser = argparse.ArgumentParser(
        description="离散动作 PPO (Proximal Policy Optimization) 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_path", type=str, default=None,
                        help="游戏环境可执行文件路径")
    parser.add_argument("--config_path", type=str, default=None,
                        help="游戏配置文件路径")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="训练总时间步数")
    parser.add_argument("--save_model_path", type=str, default=None,
                        help="最终模型保存路径")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="中断点恢复路径")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="加载已有模型权重路径")
    parser.add_argument("--port_offset", type=int, default=None,
                        help="Godot 通信端口偏移量 (11008+offset)，多进程并行训练时设不同值避免冲突")
    parser.add_argument("--run_name", type=str, default=None,
                        help="TensorBoard 日志名称，未指定则自动生成")

    cli = parser.parse_args()
    args = PPOArgs()

    # 仅覆盖命令行显式指定的字段，其余沿用 dataclass 默认值
    for field in ("env_path", "config_path", "total_timesteps",
                  "save_model_path", "resume_from", "load_model_path",
                  "port_offset", "run_name"):
        val = getattr(cli, field, None)
        if val is not None:
            setattr(args, field, val)

    if args.enable_optuna:
        run_optuna(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
