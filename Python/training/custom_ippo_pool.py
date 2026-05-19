"""
IPPO + PFSP Opponent Pool (Prioritized Fictitious Self-Play)
=============================================================
基于 custom_ippo.py，引入对手池机制解决多智能体混战的环境非平稳性问题。

三阶段训练:
  Phase 1: custom_ppo.py 课程学习 (4 agent 独立)
  Phase 2: custom_ippo.py 基线训练 (~100万步)
  Phase 3: 本脚本 — 每局只更新一个 agent，其余从对手池 PFSP 采样 (仅推理)

PFSP 采样公式 (AlphaStar, Vinyals et al. Nature 2019):
  P(c) ∝ exp( (0.5 - win_rate(c)) / temperature )
  胜率越低 → 采样概率越高 → 训练 agent 面对有挑战但可学的对手

复用 custom_ippo.py 中的:
  - IPPOAgent, compute_gae, compute_actor_loss, compute_critic_loss
  - evaluate_recurrent_sequences, train_agent_update
  - log_ippo, save_ippo_model, collect_parallel_rollout_ippo
"""

import os
import pathlib
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    RewardNormalizer,
    _serialize_args,
    init_training_setup,
)

from custom_ppo_dataclass import (
    AgentConfig,
    IppoArgs,
    NetworkType,
    OpponentPool,
    PoolEntry,
    RolloutData,
)
from ppo_networks import DiscreteActorCriticAgent

# --- 复用 custom_ippo.py 的核心函数 ---
from custom_ippo import (
    IPPOAgent,
    collect_parallel_rollout_ippo,
    compute_actor_loss,
    compute_critic_loss,
    compute_gae,
    evaluate_recurrent_sequences,
    log_ippo,
    save_ippo_model,
    train_agent_update,
)


# =========================================================================
#  对手池 Checkpoint I/O
# =========================================================================

def save_agent_checkpoint(
    agent: IPPOAgent,
    agent_id: int,
    save_dir: str,
    global_step: int,
) -> str:
    """保存单个 agent 的网络权重 (仅用于对手池推理)。

    Returns:
        保存的 .pt 文件完整路径。
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"agent_{agent_id}_step_{global_step}.pt"

    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "agent_id": agent_id,
        "global_step": global_step,
        "network_type": str(agent.network_type.value
                           if hasattr(agent.network_type, 'value')
                           else agent.network_type),
    }
    torch.save(checkpoint, str(path))
    return str(path)


def load_agent_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """加载单 agent checkpoint 并返回其内容字典。

    返回 dict 包含: agent_state_dict, agent_id, global_step, network_type。
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


# =========================================================================
#  对手池管理
# =========================================================================

def _init_opponent_pool(args: IppoArgs) -> OpponentPool:
    """根据 IppoArgs 创建并初始化对手池。"""
    pool = OpponentPool(
        max_size=args.pool_max_size,
        save_interval=args.pool_save_interval,
        elo_k_factor=args.pool_elo_k_factor,
        win_rate_ema=args.pool_win_rate_ema,
        epsilon=args.pool_epsilon,
        temperature=args.pool_pfsp_temperature,
        use_recency_bias=args.pool_use_recency_bias,
        recency_scale=args.pool_recency_scale,
    )
    return pool


def _seed_pool_from_checkpoints(
    pool: OpponentPool,
    checkpoint_dir: str,
    device: torch.device,
    n_agents: int,
) -> None:
    """从 Phase 2 的 checkpoint 目录加载种子快照到对手池。

    期望目录结构: checkpoint_dir/agent_{id}_step_{step}.pt
    如果目录为空或不存在，不添加任何条目（pool 将在 Phase 3 首次保存时填充）。
    """
    ckpt_dir = pathlib.Path(checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"[Pool] Seed checkpoint directory not found: {checkpoint_dir}, "
              f"pool will be populated during Phase 3 training.")
        return

    pt_files = sorted(ckpt_dir.glob("agent_*_step_*.pt"))
    if not pt_files:
        print(f"[Pool] No checkpoint files found in {checkpoint_dir}, "
              f"pool will be populated during Phase 3 training.")
        return

    # 收集所有可用步数，取最新的若干组 (每组 4 个 agent)
    step_agent_map: dict[int, dict[int, str]] = {}
    for f in pt_files:
        try:
            # 文件名格式: agent_{id}_step_{step}.pt
            parts = f.stem.split("_")
            agent_id = int(parts[1])
            step = int(parts[3])
            step_agent_map.setdefault(step, {})[agent_id] = str(f)
        except (IndexError, ValueError):
            continue

    sorted_steps = sorted(step_agent_map.keys())
    # 取最近的 3 组 (约 3 组 × 4 agent = 12 条目，为后续快照留空间)
    recent_steps = sorted_steps[-3:] if len(sorted_steps) >= 3 else sorted_steps

    for step in recent_steps:
        for agent_id in range(n_agents):
            if agent_id in step_agent_map[step]:
                ckpt_data = load_agent_checkpoint(
                    step_agent_map[step][agent_id], device,
                )
                entry = PoolEntry(
                    checkpoint_path=step_agent_map[step][agent_id],
                    agent_id=agent_id,
                    global_step=int(ckpt_data.get("global_step", step)),
                )
                pool.entries.append(entry)
                print(f"[Pool] Seeded: agent_{agent_id} @ step {step}")

    print(f"[Pool] Seeded {len(pool.entries)} entries from {checkpoint_dir}")


def _pfsp_sample(
    pool: OpponentPool,
    slot: int,
    rng: np.random.Generator,
) -> Optional[PoolEntry]:
    """从对手池的指定 slot 中按 PFSP 优先级采样一个对手快照。

    采样公式:
      P(c) ∝ max(0, exp( (0.5 - win_rate(c)) / temperature ))

    另有 epsilon 概率均匀采样，可选 recency bias。

    Returns:
        采样到的 PoolEntry，如果该 slot 无可用条目返回 None。
    """
    candidates = [e for e in pool.entries if e.agent_id == slot]
    if not candidates:
        return None

    n = len(candidates)

    # Epsilon 探索: 均匀随机
    if rng.random() < pool.epsilon:
        idx = rng.integers(0, n)
        return candidates[idx]

    # PFSP 权重计算
    weights = np.zeros(n, dtype=np.float64)
    for i, entry in enumerate(candidates):
        # 核心 PFSP: (0.5 - win_rate) — 胜率越低权重越高
        pfsp_score = 0.5 - entry.win_rate
        if pfsp_score <= 0:
            # 胜率 >= 0.5 的对手也有被采样的机会，只是权重很低
            pfsp_score = 1e-6
        weights[i] = np.exp(pfsp_score / pool.temperature)

        # Recency bias: 新快照获得轻微加权
        if pool.use_recency_bias:
            recency = max(0.1, 1.0 / (1.0 + entry.age / pool.recency_scale))
            weights[i] *= recency

    # Softmax 归一化
    total = weights.sum()
    if total <= 0:
        return candidates[rng.integers(0, n)]

    probs = weights / total
    idx = rng.choice(n, p=probs)
    return candidates[idx]


def _prune_pool(pool: OpponentPool, n_agents: int):
    """裁剪对手池: 当某 slot 超过 max_size/n_agents 时淘汰最弱条目。"""
    max_per_slot = pool.max_size // n_agents
    if max_per_slot <= 0:
        return

    for slot in range(n_agents):
        slot_entries = [(i, e) for i, e in enumerate(pool.entries)
                       if e.agent_id == slot]
        if len(slot_entries) <= max_per_slot:
            continue

        # 优先淘汰胜率最高（太弱，训练价值低）的条目
        slot_entries.sort(key=lambda x: x[1].win_rate, reverse=True)
        to_remove = slot_entries[max_per_slot:]
        # 从池中移除
        for idx, entry in sorted(to_remove, key=lambda x: x[0], reverse=True):
            pool.entries.pop(idx)


def _add_current_snapshots(
    pool: OpponentPool,
    agents: list[IPPOAgent],
    save_dir: str,
    global_step: int,
    n_agents: int,
):
    """保存当前所有 agent 权重的快照到磁盘并加入对手池。"""
    for agent_id in range(n_agents):
        path = save_agent_checkpoint(
            agents[agent_id], agent_id, save_dir, global_step,
        )
        entry = PoolEntry(
            checkpoint_path=path,
            agent_id=agent_id,
            global_step=global_step,
        )
        pool.entries.append(entry)

    pool.last_save_step = global_step
    _prune_pool(pool, n_agents)
    print(f"[Pool] Added {n_agents} snapshots @ step {global_step}, "
          f"total entries: {len(pool.entries)}")


# =========================================================================
#  对手池 Rollout 收集 (修改自 collect_parallel_rollout_ippo)
# =========================================================================

def collect_pool_rollout(
    agents_cfg: list[AgentConfig],
    agents: list[IPPOAgent],
    envs: GodotDiscreteEnvWrapper,
    rollout_steps: int,
    device: torch.device,
    next_obs_all: torch.Tensor,
    next_done_all: torch.Tensor,
    global_step: int,
    episode_returns: list[deque],
    accum_rewards: np.ndarray,
    reward_normalizers: list[Optional[RewardNormalizer]],
    rnn_states: list[Optional[torch.Tensor]],
    step_increment: int,
) -> tuple[list[RolloutData], int, list[Optional[torch.Tensor]], list[list[float]]]:
    """对手池模式下的并行 rollout 收集。

    所有 agent (训练+对手) 均使用 get_action_and_value 进行推理，
    但仅 `agents_cfg[i].train=True` 的 agent 会记录 next_value（供后续更新用）。
    对手 agent 的权重通过 pool 管理器在调用前加载，episode return 也追踪（用于胜率计算）。
    """
    n_agents = len(agents_cfg)
    total_slots = envs.num_envs
    n_game_envs = total_slots // n_agents
    obs_shape = envs.single_observation_space.shape
    obs_dim = obs_shape[0]

    # 预分配 per-agent 缓冲区
    buffers: list[dict] = []
    for _ in range(n_agents):
        buffers.append({
            "obs": torch.zeros((rollout_steps, n_game_envs) + obs_shape, device=device),
            "actions": torch.zeros((rollout_steps, n_game_envs), dtype=torch.long, device=device),
            "logprobs": torch.zeros((rollout_steps, n_game_envs), device=device),
            "rewards": torch.zeros((rollout_steps, n_game_envs), device=device),
            "dones": torch.zeros((rollout_steps, n_game_envs), device=device),
            "values": torch.zeros((rollout_steps, n_game_envs), device=device),
            "rnn_states": None,
        })

    for i, agent in enumerate(agents):
        if agent.is_recurrent:
            buffers[i]["rnn_states"] = torch.zeros(
                (rollout_steps, n_game_envs, agent.recurrent_state_size),
                device=device,
            )
            if rnn_states[i] is None:
                rnn_states[i] = agent.get_initial_state(n_game_envs, device)

    next_obs_all = next_obs_all.clone()
    next_done_all = next_done_all.clone()
    new_episode_returns: list[list[float]] = [[] for _ in range(n_agents)]

    for step in range(rollout_steps):
        global_step += step_increment
        next_obs_by_env = next_obs_all.view(n_game_envs, n_agents, obs_dim)
        next_done_by_env = next_done_all.view(n_game_envs, n_agents)
        actions_by_env = np.full((n_game_envs, n_agents), 4, dtype=np.int64)

        for i in range(n_agents):
            obs_i = next_obs_by_env[:, i, :]  # shape (n_game_envs, obs_dim)
            done_i = next_done_by_env[:, i]   # shape (n_game_envs,)
            buffers[i]["obs"][step] = obs_i
            buffers[i]["dones"][step] = done_i

            with torch.no_grad():
                if agents[i].is_recurrent:
                    rnn_states[i] = rnn_states[i] * (1.0 - done_i).view(-1, 1)
                    buffers[i]["rnn_states"][step] = rnn_states[i]

                action, logprob, _, value, next_rnn_state = agents[i].get_action_and_value(
                    obs_i,
                    rnn_state=rnn_states[i],
                    return_state=True,
                )
                buffers[i]["actions"][step] = action
                buffers[i]["logprobs"][step] = logprob
                buffers[i]["values"][step] = value.flatten()
                if agents[i].is_recurrent:
                    rnn_states[i] = next_rnn_state.detach()

            actions_by_env[:, i] = action.cpu().numpy().astype(np.int64)

        # 所有 agent 一起执行动作
        next_obs_raw, rewards_raw, terminations, truncations, infos = envs.step(
            actions_by_env.reshape(-1)
        )
        dones_raw = np.logical_or(terminations, truncations)
        rewards_by_env = np.asarray(rewards_raw, dtype=np.float32).reshape(
            n_game_envs, n_agents
        )
        dones_by_env = np.asarray(dones_raw, dtype=bool).reshape(n_game_envs, n_agents)

        next_obs_all = torch.tensor(
            np.array(next_obs_raw, dtype=np.float32), device=device
        )
        next_done_all = torch.tensor(dones_raw, dtype=torch.float32, device=device)

        for i in range(n_agents):
            reward_i = rewards_by_env[:, i]
            if agents_cfg[i].train and reward_normalizers[i] is not None:
                reward_i_norm = reward_normalizers[i].normalize_array(reward_i)
                reward_normalizers[i].update_array(reward_i)
            else:
                reward_i_norm = reward_i

            buffers[i]["rewards"][step] = torch.tensor(
                reward_i_norm, dtype=torch.float32, device=device
            )

            # 追踪所有 agent 的 episode return（训练+对手），用于胜率计算
            accum_rewards[i] += reward_i.astype(np.float64)
            for env_i, done in enumerate(dones_by_env[:, i]):
                if done:
                    ep_ret = float(accum_rewards[i, env_i])
                    episode_returns[i].append(ep_ret)
                    new_episode_returns[i].append(ep_ret)
                    accum_rewards[i, env_i] = 0.0

    # 包装为 RolloutData
    rollouts = []
    next_obs_by_env = next_obs_all.view(n_game_envs, n_agents, obs_dim)
    next_done_by_env = next_done_all.view(n_game_envs, n_agents)
    for i in range(n_agents):
        next_val = None
        if agents_cfg[i].train:
            with torch.no_grad():
                next_val = agents[i].get_value(
                    next_obs_by_env[:, i, :],
                    rnn_state=rnn_states[i],
                ).flatten()

        rollouts.append(RolloutData(
            obs=buffers[i]["obs"],
            actions=buffers[i]["actions"],
            logprobs=buffers[i]["logprobs"],
            rewards=buffers[i]["rewards"],
            dones=buffers[i]["dones"],
            values=buffers[i]["values"],
            next_obs=next_obs_by_env[:, i, :],
            next_done=next_done_by_env[:, i],
            next_value=next_val,
            rnn_states=buffers[i]["rnn_states"],
            next_rnn_state=rnn_states[i],
        ))

    return rollouts, global_step, rnn_states, new_episode_returns


# =========================================================================
#  胜率 / ELO 统计追踪
# =========================================================================

def _update_opponent_stats(
    pool: OpponentPool,
    selections: dict,          # {slot: PoolEntry} 本 update 采样的对手
    episode_returns: list[deque],
    training_agent_id: int,
):
    """根据最新 episode returns 更新对手池条目的胜率和 ELO。

    '胜' 定义: 训练 agent 的 episode return > 对手 episode return 均值的加权比较。
    由于可能多局游戏合并统计，使用最近 episode return 的滑动平均做近似。
    """
    # 获取训练 agent 最近的 episode return 均值
    train_returns = list(episode_returns[training_agent_id])
    if not train_returns:
        return
    train_avg = np.mean(train_returns)

    # 更新每个被采样的对手条目
    for slot, entry in selections.items():
        if entry is None:
            continue
        opp_returns = list(episode_returns[slot])
        if not opp_returns:
            continue
        opp_avg = np.mean(opp_returns)

        # 训练 agent 是否 "胜出"
        outcome = 1.0 if train_avg > opp_avg else 0.0

        # EMA 胜率更新
        ema = pool.win_rate_ema
        entry.win_rate = ema * entry.win_rate + (1.0 - ema) * outcome
        entry.n_games += 1

        # ELO 更新
        expected = 1.0 / (1.0 + 10.0 ** ((entry.elo_rating - pool.training_agent_elo) / 400.0))
        entry.elo_rating += pool.elo_k_factor * (outcome - expected)

    # 更新训练 agent 的 ELO (取所有对手 ELO 的均值作为参考)
    opponent_elos = [e.elo_rating for e in selections.values() if e is not None]
    if opponent_elos:
        avg_opp_elo = np.mean(opponent_elos)
        expected_train = 1.0 / (1.0 + 10.0 ** ((avg_opp_elo - pool.training_agent_elo) / 400.0))
        # 用所有对手的平均 outcome 更新训练 agent ELO
        outcomes = []
        for entry in selections.values():
            if entry is None:
                continue
            opp_ret = list(episode_returns[entry.agent_id])
            outcomes.append(1.0 if (list(episode_returns[training_agent_id]) and
                            np.mean(list(episode_returns[training_agent_id])) >
                            np.mean(opp_ret)) else 0.0)
        if outcomes:
            avg_outcome = np.mean(outcomes)
            pool.training_agent_elo += pool.elo_k_factor * (avg_outcome - expected_train)


def _advance_pool_ages(pool: OpponentPool, steps: int):
    """递增池中所有条目的 age 计数器。"""
    for entry in pool.entries:
        entry.age += steps


# =========================================================================
#  对手池 TensorBoard 日志
# =========================================================================

def _log_opponent_pool(
    writer,
    pool: OpponentPool,
    global_step: int,
    n_agents: int,
    training_agent_id: int,
):
    """将对手池的统计指标写入 TensorBoard。"""
    writer.add_scalar("pool/total_entries", len(pool.entries), global_step)
    writer.add_scalar("pool/training_agent_elo", pool.training_agent_elo, global_step)

    for slot in range(n_agents):
        slot_entries = [e for e in pool.entries if e.agent_id == slot]
        writer.add_scalar(f"pool/agent_{slot}/entries", len(slot_entries), global_step)

        if slot_entries:
            avg_elo = np.mean([e.elo_rating for e in slot_entries])
            avg_wr = np.mean([e.win_rate for e in slot_entries])
            avg_age = np.mean([e.age for e in slot_entries])
            avg_games = np.mean([e.n_games for e in slot_entries])
            writer.add_scalar(f"pool/agent_{slot}/avg_elo", avg_elo, global_step)
            writer.add_scalar(f"pool/agent_{slot}/avg_win_rate", avg_wr, global_step)
            writer.add_scalar(f"pool/agent_{slot}/avg_age", avg_age, global_step)
            writer.add_scalar(f"pool/agent_{slot}/avg_games", avg_games, global_step)


# =========================================================================
#  对手池模式下的终端日志增强
# =========================================================================

def _print_pool_status(
    pool: OpponentPool,
    n_agents: int,
    training_agent_id: int,
    update: int,
    selections: dict,
):
    """打印对手池的终端状态摘要。"""
    lines = [f"\n[Pool Status - Update {update}]"]
    lines.append(f"  Training agent: {training_agent_id} (ELO: {pool.training_agent_elo:.0f})")
    lines.append(f"  Total entries: {len(pool.entries)}")

    for slot in range(n_agents):
        slot_entries = [e for e in pool.entries if e.agent_id == slot]
        marker = " <-- TRAIN" if slot == training_agent_id else ""
        sel = selections.get(slot)
        sel_info = ""
        if sel is not None:
            sel_info = (f" | Sampled: step={sel.global_step}, "
                       f"ELO={sel.elo_rating:.0f}, WR={sel.win_rate:.2f}")
        lines.append(f"  Agent {slot}{marker}: {len(slot_entries)} entries{sel_info}")

    print("\n".join(lines))


# =========================================================================
#  对手池主训练循环
# =========================================================================

def train_with_opponent_pool(
    args: IppoArgs,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    envs: GodotDiscreteEnvWrapper,
    device: torch.device,
    writer,
    reward_normalizers: list[Optional[RewardNormalizer]],
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    pool: OpponentPool,
) -> None:
    """对手池模式主训练循环。

    每个 update:
      1. 轮换选择训练 agent
      2. 从对手池 PFSP 采样其余 agent 的对手权重
      3. 加载对手权重到对应 agent 网络 (eval 模式)
      4. 收集 rollout (训练 agent 记录 value/logprob, 对手仅推理)
      5. 只更新训练 agent 的网络
      6. 更新对手池统计 (胜率/ELO)
      7. 定期保存快照到池
    """
    n_agents = len(args.agent_configs)
    n_game_envs = args.num_game_envs
    global_step = 0
    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    # 按 count_steps_by 计算总更新次数
    if args.count_steps_by == "env_steps":
        num_updates = args.total_timesteps // (n_game_envs * args.num_steps)
        step_increment = n_game_envs
    elif args.count_steps_by == "agent_steps":
        num_updates = args.total_timesteps // (n_game_envs * n_agents * args.num_steps)
        step_increment = args.num_envs
    else:
        raise ValueError(
            f"Unknown count_steps_by='{args.count_steps_by}'. "
            "Expected 'env_steps' or 'agent_steps'."
        )

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states: list[Optional[torch.Tensor]] = [
        agent.get_initial_state(n_game_envs, device) if agent.is_recurrent else None
        for agent in agents
    ]

    print(f"\n[Pool Training] Starting opponent pool training")
    print(f"  Updates: {num_updates}, Steps/update: {args.num_steps}")
    print(f"  Parallel games: {n_game_envs}, Agents: {n_agents}")
    print(f"  Pool size limit: {pool.max_size}, "
          f"Save interval: {pool.save_interval}")
    print(f"  PFSP: temperature={pool.temperature}, "
          f"epsilon={pool.epsilon}, recency_bias={pool.use_recency_bias}")

    for update in range(1, num_updates + 1):
        # 1. 选择训练 agent
        if args.pool_training_agent_selection == "round_robin":
            training_agent_id = (update - 1) % n_agents
        elif args.pool_training_agent_selection == "random":
            training_agent_id = int(rng.integers(0, n_agents))
        else:
            raise ValueError(
                f"Unknown pool_training_agent_selection: "
                f"'{args.pool_training_agent_selection}'"
            )

        # 2. 设置 agent_configs: 只有训练 agent 标记为 train=True
        agent_configs = []
        for i in range(n_agents):
            cfg = args.agent_configs[i]
            agent_configs.append(AgentConfig(
                agent_id=cfg.agent_id,
                train=(i == training_agent_id),
                network_type=cfg.network_type,
                self_hidden=cfg.self_hidden,
                player_hidden=cfg.player_hidden,
                ball_hidden=cfg.ball_hidden,
                enemy_hidden=cfg.enemy_hidden,
                map_hidden=cfg.map_hidden,
                trunk_hiddens=cfg.trunk_hiddens,
                mlp_hiddens=cfg.mlp_hiddens,
                gru_hidden=cfg.gru_hidden,
                gru_num_layers=cfg.gru_num_layers,
                gru_input_layernorm=cfg.gru_input_layernorm,
                learning_rate=cfg.learning_rate,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_coef=cfg.clip_coef,
                ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                reward_norm=cfg.reward_norm,
                reward_clip=cfg.reward_clip,
            ))

        # 3. 从对手池采样 + 加载权重
        selections: dict[int, Optional[PoolEntry]] = {}
        for slot in range(n_agents):
            if slot == training_agent_id:
                selections[slot] = None
                continue
            entry = _pfsp_sample(pool, slot, rng)
            selections[slot] = entry

            if entry is not None:
                # 加载对手权重到 agent 网络
                ckpt = load_agent_checkpoint(entry.checkpoint_path, device)
                agents[slot].load_state_dict(ckpt["agent_state_dict"])
                agents[slot].eval()
                # 重置 RNN 状态（对手权重切换）
                if agents[slot].is_recurrent:
                    rnn_states[slot] = agents[slot].get_initial_state(
                        n_game_envs, device,
                    )
            else:
                # 池中无该 slot 条目: 使用当前权重 (即初始/预设网络)
                agents[slot].eval()

        # 训练 agent 保持训练模式
        agents[training_agent_id].train()

        # 4. 学习率退火 (仅训练 agent)
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            optimizers[training_agent_id].param_groups[0]["lr"] = (
                progress * args.agent_configs[training_agent_id].learning_rate
            )

        # 5. 收集 rollout
        rollouts, global_step, rnn_states, new_episode_returns = collect_pool_rollout(
            agent_configs, agents, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards, reward_normalizers,
            rnn_states, step_increment,
        )

        next_obs = torch.stack([r.next_obs for r in rollouts], dim=1).reshape(
            args.num_envs, -1
        )
        next_done = torch.stack([r.next_done for r in rollouts], dim=1).reshape(
            args.num_envs
        )

        # 6. 只更新训练 agent
        losses: list[Optional[dict]] = [None] * n_agents
        explained_vars: list[float] = [0.0] * n_agents

        metrics = train_agent_update(
            agents[training_agent_id],
            optimizers[training_agent_id],
            rollouts[training_agent_id],
            agent_configs[training_agent_id],
            args,
            device,
        )
        losses[training_agent_id] = metrics
        explained_vars[training_agent_id] = metrics.get("explained_var", 0.0)

        # 7. 更新对手池统计
        _update_opponent_stats(pool, selections, episode_returns, training_agent_id)
        _advance_pool_ages(pool, args.num_steps * step_increment)

        # 8. 日志 (复用 custom_ippo 的 log_ippo)
        log_ippo(
            writer, global_step, agent_configs, optimizers,
            losses, explained_vars,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
            new_episode_returns=new_episode_returns,
        )

        # 池指标日志
        if update % args.pool_log_every_n_updates == 0:
            _log_opponent_pool(writer, pool, global_step, n_agents, training_agent_id)
            _print_pool_status(pool, n_agents, training_agent_id, update, selections)

        # 9. 定期保存快照到池
        if global_step - pool.last_save_step >= pool.save_interval:
            pool_dir = args.pool_checkpoint_dir or os.path.join(
                args.experiment_dir, "pool_checkpoints"
            )
            _add_current_snapshots(
                pool, agents, pool_dir, global_step, n_agents,
            )


# =========================================================================
#  主入口
# =========================================================================

def main():
    """对手池 IPPO 训练入口。"""
    args = IppoArgs()
    writer, device, envs, seg, run_name = init_training_setup(args)

    n_agents = len(args.agent_configs)
    args.num_agents = n_agents
    args.num_envs = envs.num_envs
    if args.num_envs % n_agents != 0:
        raise ValueError(
            "IPPO expects Godot training slots to be n_parallel * n_agents: "
            f"envs.num_envs={args.num_envs}, len(args.agent_configs)={n_agents}."
        )
    args.num_game_envs = args.num_envs // n_agents
    if args.num_game_envs != args.n_parallel:
        raise ValueError(
            "Godot env slot count does not match the configured parallel game count: "
            f"envs.num_envs={args.num_envs}, n_agents={n_agents}, "
            f"derived n_parallel={args.num_game_envs}, "
            f"args.n_parallel={args.n_parallel}."
        )

    obs_shape = envs.single_observation_space.shape
    if len(obs_shape) != 1 or obs_shape[0] != seg.total:
        raise ValueError(
            f"Observation dimension mismatch: env observation shape={obs_shape}, "
            f"configured segment total={seg.total}."
        )

    args.batch_size = args.num_game_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size <= 0:
        raise ValueError(
            "num_minibatches is too large for the configured IPPO batch: "
            f"batch_size={args.batch_size}, "
            f"num_minibatches={args.num_minibatches}."
        )

    n_actions = int(envs.single_action_space.n)

    # 创建 per-agent 网络、优化器、奖励归一化器
    agents: list[IPPOAgent] = []
    optimizers: list[optim.Optimizer] = []
    reward_normalizers: list[Optional[RewardNormalizer]] = []

    for cfg in args.agent_configs:
        agent = IPPOAgent(n_actions, seg, cfg).to(device)

        tag = f"[Agent {cfg.agent_id}]"
        print(f"{tag} network_type={cfg.network_type}, "
              f"params={agent.num_params():,}")

        agents.append(agent)
        optimizers.append(
            optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        )

        if cfg.train and cfg.reward_norm:
            reward_normalizers.append(RewardNormalizer(clip=cfg.reward_clip))
        else:
            reward_normalizers.append(None)

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(
        np.array(next_obs_array, dtype=np.float32), device=device
    )
    if next_obs.shape[0] != args.num_envs:
        raise ValueError(
            "Reset observation count does not match Godot env slots: "
            f"next_obs.shape[0]={next_obs.shape[0]}, "
            f"envs.num_envs={args.num_envs}."
        )
    next_done = torch.zeros(args.num_envs, device=device)

    # 初始化对手池
    pool = _init_opponent_pool(args)

    # 从 Phase 2 checkpoint 目录初始化池
    if args.pool_initial_checkpoint_dir is not None:
        _seed_pool_from_checkpoints(
            pool, args.pool_initial_checkpoint_dir, device, n_agents,
        )

    # 如果没有种子 checkpoint，立即用当前初始权重建立首个快照
    if len(pool.entries) == 0:
        print("[Pool] No seed entries found, saving initial snapshots...")
        pool_dir = args.pool_checkpoint_dir or os.path.join(
            args.experiment_dir, "pool_checkpoints"
        )
        _add_current_snapshots(pool, agents, pool_dir, 0, n_agents)

    # 开始训练
    train_with_opponent_pool(
        args, agents, optimizers, envs, device, writer,
        reward_normalizers, next_obs, next_done,
        pool,
    )

    # 训练结束后的模型保存 (复用 custom_ippo 的 save_ippo_model)
    if args.save_model_path is not None:
        save_ippo_model(
            args.save_model_path, agents, optimizers,
            reward_normalizers, args,
        )

    # 资源清理
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
