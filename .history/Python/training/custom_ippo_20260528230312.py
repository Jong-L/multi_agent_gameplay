"""
Independent PPO (IPPO),离散动作空间
每个智能体拥有独立的 PPO 网络、优化器和超参数配置，
在同一个 Godot 环境中同时训练多个智能体。
支持多个 Godot 并行环境；每个环境内的同编号智能体数据用于更新同一个 agent 网络。
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
    load_full_checkpoint,
)

from custom_ppo_dataclass import AgentConfig, IppoArgs, RolloutData
from ppo_networks import DiscreteActorCriticAgent

class IPPOAgent(DiscreteActorCriticAgent):
    """Independent PPO policy/value network."""

    pass


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
    rollout: RolloutData,
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
    seq_envs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate recurrent chunks with the same sequence GRU path used by PPO."""
    if rollout.rnn_states is None:
        raise ValueError("Recurrent agent update requires rollout.rnn_states.")

    num_envs = rollout.obs.shape[1]
    all_indices = []
    all_logprobs = []
    all_entropies = []
    all_values = []

    for start, end, env_i in zip(seq_starts, seq_ends, seq_envs):
        start = int(start)
        end = int(end)
        env_i = int(env_i)
        seq_len = end - start

        done_seq = rollout.dones[start:end, env_i]
        done_positions = torch.where(done_seq > 0.5)[0].cpu().tolist()
        split_points = [0] + [int(p) for p in done_positions] + [seq_len]

        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end = split_points[i + 1]
            if sub_start >= sub_end:
                continue

            abs_start = start + sub_start
            abs_end = start + sub_end

            state = rollout.rnn_states[abs_start, env_i].unsqueeze(0).detach()
            sub_obs = rollout.obs[abs_start:abs_end, env_i]
            sub_actions = rollout.actions[abs_start:abs_end, env_i]

            logprobs, entropies, values, state = agent.evaluate_sequence(
                sub_obs, sub_actions, state,
            )

            all_indices.append(
                torch.arange(abs_start, abs_end, device=device) * num_envs + env_i
            )
            all_logprobs.append(logprobs)
            all_entropies.append(entropies)
            all_values.append(values)

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_logprobs, dim=0),
        torch.cat(all_entropies, dim=0),
        torch.cat(all_values, dim=0),
    )

def collect_parallel_rollout_ippo(
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
    """Collect per-agent IPPO data from n_parallel game instances.

    Slot layout is assumed to be env_index * n_agents + agent_index.
    Each returned RolloutData belongs to one logical agent and contains all
    parallel game instances on the second tensor dimension.
    """
    n_agents = len(agents_cfg)
    total_slots = envs.num_envs

    n_game_envs = total_slots // n_agents
    obs_shape = envs.single_observation_space.shape
    obs_dim = obs_shape[0]

    buffers: list[dict] = []#（n_agents）每个智能体的缓冲区
    for _ in range(n_agents):
        buffers.append({
            "obs": torch.zeros((rollout_steps, n_game_envs) + obs_shape, device=device),#（T,n_games,obs_dim)
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
            obs_i = next_obs_by_env[:, i, :]# shape (n_game_envs, obs_dim)
            done_i = next_done_by_env[:, i]# shape (n_game_envs,)
            buffers[i]["obs"][step] = obs_i
            buffers[i]["dones"][step] = done_i

            use_policy_action = agents_cfg[i].train or getattr(
                agents_cfg[i], "act_when_not_training", False
            )
            if use_policy_action:
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

                actions_by_env[:, i] = action.cpu().numpy().astype(np.int64)# shape (n_game_envs,)
            else:
                random_actions = np.random.randint(0, envs.single_action_space.n, size=n_game_envs, dtype=np.int64)
                # random_actions=np.full(n_game_envs,4,dtype=np.float64)
                buffers[i]["actions"][step] = torch.tensor(random_actions, device=device)
                actions_by_env[:, i] = random_actions
                buffers[i]["logprobs"][step] = 0.0
                buffers[i]["values"][step] = 0.0

        next_obs_raw, rewards_raw, terminations, truncations, infos = envs.step(
            actions_by_env.reshape(-1)# shape (n_game_envs * n_agents,)
        )
        dones_raw = np.logical_or(terminations, truncations)
        rewards_by_env = np.asarray(rewards_raw, dtype=np.float32).reshape(
            n_game_envs, n_agents
        )
        dones_by_env = np.asarray(dones_raw, dtype=bool).reshape(n_game_envs, n_agents)

        next_obs_all = torch.tensor(np.array(next_obs_raw, dtype=np.float32), device=device)
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

            if agents_cfg[i].train:
                accum_rewards[i] += reward_i.astype(np.float64)
                for env_i, done in enumerate(dones_by_env[:, i]):
                    if done:
                        ep_ret = float(accum_rewards[i, env_i])
                        episode_returns[i].append(ep_ret)
                        new_episode_returns[i].append(ep_ret)
                        accum_rewards[i, env_i] = 0.0

    rollouts = []
    next_obs_by_env = next_obs_all.view(n_game_envs, n_agents, obs_dim)
    next_done_by_env = next_done_all.view(n_game_envs, n_agents)
    for i in range(n_agents):#合并为rollout
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
    new_episode_returns: Optional[list[list[float]]] = None,
) -> list[str]:
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
            # 仅在新 episode 完成时写入并行环境均值，否则保持上一回合结果不变
            if new_episode_returns is not None and len(new_episode_returns[i]) > 0:
                writer.add_scalar(f"{tag}/charts/episodic_return", float(np.mean(new_episode_returns[i])), global_step)

    # 终端日志
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    if update > 0 and num_updates > 0:
        return_strs = []
        for i in range(len(agents_cfg)):
            if agents_cfg[i].train and len(episode_returns[i]) > 0:
                mean_ret = np.mean(np.array(episode_returns[i]))
                return_strs.append(f"p{i}:{mean_ret:.1f}")

        returns_summary = "  ".join(return_strs)

        kl_summary = "  ".join(
            f"p{i}:{losses[i]['approx_kl']:.4f}"
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
            f"  training time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        )


#  模型保存
def _make_agent_model_path(save_path: str, agent_id: int) -> pathlib.Path:
    save_path = pathlib.Path(save_path).with_suffix(".pt")
    return save_path.with_name(f"{save_path.stem}_agent{agent_id}{save_path.suffix}")


def save_ippo_model(
    save_path: str,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    reward_normalizers: list[Optional[RewardNormalizer]],
    args: IppoArgs,
    extra: Optional[dict] = None,
    train_only: bool = True,
) -> list[str]:
    save_path = pathlib.Path(save_path).with_suffix(".pt") #  确保保存路径以.pt结尾
    saved_paths = []
    for i, agent in enumerate(agents):
        if train_only and not args.agent_configs[i].train:
            continue
        agent_id = args.agent_configs[i].agent_id
        agent_save_path = _make_agent_model_path(str(save_path), agent_id)
        agent_save_path.parent.mkdir(parents=True, exist_ok=True) 
        payload = {
            "args": _serialize_args(args),
            "agent_id": agent_id,
            "agent_state_dict": agent.state_dict(), 
            "optimizer_state_dict": optimizers[i].state_dict(), 
        }
        if reward_normalizers[i] is not None:
            payload["reward_normalizer"] = reward_normalizers[i].state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, str(agent_save_path))
        saved_paths.append(str(agent_save_path))
        print(f"[Save] IPPO agent_{agent_id} model saved to {agent_save_path}")
    return saved_paths


def _count_completed_episodes(rollouts: list[RolloutData]) -> int:
    done_rows = [rollouts[0].next_done.unsqueeze(0)]
    if rollouts[0].dones.shape[0] > 1:
        done_rows.insert(0, rollouts[0].dones[1:])
    dones = torch.cat(done_rows, dim=0)
    #所有时间步至少有1个环境done的次数；
    #一个游戏进程中多个智能体会在同一个时间步 done，应视为一个回合
    #并行环境时所有环境也会在同一个时间步 done，既可以视为一个回合，也可以视为多个回合，这里视为一个回合
    return int(torch.any(dones > 0.5, dim=1).sum().item())


def _build_train_state(
    global_step: int,
    update: int,
    episode_count: int,
    optimizers: list[optim.Optimizer],
    episode_returns: list[deque],
) -> dict:
    state: dict = dict(
        global_step=int(global_step),
        update=int(update),
        episode_count=int(episode_count),
    )
    state["lrs"] = [float(optimizer.param_groups[0]["lr"]) for optimizer in optimizers]
    state["episode_returns"] = [list(returns) for returns in episode_returns]
    return state


def _load_train_state(ckpt: dict, n_agents: int) -> tuple[int, int, int, list[deque]]:
    global_step = int(ckpt.get("global_step", 0))
    update = int(ckpt.get("update", 0))
    episode_count = int(ckpt.get("episode_count", 0))
    raw_returns = ckpt.get("episode_returns", [[] for _ in range(n_agents)])
    episode_returns = []
    for i in range(n_agents):
        returns = raw_returns[i] if i < len(raw_returns) else []
        episode_returns.append(deque(returns, maxlen=20))
    return global_step, update, episode_count, episode_returns


def _load_agent_from_checkpoint(
    ckpt: dict,
    agent_id: int,
    agent: IPPOAgent,
    optimizer: optim.Optimizer,
    reward_normalizer: Optional[RewardNormalizer],
    is_resume: bool,
    should_restore_optimizer: bool,
) -> bool:
    if "agent_state_dict" in ckpt:
        ckpt_agent_id = ckpt.get("agent_id")
        if ckpt_agent_id is not None and int(ckpt_agent_id) != agent_id:
            return False
        agent.load_state_dict(ckpt["agent_state_dict"])
        if is_resume and should_restore_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        return True

    return False


def _make_checkpoint_path(base_path: Optional[str], episode_count: int) -> Optional[str]:
    if base_path is None:
        return None
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{base}_episode{episode_count}{ext}"


def _cleanup_old_checkpoints(base_path: Optional[str], max_keep: int) -> None:
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
                if "_agent" in num_str:
                    num_str = num_str.split("_agent", 1)[0]
                episode_num = int(num_str)
                checkpoints.append((episode_num, os.path.join(dir_name, f)))
            except ValueError:
                continue

    keep_episodes = sorted({episode for episode, _ in checkpoints})[-max_keep:]
    for episode, old_path in checkpoints:
        if episode in keep_episodes:
            continue
        try:
            os.remove(old_path)
        except OSError:
            pass


def _save_checkpoint(
    ckpt_path: str,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    args: IppoArgs,
    reward_normalizers: list[Optional[RewardNormalizer]],
    extra: dict,
) -> None:
    save_ippo_model(
        ckpt_path,
        agents,
        optimizers,
        reward_normalizers,
        args,
        extra=extra,
        train_only=True,
    )


def load_checkpoint_if_requested(
    resume_path: Optional[str],
    is_resume: bool,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    reward_normalizers: list[Optional[RewardNormalizer]],
    args: IppoArgs,
    device: torch.device,
) -> tuple[int, int, int, list[deque]]:
    n_agents = len(agents)
    if not resume_path:
        return 0, 1, 0, [deque(maxlen=20) for _ in range(n_agents)]

    first_ckpt = None
    loaded_any = False

    for i, agent in enumerate(agents):
        agent_id = args.agent_configs[i].agent_id
        agent_path = _make_agent_model_path(resume_path, agent_id)
        loaded = False
        if agent_path.is_file():
            print(f"[Resume] 加载 agent_{agent_id} checkpoint: {agent_path}")
            ckpt = load_full_checkpoint(str(agent_path), device)
            loaded = _load_agent_from_checkpoint(
                ckpt, agent_id, agent, optimizers[i], reward_normalizers[i],
                is_resume, args.agent_configs[i].train,
            )
            if not loaded:
                raise KeyError(f"checkpoint agent_id does not match agent_{agent_id}: {agent_path}")
            if first_ckpt is None:
                first_ckpt = ckpt
        elif args.agent_configs[i].train:
            raise FileNotFoundError(f"Missing agent_{agent_id} checkpoint: {agent_path}")

        loaded_any = loaded_any or loaded

    if not loaded_any:
        raise FileNotFoundError(
            f"No per-agent IPPO checkpoint files found for base path: {resume_path}"
        )

    if is_resume:
        start_global_step, start_update, start_episode_count, episode_returns = _load_train_state(first_ckpt, n_agents)
        start_update += 1
        print(
            f"[Resume] 从 update {start_update} / step {start_global_step} "
            f"/ episode {start_episode_count} 继续"
        )
    else:
        print("[Load] 仅加载模型参数，其余从头初始化")
        return 0, 1, 0, [deque(maxlen=20) for _ in range(n_agents)]

    return start_global_step, start_update, start_episode_count, episode_returns


def load_ppo_models_if_requested(
    ppo_model_paths: list[Optional[str]],
    agents: list[IPPOAgent],
    device: torch.device,
) -> None:
    if not ppo_model_paths:
        return

    for i, path in enumerate(ppo_model_paths):
        if path is None:
            continue
        if i >= len(agents):
            raise ValueError(f"ppo_model_paths has more entries than agents: index={i}, n_agents={len(agents)}.")

        print(f"[PPO Init] 加载 agent_{i} PPO model: {path}")
        ckpt = load_full_checkpoint(path, device)
        if "agent_state_dict" not in ckpt:
            raise KeyError(f"PPO checkpoint missing agent_state_dict: {path}")
        agents[i].load_state_dict(ckpt["agent_state_dict"])


#  Per-Agent 训练更新
def train_agent_update(
    agent: IPPOAgent,
    optimizer: optim.Optimizer,
    rollout: RolloutData,
    cfg: AgentConfig,
    args: IppoArgs,
    device: torch.device,
) -> dict:
    """对单个智能体执行一次 PPO 更新 (多个 epoch + minibatch)
    """
    num_steps = rollout.obs.shape[0]
    n_game_envs = rollout.obs.shape[1]
    rollout_batch_size = num_steps * n_game_envs

    #GAE
    with torch.no_grad():
        advantages, target_values = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.next_value,
            rollout.next_done,
            cfg.gamma,
            cfg.gae_lambda,
        )
    # Flatten
    b_obs = rollout.obs.reshape((-1,) + rollout.obs.shape[2:]) #shape(T*n_games,obs_dim)=(batch_size,obs_dim)
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

    if agent.is_recurrent:
        seq_len = max(1, min(int(args.recurrent_seq_len), num_steps))
        seq_starts = []
        seq_ends = []
        seq_envs = []
        for env_i in range(n_game_envs):
            for start_t in range(0, num_steps, seq_len):
                seq_starts.append(start_t)
                seq_ends.append(min(start_t + seq_len, num_steps))
                seq_envs.append(env_i)

        all_seq_starts = np.asarray(seq_starts)
        all_seq_ends = np.asarray(seq_ends)
        all_seq_envs = np.asarray(seq_envs)
        seq_inds = np.arange(len(all_seq_starts))
        seqs_per_minibatch = max(
            1, (len(seq_inds) + args.num_minibatches - 1) // args.num_minibatches
        )

        for epoch in range(args.update_epochs):
            np.random.shuffle(seq_inds)

            for start in range(0, len(seq_inds), seqs_per_minibatch):
                mb_seq_inds = seq_inds[start:start + seqs_per_minibatch]
                mb_inds, new_logprob, entropy, new_value = evaluate_recurrent_sequences(
                    agent,
                    rollout,
                    all_seq_starts[mb_seq_inds],
                    all_seq_ends[mb_seq_inds],
                    all_seq_envs[mb_seq_inds],
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
                if approx_kl > args.target_kl:
                    break
    else:
        b_inds = np.arange(rollout_batch_size)#batch indices
        minibatch_size = min(args.minibatch_size, rollout_batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, rollout_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value( #shape(minibatch_size,1)
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

            # KL 早停
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
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
    start_global_step: int = 0,
    start_update: int = 1,
    start_episode_count: int = 0,
    start_episode_returns: Optional[list[deque]] = None,
) -> dict:
    """IPPO 主训练循环"""
    n_agents = len(args.agent_configs)
    n_game_envs = args.num_game_envs# 每个智能体的并行环境数
    global_step = start_global_step
    start_time = time.time()

    # 按 count_steps_by 计算总更新次数
    if args.count_steps_by == "env_steps":
        num_updates = args.total_timesteps // (n_game_envs * args.num_steps)
        step_increment = n_game_envs
    elif args.count_steps_by == "agent_steps":
        num_updates = args.total_timesteps // (n_game_envs*n_agents * args.num_steps)
        step_increment = args.num_envs
    else:
        raise ValueError(
            f"Unknown count_steps_by='{args.count_steps_by}'. "
            "Expected 'env_steps' or 'agent_steps'."
        )

    episode_count = start_episode_count

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)] #每个智能体最近20个回合的奖励
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)#每个智能体每回合累计奖励
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents] #每个智能体的初始 RNN 状态

    if start_episode_returns is not None:
        episode_returns = start_episode_returns

    next_checkpoint_episode = None
    if args.save_checkpoint and args.save_every_n_episodes > 0:
        interval = int(args.save_every_n_episodes)
        next_checkpoint_episode = ((episode_count // interval) + 1) * interval
    train.last_train_state = _build_train_state(
        global_step, start_update - 1, episode_count,
        optimizers, episode_returns,
    )

    for update in range(start_update, num_updates + 1):
        # 学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            for i in range(n_agents):
                if args.agent_configs[i].train:
                    optimizers[i].param_groups[0]["lr"] = progress * args.agent_configs[i].learning_rate

        # 经验采集
        rollouts, global_step, rnn_states, new_episode_returns = collect_parallel_rollout_ippo(
            args.agent_configs, agents, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards, reward_normalizers,
            rnn_states, step_increment,
        )

        next_obs = torch.stack([r.next_obs for r in rollouts], dim=1).reshape(args.num_envs, -1)
        next_done = torch.stack([r.next_done for r in rollouts], dim=1).reshape(args.num_envs)
        episode_count += _count_completed_episodes(rollouts)

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
            new_episode_returns=new_episode_returns,
        )

        train.last_train_state = _build_train_state(
            global_step, update, episode_count,
            optimizers, episode_returns,
        )

        if args.save_checkpoint and next_checkpoint_episode is not None and episode_count >= next_checkpoint_episode:
            ckpt_path = _make_checkpoint_path(args.save_model_path, episode_count)
            if ckpt_path:
                _save_checkpoint(
                    ckpt_path, agents, optimizers,
                    args, reward_normalizers, train.last_train_state,
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
        optimizers, episode_returns,
    )


#  主训练入口
def main():
    # 初始化
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
    args.num_game_envs = args.num_envs // n_agents# 每个智能体的并行环境数
    if args.num_game_envs != args.n_parallel:
        raise ValueError(
            "Godot env slot count does not match the configured parallel game count: "
            f"envs.num_envs={args.num_envs}, n_agents={n_agents}, "
            f"derived n_parallel={args.num_game_envs}, args.n_parallel={args.n_parallel}."
        )

    obs_shape = envs.single_observation_space.shape
    if len(obs_shape) != 1 or obs_shape[0] != seg.total:
        raise ValueError(f"Observation dimension mismatch: env observation shape={obs_shape}, configured segment total={seg.total}.")

    args.batch_size = args.num_game_envs * args.num_steps #每个智能体的样本数
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size <= 0:
        raise ValueError(
            "num_minibatches is too large for the configured IPPO batch: "
            f"batch_size={args.batch_size}, num_minibatches={args.num_minibatches}."
        )

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

    load_ppo_models_if_requested(args.ppo_model_paths, agents, device)

    #note：中断点加载优先级更高.此步会覆盖上一步的ppo模型
    resume_path = args.resume_from or args.load_model_path
    is_resume = bool(args.resume_from)
    start_global_step, start_update, start_episode_count, episode_returns = (
        load_checkpoint_if_requested(
            resume_path, is_resume,
            agents, optimizers, reward_normalizers,
            args, device,
        )
    )

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed) #shape (n_parallel*n_agents, obs_dim)
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device)
    if next_obs.shape[0] != args.num_envs:
        raise ValueError(
            "Reset observation count does not match Godot env slots: "
            f"next_obs.shape[0]={next_obs.shape[0]}, envs.num_envs={args.num_envs}."
        )
    next_done = torch.zeros(args.num_envs).to(device)

    final_state = None
    final_state = train(
        args, agents, optimizers, envs, device, writer,
        reward_normalizers, next_obs, next_done,
        start_global_step, start_update, start_episode_count,
        episode_returns,
    )

    # 正常训练结束后的保存
    if args.save_model_path is not None and final_state is not None:
        save_ippo_model(
            args.save_model_path,
            agents, optimizers, reward_normalizers,
            args, extra=final_state, train_only=True,
        )

    # 资源清理
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
