"""
Train one IPPO agent against a behavioral-average opponent population.

This is intended as the third comparison experiment after:
  1. direct IPPO training
  2. opponent-pool/PFSP training

Default behavior:
  - main agent: agent_0
  - main checkpoint: saved_models/ippo_bootstrap_agent0.pt
  - opponents: latest 20 complete checkpoint slots from
    saved_models/ippo_pool_checkpoints
  - opponent action rule: every frozen checkpoint policy for that opponent
    computes action probabilities on the current observation; probabilities are
    averaged; the argmax action is executed.

This is a behavioral average, not a neural-network parameter average. The
opponent checkpoints are assumed to be MLP policies. Recurrent opponents are
rejected deliberately: averaging recurrent policies would require one hidden
state per checkpoint and consistent resets for every member of the ensemble.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from custom_ippo import (
    IPPOAgent,
    _build_train_state,
    _count_completed_episodes,
    log_ippo,
    train_agent_update,
)
from custom_ppo_dataclass import RolloutData
from custom_ippo_pool import (
    IppoPoolArgs,
    _agent_ids,
    _load_agent_state,
    _phase_update_count,
    _pool_checkpoint_dir,
    _with_train_flags,
    close_context,
    save_selected_agents,
    setup_training_context,
)
from continue_ippo_pool_agent import (
    find_recent_pool_checkpoints,
    load_main_agent_checkpoint,
)


@dataclass
class ContinueAverageOpponentAgentArgs(IppoPoolArgs):
    """Config for final-only training against average opponent policies."""

    main_agent_id: int = 0
    main_checkpoint_path: str = "saved_models/ippo_bootstrap_agent0.pt"
    load_mode: str = "weights"
    opponent_checkpoint_dir: str = "saved_models/ippo_pool_checkpoints"
    opponent_keep_per_agent: int = 20
    strict_opponent_count: bool = True
    phase_name: str = "agent0_vs_average_opponents"
    save_model_path: Optional[str] = "saved_models/ippo_average_opponent_agent0_final"
    pool_checkpoint_dir: Optional[str] = "saved_models/ippo_average_opponent_checkpoints"
    pool_final_timesteps: int = 1_000_0000
    run_name: Optional[str] = "ippo_average_opponent_agent0"
    ppo_model_paths: list[Optional[str]] = None

    def __post_init__(self) -> None:
        if self.ppo_model_paths is None:
            self.ppo_model_paths = [None for _ in self.agent_configs]


@dataclass
class AverageOpponentEnsemble:
    """Frozen policies for one non-main agent slot."""

    agent_id: int
    checkpoint_paths: list[pathlib.Path]
    policies: list[IPPOAgent]

    @property
    def size(self) -> int:
        return len(self.policies)

    def act_argmax_average_prob(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.policies:
            raise RuntimeError(f"Average opponent agent_{self.agent_id} has no policies.")

        prob_sum: Optional[torch.Tensor] = None
        for policy in self.policies:
            features, _ = policy._forward_features(obs)
            logits = policy.actor(features)
            probs = torch.softmax(logits, dim=-1)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        avg_probs = prob_sum / float(len(self.policies))
        return torch.argmax(avg_probs, dim=-1)


def _parse_args() -> ContinueAverageOpponentAgentArgs:
    args = ContinueAverageOpponentAgentArgs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-agent-id", type=int, default=args.main_agent_id)
    parser.add_argument("--main-checkpoint-path", default=args.main_checkpoint_path)
    parser.add_argument("--load-mode", choices=["resume", "weights"], default=args.load_mode)
    parser.add_argument("--opponent-checkpoint-dir", default=args.opponent_checkpoint_dir)
    parser.add_argument("--opponent-keep-per-agent", type=int, default=args.opponent_keep_per_agent)
    parser.add_argument("--allow-fewer-opponents", action="store_true")
    parser.add_argument("--timesteps", type=int, default=args.pool_final_timesteps)
    parser.add_argument("--save-model-path", default=args.save_model_path)
    parser.add_argument("--pool-checkpoint-dir", default=args.pool_checkpoint_dir)
    parser.add_argument("--pool-save-interval", type=int, default=args.pool_save_interval)
    parser.add_argument("--phase-name", default=args.phase_name)
    parser.add_argument("--run-name", default=args.run_name)
    parser.add_argument("--env-path", default=args.env_path)
    parser.add_argument("--config-path", default=args.config_path)
    parser.add_argument("--n-parallel", type=int, default=args.n_parallel)
    parser.add_argument("--seed", type=int, default=args.seed)
    parser.add_argument("--num-steps", type=int, default=args.num_steps)
    parser.add_argument("--num-minibatches", type=int, default=args.num_minibatches)
    parser.add_argument("--update-epochs", type=int, default=args.update_epochs)
    parser.add_argument("--count-steps-by", choices=["env_steps", "agent_steps"], default=args.count_steps_by)
    parser.add_argument("--port-offset", type=int, default=args.port_offset)
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--track", action="store_true")
    parsed = parser.parse_args()

    args.main_agent_id = int(parsed.main_agent_id)
    args.main_checkpoint_path = parsed.main_checkpoint_path
    args.load_mode = parsed.load_mode
    args.opponent_checkpoint_dir = parsed.opponent_checkpoint_dir
    args.opponent_keep_per_agent = int(parsed.opponent_keep_per_agent)
    args.strict_opponent_count = not bool(parsed.allow_fewer_opponents)
    args.pool_final_timesteps = int(parsed.timesteps)
    args.save_model_path = parsed.save_model_path
    args.pool_checkpoint_dir = parsed.pool_checkpoint_dir
    args.pool_save_interval = int(parsed.pool_save_interval)
    args.phase_name = parsed.phase_name
    args.run_name = parsed.run_name
    args.env_path = parsed.env_path
    args.config_path = parsed.config_path
    args.n_parallel = int(parsed.n_parallel)
    args.seed = int(parsed.seed)
    args.num_steps = int(parsed.num_steps)
    args.num_minibatches = int(parsed.num_minibatches)
    args.update_epochs = int(parsed.update_epochs)
    args.count_steps_by = parsed.count_steps_by
    args.port_offset = int(parsed.port_offset)
    args.anneal_lr = not parsed.no_anneal_lr
    args.track = bool(parsed.track)
    args.use_opponent_pool = False
    args.pool_final_agent_id = args.main_agent_id
    args.pool_final_save_agent_ids = (args.main_agent_id,)
    args.ppo_model_paths = [None for _ in args.agent_configs]
    args.resume_from = None
    args.load_model_path = None
    args.save_checkpoint = False
    return args


def _freeze_policy(policy: IPPOAgent) -> IPPOAgent:
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy


def _make_policy_like(ctx: Any, agent_id: int) -> IPPOAgent:
    n_actions = int(ctx.envs.single_action_space.n)
    cfg = ctx.args.agent_configs[agent_id]
    policy = IPPOAgent(n_actions, ctx.agents[agent_id].seg, cfg).to(ctx.device)
    return policy


def _build_average_opponent_ensembles(
    args: ContinueAverageOpponentAgentArgs,
    ctx: Any,
) -> dict[int, AverageOpponentEnsemble]:
    opponent_ids = [agent_id for agent_id in _agent_ids(args) if agent_id != int(args.main_agent_id)]
    paths_by_agent: dict[int, list[pathlib.Path]] = {}

    for agent_id in opponent_ids:
        paths = find_recent_pool_checkpoints(
            args.opponent_checkpoint_dir,
            agent_id,
            int(args.opponent_keep_per_agent),
        )
        if args.strict_opponent_count and len(paths) != int(args.opponent_keep_per_agent):
            raise RuntimeError(
                f"Expected {args.opponent_keep_per_agent} checkpoints for agent_{agent_id}, "
                f"found {len(paths)} under {args.opponent_checkpoint_dir}. "
                "Pass --allow-fewer-opponents to train with the available complete slots."
            )
        paths_by_agent[agent_id] = paths

    complete_slot_count = min(len(paths) for paths in paths_by_agent.values())
    if complete_slot_count <= 0:
        raise RuntimeError("No complete average-opponent checkpoint slots were found.")

    ensembles: dict[int, AverageOpponentEnsemble] = {}
    for agent_id in opponent_ids:
        paths = paths_by_agent[agent_id][-complete_slot_count:]
        policies: list[IPPOAgent] = []
        for path in paths:
            policy = _make_policy_like(ctx, agent_id)
            _load_agent_state(str(path), agent_id, policy, ctx.device)
            if policy.is_recurrent:
                raise ValueError(
                    "Average-opponent training only supports MLP/segmented-MLP opponents. "
                    f"Recurrent checkpoint rejected: {path}"
                )
            policies.append(_freeze_policy(policy))
        ensembles[agent_id] = AverageOpponentEnsemble(
            agent_id=agent_id,
            checkpoint_paths=paths,
            policies=policies,
        )
        print(
            f"[Average Opponent] agent_{agent_id}: loaded {len(paths)} policies "
            f"from {paths[0].name} -> {paths[-1].name}"
        )

    return ensembles


def _average_opponent_rows(
    ensembles: dict[int, AverageOpponentEnsemble],
) -> list[dict[str, Any]]:
    rows = []
    for agent_id, ensemble in sorted(ensembles.items()):
        for slot_index, path in enumerate(ensemble.checkpoint_paths):
            rows.append({
                "agent_id": int(agent_id),
                "slot_index": int(slot_index),
                "checkpoint_path": str(path),
            })
    return rows


def collect_parallel_rollout_average_opponents(
    agents_cfg,
    agents: list[IPPOAgent],
    average_opponents: dict[int, AverageOpponentEnsemble],
    envs,
    rollout_steps: int,
    device: torch.device,
    next_obs_all: torch.Tensor,
    next_done_all: torch.Tensor,
    global_step: int,
    episode_returns: list[deque],
    accum_rewards: np.ndarray,
    reward_normalizers,
    rnn_states: list[Optional[torch.Tensor]],
    step_increment: int,
) -> tuple[list[RolloutData], int, list[Optional[torch.Tensor]], list[list[float]]]:
    """Collect rollout data while non-main agents use averaged frozen policies."""
    n_agents = len(agents_cfg)
    total_slots = envs.num_envs
    n_game_envs = total_slots // n_agents
    obs_shape = envs.single_observation_space.shape
    obs_dim = obs_shape[0]

    buffers: list[dict[str, Any]] = []
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
            obs_i = next_obs_by_env[:, i, :]
            done_i = next_done_by_env[:, i]
            buffers[i]["obs"][step] = obs_i
            buffers[i]["dones"][step] = done_i

            if agents_cfg[i].train:
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
            elif i in average_opponents:
                with torch.no_grad():
                    action = average_opponents[i].act_argmax_average_prob(obs_i)
                buffers[i]["actions"][step] = action
                actions_by_env[:, i] = action.cpu().numpy().astype(np.int64)
            else:
                random_actions = np.random.randint(
                    0,
                    envs.single_action_space.n,
                    size=n_game_envs,
                    dtype=np.int64,
                )
                buffers[i]["actions"][step] = torch.tensor(random_actions, device=device)
                actions_by_env[:, i] = random_actions

        next_obs_raw, rewards_raw, terminations, truncations, _ = envs.step(
            actions_by_env.reshape(-1)
        )
        dones_raw = np.logical_or(terminations, truncations)
        rewards_by_env = np.asarray(rewards_raw, dtype=np.float32).reshape(
            n_game_envs,
            n_agents,
        )
        dones_by_env = np.asarray(dones_raw, dtype=bool).reshape(n_game_envs, n_agents)

        next_obs_all = torch.tensor(np.asarray(next_obs_raw, dtype=np.float32), device=device)
        next_done_all = torch.tensor(dones_raw, dtype=torch.float32, device=device)

        for i in range(n_agents):
            reward_i = rewards_by_env[:, i]
            if agents_cfg[i].train and reward_normalizers[i] is not None:
                reward_i_norm = reward_normalizers[i].normalize_array(reward_i)
                reward_normalizers[i].update_array(reward_i)
            else:
                reward_i_norm = reward_i

            buffers[i]["rewards"][step] = torch.tensor(
                reward_i_norm,
                dtype=torch.float32,
                device=device,
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


def _save_main_snapshot(
    ctx: Any,
    main_agent_id: int,
    phase_name: str,
    average_opponents: dict[int, AverageOpponentEnsemble],
) -> None:
    checkpoint_dir = _pool_checkpoint_dir(ctx.args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_path = checkpoint_dir / f"{phase_name}_step{ctx.global_step}"
    train_state = _build_train_state(
        ctx.global_step,
        ctx.start_update,
        ctx.episode_count,
        ctx.optimizers,
        ctx.episode_returns,
    )
    save_selected_agents(
        str(base_path),
        {main_agent_id},
        ctx.agents,
        ctx.optimizers,
        ctx.reward_normalizers,
        ctx.args,
        extra={
            **train_state,
            "opponent_mode": "average_action_probability_argmax",
            "average_opponent_rows": _average_opponent_rows(average_opponents),
        },
    )


def train_average_opponent_phase(
    ctx: Any,
    average_opponents: dict[int, AverageOpponentEnsemble],
    main_agent_id: int,
    phase_timesteps: int,
    phase_name: str,
) -> None:
    args = ctx.args
    n_agents = len(args.agent_configs)
    n_game_envs = args.num_game_envs

    args.agent_configs = _with_train_flags(args, train_ids={main_agent_id})
    phase_updates, step_increment = _phase_update_count(args, phase_timesteps)
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, ctx.device) for agent in ctx.agents]
    start_time = time.time()
    last_snapshot_step = ctx.global_step

    for local_update in range(1, phase_updates + 1):
        if args.anneal_lr:
            progress = 1.0 - (local_update - 1.0) / phase_updates
            cfg = args.agent_configs[main_agent_id]
            ctx.optimizers[main_agent_id].param_groups[0]["lr"] = progress * cfg.learning_rate

        rollouts, ctx.global_step, rnn_states, new_episode_returns = (
            collect_parallel_rollout_average_opponents(
                args.agent_configs,
                ctx.agents,
                average_opponents,
                ctx.envs,
                args.num_steps,
                ctx.device,
                ctx.next_obs,
                ctx.next_done,
                ctx.global_step,
                ctx.episode_returns,
                accum_rewards,
                ctx.reward_normalizers,
                rnn_states,
                step_increment,
            )
        )
        ctx.next_obs = torch.stack([r.next_obs for r in rollouts], dim=1).reshape(args.num_envs, -1)
        ctx.next_done = torch.stack([r.next_done for r in rollouts], dim=1).reshape(args.num_envs)
        ctx.episode_count += _count_completed_episodes(rollouts)

        losses: list[Optional[dict]] = [None for _ in range(n_agents)]
        explained_vars = [0.0 for _ in range(n_agents)]
        metrics = train_agent_update(
            ctx.agents[main_agent_id],
            ctx.optimizers[main_agent_id],
            rollouts[main_agent_id],
            args.agent_configs[main_agent_id],
            args,
            ctx.device,
        )
        losses[main_agent_id] = metrics
        explained_vars[main_agent_id] = metrics.get("explained_var", 0.0)

        log_ippo(
            ctx.writer,
            ctx.global_step,
            args.agent_configs,
            ctx.optimizers,
            losses,
            explained_vars,
            ctx.episode_returns,
            start_time,
            update=local_update,
            num_updates=phase_updates,
            new_episode_returns=new_episode_returns,
        )
        ctx.start_update += 1

        if ctx.global_step - last_snapshot_step >= args.pool_save_interval:
            _save_main_snapshot(ctx, main_agent_id, phase_name, average_opponents)
            last_snapshot_step = ctx.global_step

    _save_main_snapshot(ctx, main_agent_id, f"{phase_name}_end", average_opponents)


def run_continue_average_opponent_agent(args: ContinueAverageOpponentAgentArgs) -> None:
    if args.main_agent_id not in _agent_ids(args):
        raise ValueError(f"main_agent_id={args.main_agent_id} is not in agent ids: {_agent_ids(args)}")

    setup_args = copy.deepcopy(args)
    setup_args.agent_configs = _with_train_flags(
        setup_args,
        train_ids={int(args.main_agent_id)},
    )
    setup_args.ppo_model_paths = [None for _ in setup_args.agent_configs]
    setup_args.resume_from = None
    setup_args.load_model_path = None

    ctx = None
    try:
        ctx = setup_training_context(setup_args)
        load_main_agent_checkpoint(args, ctx)
        average_opponents = _build_average_opponent_ensembles(args, ctx)
        print(
            f"[Average Opponent] train agent_{args.main_agent_id} for "
            f"{args.pool_final_timesteps} {args.count_steps_by}; "
            f"opponent ensemble size={next(iter(average_opponents.values())).size}"
        )
        train_average_opponent_phase(
            ctx,
            average_opponents,
            int(args.main_agent_id),
            int(args.pool_final_timesteps),
            args.phase_name,
        )

        final_state = _build_train_state(
            ctx.global_step,
            ctx.start_update,
            ctx.episode_count,
            ctx.optimizers,
            ctx.episode_returns,
        )
        final_state["continued_from"] = str(pathlib.Path(args.main_checkpoint_path))
        final_state["load_mode"] = args.load_mode
        final_state["extra_phase_timesteps"] = int(args.pool_final_timesteps)
        final_state["opponent_mode"] = "average_action_probability_argmax"
        final_state["average_opponent_rows"] = _average_opponent_rows(average_opponents)
        save_selected_agents(
            ctx.args.save_model_path,
            {int(args.main_agent_id)},
            ctx.agents,
            ctx.optimizers,
            ctx.reward_normalizers,
            ctx.args,
            extra=final_state,
        )
    finally:
        close_context(ctx)


def main() -> None:
    run_continue_average_opponent_agent(_parse_args())


if __name__ == "__main__":
    main()
