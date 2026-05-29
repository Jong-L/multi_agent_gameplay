"""
Continue one IPPO agent against a sampled opponent pool.

This is a small final-phase entry point for experiments where the prior
pool-cycle rounds are treated as population initialization and the target
agent needs an additional standalone opponent-pool training budget.

Default behavior:
  - main agent: agent_0
  - main checkpoint: saved_models/ippo_pool_final_agent0.pt
  - opponent pool: recent 20 checkpoints per agent from
    saved_models/ippo_pool_checkpoints
  - training budget: 5,000,000 env steps
  - load mode: resume/checkpoint (agent weights + optimizer + reward normalizer + counters)

Use --load-mode weights to initialize from model parameters only.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from custom_ppo_dataclass import PoolEntry
from custom_ippo import _build_train_state
from custom_ippo_pool import (
    IppoPoolArgs,
    OpponentPoolState,
    _agent_ids,
    _capture_agent_states,
    _with_train_flags,
    close_context,
    save_selected_agents,
    setup_training_context,
    train_pool_phase,
)
from godot_env_wrapper import load_full_checkpoint


_ROUND_RE = re.compile(r"round(\d+)_agent(\d+)")
_FINAL_RE = re.compile(r"final_agent(\d+)")
_STEP_RE = re.compile(r"_step(\d+)")
_EPISODE_RE = re.compile(r"_episode(\d+)")
_AGENT_RE_TEMPLATE = r"_agent{agent_id}(?:\.pt|$)"


@dataclass
class ContinuePoolAgentArgs(IppoPoolArgs):
    """Config for final-only opponent-pool continuation."""

    main_agent_id: int = 0
    main_checkpoint_path: str = "saved_models/ippo_pool_final_agent0.pt"
    load_mode: str = "resume"
    opponent_checkpoint_dir: str = "saved_models/ippo_pool_checkpoints"
    opponent_keep_per_agent: int = 20
    phase_name: str = "agent0_extra_pool"
    save_model_path: Optional[str] = "saved_models/ippo_pool_agent0_extra_final"
    pool_final_timesteps: int = 5_000_000
    run_name: Optional[str] = "ippo_pool_agent0_extra"
    ppo_model_paths: list[Optional[str]] = None

    def __post_init__(self) -> None:
        if self.ppo_model_paths is None:
            self.ppo_model_paths = [None for _ in self.agent_configs]


def _parse_args() -> ContinuePoolAgentArgs:
    args = ContinuePoolAgentArgs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-agent-id", type=int, default=args.main_agent_id)
    parser.add_argument("--main-checkpoint-path", default=args.main_checkpoint_path)
    parser.add_argument("--load-mode", choices=["resume", "checkpoint", "weights"], default=args.load_mode)
    parser.add_argument("--opponent-checkpoint-dir", default=args.opponent_checkpoint_dir)
    parser.add_argument("--opponent-keep-per-agent", type=int, default=args.opponent_keep_per_agent)
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
    parser.add_argument("--pool-epsilon", type=float, default=args.pool_epsilon)
    parser.add_argument("--pool-pfsp-temperature", type=float, default=args.pool_pfsp_temperature)
    parser.add_argument("--pool-reward-ema", type=float, default=args.pool_reward_ema)
    parser.add_argument("--pool-default-reward-score", type=float, default=args.pool_default_reward_score)
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--track", action="store_true")
    parsed = parser.parse_args()

    args.main_agent_id = int(parsed.main_agent_id)
    args.main_checkpoint_path = parsed.main_checkpoint_path
    args.load_mode = parsed.load_mode
    args.opponent_checkpoint_dir = parsed.opponent_checkpoint_dir
    args.opponent_keep_per_agent = int(parsed.opponent_keep_per_agent)
    args.pool_initial_keep_per_agent = int(parsed.opponent_keep_per_agent)
    args.pool_slots_per_agent = max(int(args.pool_slots_per_agent), int(parsed.opponent_keep_per_agent))
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
    args.pool_epsilon = float(parsed.pool_epsilon)
    args.pool_pfsp_temperature = float(parsed.pool_pfsp_temperature)
    args.pool_reward_ema = float(parsed.pool_reward_ema)
    args.pool_default_reward_score = float(parsed.pool_default_reward_score)
    args.anneal_lr = not parsed.no_anneal_lr
    args.track = bool(parsed.track)
    args.use_opponent_pool = True
    args.pool_final_agent_id = args.main_agent_id
    args.pool_final_save_agent_ids = (args.main_agent_id,)
    args.ppo_model_paths = [None for _ in args.agent_configs]
    args.save_checkpoint = False
    return args


def _checkpoint_sort_key(path: pathlib.Path, agent_id: int) -> tuple[int, int, int, float, str]:
    """Sort pool checkpoint files by parsed round first, then step/episode."""
    name = path.name
    final_match = _FINAL_RE.search(name)
    round_match = _ROUND_RE.search(name)
    if final_match and int(final_match.group(1)) == int(agent_id):
        phase_rank = 1_000_000
        round_number = 1_000_000
    elif round_match and int(round_match.group(2)) == int(agent_id):
        phase_rank = 0
        round_number = int(round_match.group(1))
    else:
        phase_rank = -1
        round_number = -1

    step_match = _STEP_RE.search(name)
    episode_match = _EPISODE_RE.search(name)
    progress_number = int(step_match.group(1)) if step_match else (
        int(episode_match.group(1)) if episode_match else 0
    )
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return phase_rank, round_number, progress_number, mtime, name


def _checkpoint_source(path: pathlib.Path, agent_id: int) -> str:
    name = path.name
    final_match = _FINAL_RE.search(name)
    if final_match and int(final_match.group(1)) == int(agent_id):
        return "recent_final"
    round_match = _ROUND_RE.search(name)
    if round_match and int(round_match.group(2)) == int(agent_id):
        return f"recent_round{round_match.group(1)}"
    return "recent"


def find_recent_pool_checkpoints(
    checkpoint_dir: str,
    agent_id: int,
    keep_latest: int,
) -> list[pathlib.Path]:
    root = pathlib.Path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"Opponent checkpoint directory does not exist: {checkpoint_dir}")

    agent_re = re.compile(_AGENT_RE_TEMPLATE.format(agent_id=agent_id))
    candidates = [
        path for path in root.rglob("*.pt")
        if agent_re.search(path.name)
    ]
    candidates.sort(key=lambda p: _checkpoint_sort_key(p, agent_id))
    if keep_latest > 0:
        candidates = candidates[-keep_latest:]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for agent_{agent_id} under {checkpoint_dir}"
        )
    return candidates


def build_recent_opponent_pool(args: ContinuePoolAgentArgs) -> OpponentPoolState:
    pool = OpponentPoolState(
        agent_ids=_agent_ids(args),
        per_agent_max_size=args.pool_slots_per_agent,
        epsilon=args.pool_epsilon,
        temperature=args.pool_pfsp_temperature,
        reward_ema_coef=args.pool_reward_ema,
        default_reward_score=args.pool_default_reward_score,
        delete_replaced_checkpoints=args.pool_delete_replaced_checkpoints,
    )

    for agent_id in _agent_ids(args):
        paths = find_recent_pool_checkpoints(
            args.opponent_checkpoint_dir,
            agent_id,
            args.opponent_keep_per_agent,
        )
        for path in paths:
            _, _, progress_number, _, _ = _checkpoint_sort_key(path, agent_id)
            pool.add_entry(
                PoolEntry(
                    checkpoint_path=str(path),
                    agent_id=agent_id,
                    global_step=progress_number,
                    source=_checkpoint_source(path, agent_id),
                )
            )
        first = pathlib.Path(paths[0]).name
        last = pathlib.Path(paths[-1]).name
        print(
            f"[Pool Init] agent_{agent_id}: loaded {len(paths)} recent checkpoints "
            f"from {first} -> {last}"
        )

    return pool


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _restore_episode_returns(raw_returns: Any, n_agents: int) -> list[deque]:
    restored = [deque(maxlen=20) for _ in range(n_agents)]
    if raw_returns is None:
        return restored
    if raw_returns and all(isinstance(item, (int, float)) for item in raw_returns):
        restored[0] = deque(raw_returns, maxlen=20)
        return restored
    for idx in range(min(n_agents, len(raw_returns))):
        restored[idx] = deque(raw_returns[idx], maxlen=20)
    return restored


def load_main_agent_checkpoint(
    args: ContinuePoolAgentArgs,
    ctx: Any,
) -> None:
    ckpt = load_full_checkpoint(args.main_checkpoint_path, ctx.device)
    ckpt_agent_id = ckpt.get("agent_id")
    if ckpt_agent_id is not None and int(ckpt_agent_id) != int(args.main_agent_id):
        raise ValueError(
            f"Main checkpoint agent_id mismatch: expected {args.main_agent_id}, "
            f"got {ckpt_agent_id}, path={args.main_checkpoint_path}"
        )
    if "agent_state_dict" not in ckpt:
        raise KeyError(f"Main checkpoint missing agent_state_dict: {args.main_checkpoint_path}")

    main_id = int(args.main_agent_id)
    ctx.agents[main_id].load_state_dict(ckpt["agent_state_dict"])
    print(f"[Main Load] agent_{main_id} weights <- {args.main_checkpoint_path}")

    if args.load_mode == "weights":
        print("[Main Load] load_mode=weights: optimizer, reward normalizer, counters start fresh.")
        ctx.global_step = 0
        ctx.start_update = 1
        ctx.episode_count = 0
        ctx.episode_returns = [deque(maxlen=20) for _ in ctx.agents]
        return

    if "optimizer_state_dict" in ckpt:
        ctx.optimizers[main_id].load_state_dict(ckpt["optimizer_state_dict"])
        _move_optimizer_state_to_device(ctx.optimizers[main_id], ctx.device)
        print(f"[Main Load] agent_{main_id} optimizer restored")
    else:
        print(f"[Main Load] warning: optimizer_state_dict not found in {args.main_checkpoint_path}")

    if "reward_normalizer" in ckpt:
        if ctx.reward_normalizers[main_id] is None:
            print("[Main Load] warning: checkpoint has reward_normalizer but current config disabled it")
        else:
            ctx.reward_normalizers[main_id].load_state_dict(ckpt["reward_normalizer"])
            print(f"[Main Load] agent_{main_id} reward_normalizer restored")
    else:
        print(f"[Main Load] warning: reward_normalizer not found in {args.main_checkpoint_path}")

    ctx.global_step = int(ckpt.get("global_step", 0))
    ctx.start_update = int(ckpt.get("update", 0)) + 1
    ctx.episode_count = int(ckpt.get("episode_count", 0))
    ctx.episode_returns = _restore_episode_returns(
        ckpt.get("episode_returns"),
        len(ctx.agents),
    )
    print(
        f"[Main Load] counters restored: step={ctx.global_step}, "
        f"update={ctx.start_update}, episode={ctx.episode_count}"
    )


def run_continue_pool_agent(args: ContinuePoolAgentArgs) -> None:
    if args.main_agent_id not in _agent_ids(args):
        raise ValueError(f"main_agent_id={args.main_agent_id} is not in agent ids: {_agent_ids(args)}")

    pool = build_recent_opponent_pool(args)
    setup_args = copy.deepcopy(args)
    setup_args.agent_configs = _with_train_flags(
        setup_args,
        train_ids={int(args.main_agent_id)},
        policy_opponent_ids=set(_agent_ids(args)) - {int(args.main_agent_id)},
    )
    setup_args.ppo_model_paths = [None for _ in setup_args.agent_configs]
    setup_args.load_mode = "weights"

    ctx = None
    try:
        ctx = setup_training_context(setup_args)
        load_main_agent_checkpoint(args, ctx)
        current_agent_states = _capture_agent_states(ctx.agents)
        rng = random.Random(args.seed)
        print(
            f"[Final Only] train agent_{args.main_agent_id} for "
            f"{args.pool_final_timesteps} {args.count_steps_by}"
        )
        train_pool_phase(
            ctx,
            pool,
            current_agent_states,
            int(args.main_agent_id),
            int(args.pool_final_timesteps),
            args.phase_name,
            rng,
        )

        for agent_id, state in enumerate(current_agent_states):
            ctx.agents[agent_id].load_state_dict(
                {key: value.to(ctx.device) for key, value in state.items()}
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
        save_selected_agents(
            ctx.args.save_model_path,
            {int(args.main_agent_id)},
            ctx.agents,
            ctx.optimizers,
            ctx.reward_normalizers,
            ctx.args,
            extra={**final_state, "pool_rows": pool.to_rows()},
        )
    finally:
        close_context(ctx)


def main() -> None:
    run_continue_pool_agent(_parse_args())


if __name__ == "__main__":
    main()
