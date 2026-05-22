"""
Opponent-pool IPPO training for Tiny Swords.

This script intentionally reuses the existing custom_ippo.py PPO/IPPO pieces:
rollout collection, PPO updates, logging, checkpoint save/load, and Godot setup.

Supported modes:
  - ippo_bootstrap: load four single-agent PPO checkpoints, train all four
    agents together, save intermediate IPPO checkpoints, then continue direct
    IPPO training and save selected final agents.
  - pool_cycle: build opponent_pool[agent_id][slot_index] from recent IPPO
    checkpoints, train one main agent per phase, and sample frozen opponents
    with higher probability for groups that produced lower main-agent returns.
  - evaluate: compare two agent0 checkpoints against the same sampled opponent
    groups and write a CSV summary.

The defaults mirror the experiment idea in the prompt, but all important
numbers are dataclass fields or CLI overrides.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import pathlib
import random
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim

from custom_ppo_dataclass import AgentConfig, IppoArgs, PoolEntry
from custom_ippo import (
    IPPOAgent,
    _build_train_state,
    _count_completed_episodes,
    collect_parallel_rollout_ippo,
    load_checkpoint_if_requested,
    load_ppo_models_if_requested,
    log_ippo,
    save_ippo_model,
    train as train_ippo,
    train_agent_update,
)
from godot_env_wrapper import RewardNormalizer, init_training_setup, load_full_checkpoint


@dataclass
class IppoPoolArgs(IppoArgs):
    """Configuration for the staged IPPO/opponent-pool experiment."""

    run_mode: str = "pool_cycle"
    """ippo_bootstrap / pool_cycle / evaluate / full"""

    bootstrap_save_model_path: Optional[str] = "saved_models/ippo_bootstrap"
    """Base path for the first direct-IPPO stage and its episode checkpoints."""

    bootstrap_final_save_model_path: Optional[str] = "saved_models/ippo_direct"
    """Base path for the second direct-IPPO stage final checkpoint."""

    eval_ippo_agent0_path: Optional[str] = None
    """Direct-IPPO agent0 checkpoint used by evaluate mode."""

    eval_pool_agent0_path: Optional[str] = None
    """Opponent-pool-trained agent0 checkpoint used by evaluate mode."""

    eval_opponent_checkpoint_dir: Optional[str] = None
    """Opponent checkpoint directory for evaluate mode; falls back to pool dirs."""

    eval_deterministic: bool = True
    """Use argmax actions in evaluate mode instead of sampling."""

@dataclass
class TrainingContext:
    args: IppoPoolArgs
    writer: Any
    device: torch.device
    envs: Any
    agents: list[IPPOAgent]
    optimizers: list[optim.Optimizer]
    reward_normalizers: list[Optional[RewardNormalizer]]
    next_obs: torch.Tensor
    next_done: torch.Tensor
    global_step: int
    start_update: int
    episode_count: int
    episode_returns: list[deque]

@dataclass
class OpponentGroupStat:
    reward_ema: float
    n_games: int = 0
    last_reward: float = 0.0

class OpponentPoolState:
    """Queue-backed opponent_pool[agent_id][slot_index]."""
    def __init__(
        self,
        agent_ids: list[int],
        per_agent_max_size: int,
        epsilon: float,
        temperature: float,
        reward_ema: float,
        default_reward_score: float,
        delete_replaced_checkpoints: bool = False,
    ):
        self.agent_ids = list(agent_ids)
        
        self.entries_by_agent: dict[int, list[PoolEntry]] = {
            agent_id: [] for agent_id in self.agent_ids
        }
        self.per_agent_max_size = int(per_agent_max_size)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.reward_ema = float(reward_ema)
        self.default_reward_score = float(default_reward_score)
        self.delete_replaced_checkpoints = bool(delete_replaced_checkpoints)
        self.stats: dict[tuple[int, tuple[str, ...]], OpponentGroupStat] = {}

    def add_entry(self, entry: PoolEntry) -> Optional[PoolEntry]:
        items = self.entries_by_agent.setdefault(entry.agent_id, [])
        items.append(entry)
        removed = None
        if len(items) > self.per_agent_max_size:
            removed = items.pop(0)
            if self.delete_replaced_checkpoints:
                self._remove_checkpoint_file(removed.checkpoint_path)
        self._refresh_slot_indices(entry.agent_id)
        return removed

    def entries_for(self, agent_id: int) -> list[PoolEntry]:
        return self.entries_by_agent.get(agent_id, [])

    def group_count(self, opponent_ids: list[int]) -> int:
        if not opponent_ids:
            return 0
        return min(len(self.entries_for(agent_id)) for agent_id in opponent_ids)

    def sample_group(
        self,
        main_agent_id: int,
        opponent_ids: list[int],
        rng: random.Random,
        force_uniform: bool = False,
    ) -> tuple[int, list[PoolEntry], np.ndarray]:
        count = self.group_count(opponent_ids)
        if count <= 0:
            raise RuntimeError(
                f"Opponent pool has no complete group for opponents={opponent_ids}."
            )

        candidates = [
            [self.entries_by_agent[agent_id][slot] for agent_id in opponent_ids]
            for slot in range(count)
        ]
        if force_uniform or rng.random() < self.epsilon:
            probs = np.full(count, 1.0 / count, dtype=np.float64)
        else:
            rewards = np.asarray(
                [self._group_reward(main_agent_id, group) for group in candidates],
                dtype=np.float64,
            )
            temp = max(self.temperature, 1e-6)
            logits = -(rewards - rewards.min()) / temp
            logits -= logits.max()
            weights = np.exp(logits)
            probs = weights / weights.sum()

        slot_index = int(rng.choices(range(count), weights=probs.tolist(), k=1)[0])
        return slot_index, candidates[slot_index], probs

    def record_result(
        self,
        main_agent_id: int,
        group: list[PoolEntry],
        mean_reward: float,
        n_games: int,
    ) -> None:
        key = self._group_key(main_agent_id, group)
        stat = self.stats.get(key)
        if stat is None:
            stat = OpponentGroupStat(reward_ema=float(mean_reward))
            self.stats[key] = stat
        else:
            stat.reward_ema = (
                self.reward_ema * stat.reward_ema
                + (1.0 - self.reward_ema) * float(mean_reward)
            )
        stat.n_games += int(n_games)
        stat.last_reward = float(mean_reward)

        for entry in group:
            entry.main_reward_ema = stat.reward_ema
            entry.n_games += int(n_games)

    def to_rows(self) -> list[dict[str, Any]]:
        rows = []
        for agent_id, entries in self.entries_by_agent.items():
            for entry in entries:
                rows.append(
                    {
                        "agent_id": agent_id,
                        "slot_index": entry.slot_index,
                        "checkpoint_path": entry.checkpoint_path,
                        "global_step": entry.global_step,
                        "source": entry.source,
                        "main_reward_ema": entry.main_reward_ema,
                        "n_games": entry.n_games,
                    }
                )
        return rows

    def _refresh_slot_indices(self, agent_id: int) -> None:
        for slot_index, entry in enumerate(self.entries_by_agent.get(agent_id, [])):
            entry.slot_index = slot_index

    def _group_reward(self, main_agent_id: int, group: list[PoolEntry]) -> float:
        stat = self.stats.get(self._group_key(main_agent_id, group))
        if stat is None:
            return self.default_reward_score
        return stat.reward_ema

    @staticmethod
    def _group_key(main_agent_id: int, group: list[PoolEntry]) -> tuple[int, tuple[str, ...]]:
        return (
            int(main_agent_id),
            tuple(str(pathlib.Path(entry.checkpoint_path).resolve()) for entry in group),
        )

    @staticmethod
    def _remove_checkpoint_file(path: str) -> None:
        try:
            pathlib.Path(path).unlink()
        except OSError:
            pass

def _parse_cli(args: IppoPoolArgs) -> IppoPoolArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", choices=["ippo_bootstrap", "pool_cycle", "evaluate", "full"])
    parser.add_argument("--env-path")
    parser.add_argument("--config-path")
    parser.add_argument("--n-parallel", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--save-model-path")
    parser.add_argument("--load-model-path")
    parser.add_argument("--resume-from")
    parser.add_argument("--ppo-model-paths", nargs=4)
    parser.add_argument("--pool-initial-checkpoint-dir")
    parser.add_argument("--pool-checkpoint-dir")
    parser.add_argument("--pool-slots-per-agent", type=int)
    parser.add_argument("--pool-initial-keep-per-agent", type=int)
    parser.add_argument("--pool-phase-timesteps", type=int)
    parser.add_argument("--pool-rounds", type=int)
    parser.add_argument("--pool-final-timesteps", type=int)
    parser.add_argument("--pool-save-interval", type=int)
    parser.add_argument("--eval-ippo-agent0-path")
    parser.add_argument("--eval-pool-agent0-path")
    parser.add_argument("--eval-opponent-checkpoint-dir")
    parsed = parser.parse_args()

    mapping = {
        "run_mode": "run_mode",
        "env_path": "env_path",
        "config_path": "config_path",
        "n_parallel": "n_parallel",
        "seed": "seed",
        "total_timesteps": "total_timesteps",
        "num_steps": "num_steps",
        "save_model_path": "save_model_path",
        "load_model_path": "load_model_path",
        "resume_from": "resume_from",
        "pool_initial_checkpoint_dir": "pool_initial_checkpoint_dir",
        "pool_checkpoint_dir": "pool_checkpoint_dir",
        "pool_slots_per_agent": "pool_slots_per_agent",
        "pool_initial_keep_per_agent": "pool_initial_keep_per_agent",
        "pool_phase_timesteps": "pool_phase_timesteps",
        "pool_rounds": "pool_rounds",
        "pool_final_timesteps": "pool_final_timesteps",
        "pool_save_interval": "pool_save_interval",
        "eval_ippo_agent0_path": "eval_ippo_agent0_path",
        "eval_pool_agent0_path": "eval_pool_agent0_path",
        "eval_opponent_checkpoint_dir": "eval_opponent_checkpoint_dir",
    }
    for cli_name, field_name in mapping.items():
        value = getattr(parsed, cli_name)
        if value is not None:
            setattr(args, field_name, value)
    if parsed.ppo_model_paths is not None:
        args.ppo_model_paths = list(parsed.ppo_model_paths)
    return args

def _agent_ids(args: IppoPoolArgs) -> list[int]:
    return [cfg.agent_id for cfg in args.agent_configs]

def _with_train_flags(
    args: IppoPoolArgs,
    train_ids: set[int],
    policy_opponent_ids: Optional[set[int]] = None,
) -> list[AgentConfig]:
    policy_opponent_ids = policy_opponent_ids or set()
    configs = []
    for cfg in args.agent_configs:
        train = cfg.agent_id in train_ids#该智能体是否进行训练
        configs.append(
            replace(
                cfg,
                train=train,
                act_when_not_training=(not train and cfg.agent_id in policy_opponent_ids),
            )
        )
    return configs

def _configure_runtime_args(args: IppoPoolArgs, envs: Any, seg: Any) -> None:
    n_agents = len(args.agent_configs)
    args.num_agents = n_agents
    args.num_envs = envs.num_envs
    if args.num_envs % n_agents != 0:
        raise ValueError(
            "IPPO expects Godot slots to be n_parallel * n_agents: "
            f"envs.num_envs={args.num_envs}, n_agents={n_agents}."
        )
    args.num_game_envs = args.num_envs // n_agents#并行环境数量
    if args.num_game_envs != args.n_parallel:
        raise ValueError(
            "Godot env slot count does not match args.n_parallel: "
            f"num_game_envs={args.num_game_envs}, n_parallel={args.n_parallel}."
        )
    obs_shape = envs.single_observation_space.shape
    if len(obs_shape) != 1 or obs_shape[0] != seg.total:
        raise ValueError(
            f"Observation dimension mismatch: env={obs_shape}, configured={seg.total}."
        )
    args.batch_size = args.num_game_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size <= 0:
        raise ValueError(
            "num_minibatches is too large: "
            f"batch_size={args.batch_size}, num_minibatches={args.num_minibatches}."
        )

def _make_agents(
    args: IppoPoolArgs,
    n_actions: int,
    seg: Any,
    device: torch.device,
) -> tuple[list[IPPOAgent], list[optim.Optimizer], list[Optional[RewardNormalizer]]]:
    agents: list[IPPOAgent] = []
    optimizers: list[optim.Optimizer] = []
    reward_normalizers: list[Optional[RewardNormalizer]] = []
    for cfg in args.agent_configs:
        agent = IPPOAgent(n_actions, seg, cfg).to(device)
        print(f"[Agent {cfg.agent_id}] network_type={cfg.network_type}, params={agent.num_params():,}")
        agents.append(agent)
        optimizers.append(optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5))
        reward_normalizers.append(
            RewardNormalizer(clip=cfg.reward_clip) if cfg.reward_norm else None
        )
    return agents, optimizers, reward_normalizers


def setup_training_context(args: IppoPoolArgs) -> TrainingContext:
    writer, device, envs, seg, _ = init_training_setup(args)
    _configure_runtime_args(args, envs, seg)
    n_actions = int(envs.single_action_space.n)
    agents, optimizers, reward_normalizers = _make_agents(args, n_actions, seg, device)

    #加载PPO模型参数
    load_ppo_models_if_requested(args.ppo_model_paths, agents, device)
    resume_path = args.resume_from or args.load_model_path
    is_resume = bool(args.resume_from)
    #加载检查点，从四个模型开始ippo同时训练时应该保持检查点路径为None
    global_step, start_update, episode_count, episode_returns = load_checkpoint_if_requested(
        resume_path,
        is_resume,
        agents,
        optimizers,
        reward_normalizers,
        args,
        device,
    )

    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(np.asarray(next_obs_array, dtype=np.float32), device=device)
    if next_obs.shape[0] != args.num_envs:
        raise ValueError(
            "Reset observation count does not match Godot slots: "
            f"{next_obs.shape[0]} != {args.num_envs}."
        )
    next_done = torch.zeros(args.num_envs, device=device)
    return TrainingContext(
        args=args,
        writer=writer,
        device=device,
        envs=envs,
        agents=agents,
        optimizers=optimizers,
        reward_normalizers=reward_normalizers,
        next_obs=next_obs,
        next_done=next_done,
        global_step=global_step,
        start_update=start_update,
        episode_count=episode_count,
        episode_returns=episode_returns,
    )


def close_context(ctx: Optional[TrainingContext]) -> None:
    if ctx is None:
        return
    try:
        ctx.envs.close()
    finally:
        ctx.writer.close()


def _capture_agent_states(agents: list[IPPOAgent]) -> list[dict[str, torch.Tensor]]:
    return [
        {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
        for agent in agents
    ]


def _restore_agent_state(
    agent: IPPOAgent,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    agent.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def _load_agent_state(
    path: str,
    agent_id: int,
    agent: IPPOAgent,
    device: torch.device,
) -> dict:
    ckpt = load_full_checkpoint(str(path), device)
    ckpt_agent_id = ckpt.get("agent_id")
    if ckpt_agent_id is not None and int(ckpt_agent_id) != int(agent_id):
        raise ValueError(
            f"Checkpoint agent_id mismatch: expected {agent_id}, got {ckpt_agent_id}, path={path}"
        )
    if "agent_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing agent_state_dict: {path}")
    agent.load_state_dict(ckpt["agent_state_dict"])
    return ckpt


def save_selected_agents(
    save_path: Optional[str],
    selected_agent_ids: set[int],
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    reward_normalizers: list[Optional[RewardNormalizer]],
    args: IppoPoolArgs,
    extra: Optional[dict] = None,
) -> list[str]:
    if save_path is None:
        return []
    original_configs = args.agent_configs
    args.agent_configs = _with_train_flags(args, selected_agent_ids)
    try:
        return save_ippo_model(
            save_path,
            agents,
            optimizers,
            reward_normalizers,
            args,
            extra=extra,
            train_only=True,
        )
    finally:
        args.agent_configs = original_configs


def run_ippo_training_job(
    args: IppoPoolArgs,
    final_save_agent_ids: set[int],
) -> dict:
    ctx = None
    try:
        ctx = setup_training_context(args)
        final_state = train_ippo(
            args,
            ctx.agents,
            ctx.optimizers,
            ctx.envs,
            ctx.device,
            ctx.writer,
            ctx.reward_normalizers,
            ctx.next_obs,
            ctx.next_done,
            ctx.global_step,
            ctx.start_update,
            ctx.episode_count,
            ctx.episode_returns,
        )
        save_selected_agents(
            args.save_model_path,
            final_save_agent_ids,
            ctx.agents,
            ctx.optimizers,
            ctx.reward_normalizers,
            args,
            extra=final_state,
        )
        return final_state
    finally:
        close_context(ctx)


def run_ippo_bootstrap(args: IppoPoolArgs) -> None:
    all_ids = set(_agent_ids(args))#{0, 1, 2, 3...}

    #阶段1 同时训练并且给每个智能体保存检查点
    phase1 = copy.deepcopy(args)#配置
    phase1.use_opponent_pool = False
    phase1.agent_configs = _with_train_flags(phase1, all_ids)#所有智能体都训练
    phase1.total_timesteps = int(args.pool_bootstrap_checkpoint_timesteps)
    phase1.save_checkpoint = True
    phase1.max_checkpoints = max(
        int(phase1.max_checkpoints),
        int(args.pool_initial_keep_per_agent),
    )
    phase1.save_model_path = args.bootstrap_save_model_path
    print(
        "[Bootstrap] stage 1: "
        f"{phase1.total_timesteps} steps, checkpoints -> {phase1.save_model_path}"
    )
    #ippo同时训练一段时间并给所有智能体保存检查点
    run_ippo_training_job(phase1, all_ids)

    #阶段2 继续训练并给主智能体保存最终模型
    phase2 = copy.deepcopy(args)
    phase2.use_opponent_pool = False
    phase2.agent_configs = _with_train_flags(phase2, all_ids)#所有智能体都训练
    phase2.ppo_model_paths = [None for _ in phase2.agent_configs]#不加载模型参数
    phase2.resume_from = args.bootstrap_save_model_path #从1阶段保存的各个检查点恢复
    phase2.load_model_path = None
    phase2.total_timesteps = int(args.pool_bootstrap_checkpoint_timesteps + args.pool_bootstrap_extra_timesteps)#因为是从中断点恢复，所以加
    phase2.save_checkpoint = False#不保存检查点
    phase2.save_model_path = args.bootstrap_final_save_model_path#保存最终模型
    final_ids = {int(agent_id) for agent_id in args.pool_final_save_agent_ids}#最终保存的智能体ID
    print(
        "[Bootstrap] stage 2: "
        f"continue to {phase2.total_timesteps} total steps, final ids={sorted(final_ids)}"
    )
    run_ippo_training_job(phase2, final_ids)


_AGENT_RE_TEMPLATE = r"_agent{agent_id}(?:\.pt|$)"
_EPISODE_RE = re.compile(r"_episode(\d+)")
_STEP_RE = re.compile(r"_step(\d+)")


def _checkpoint_sort_key(path: pathlib.Path) -> tuple[int, float, str]:
    name = path.name
    match = _EPISODE_RE.search(name) or _STEP_RE.search(name)
    number = int(match.group(1)) if match else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return number, mtime, name


def find_recent_agent_checkpoints(
    checkpoint_dir: str,
    agent_id: int,
    keep_latest: int,
) -> list[pathlib.Path]:
    root = pathlib.Path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    agent_re = re.compile(_AGENT_RE_TEMPLATE.format(agent_id=agent_id))
    candidates = [
        path for path in root.rglob("*.pt")
        if agent_re.search(path.name)
    ]
    candidates.sort(key=_checkpoint_sort_key)
    if keep_latest > 0:
        candidates = candidates[-keep_latest:]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for agent_{agent_id} under {checkpoint_dir}"
        )
    return candidates


def build_initial_pool(args: IppoPoolArgs) -> OpponentPoolState:
    checkpoint_dir = args.pool_initial_checkpoint_dir or args.pool_checkpoint_dir
    if checkpoint_dir is None:
        raise ValueError(
            "pool_initial_checkpoint_dir or pool_checkpoint_dir is required for pool mode."
        )

    pool = OpponentPoolState(
        agent_ids=_agent_ids(args),
        per_agent_max_size=args.pool_slots_per_agent,
        epsilon=args.pool_epsilon,
        temperature=args.pool_pfsp_temperature,
        reward_ema=args.pool_reward_ema,
        default_reward_score=args.pool_default_reward_score,
        delete_replaced_checkpoints=args.pool_delete_replaced_checkpoints,
    )
    for agent_id in _agent_ids(args):
        paths = find_recent_agent_checkpoints(
            checkpoint_dir,
            agent_id,
            args.pool_initial_keep_per_agent,
        )
        for path in paths:
            number, _, _ = _checkpoint_sort_key(path)
            pool.add_entry(
                PoolEntry(
                    checkpoint_path=str(path),
                    agent_id=agent_id,
                    global_step=number,
                    source="initial",
                )
            )
    print(
        "[Pool] loaded "
        + " ".join(
            f"agent_{agent_id}:{len(pool.entries_for(agent_id))}"
            for agent_id in _agent_ids(args)
        )
    )
    return pool


def _pool_checkpoint_dir(args: IppoPoolArgs) -> pathlib.Path:
    if args.pool_checkpoint_dir:
        return pathlib.Path(args.pool_checkpoint_dir)
    return pathlib.Path(args.experiment_dir) / "pool_checkpoints"


def save_main_agent_to_pool(
    pool: OpponentPoolState,
    main_agent_id: int,
    ctx: TrainingContext,
    phase_name: str,
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
    saved_paths = save_selected_agents(
        str(base_path),
        {main_agent_id},
        ctx.agents,
        ctx.optimizers,
        ctx.reward_normalizers,
        ctx.args,
        extra={**train_state, "pool_rows": pool.to_rows()},
    )
    if not saved_paths:
        return
    pool.add_entry(
        PoolEntry(
            checkpoint_path=saved_paths[0],
            agent_id=main_agent_id,
            global_step=ctx.global_step,
            source=phase_name,
        )
    )
    print(
        f"[Pool] agent_{main_agent_id} snapshot added; "
        f"size={len(pool.entries_for(main_agent_id))}/{pool.per_agent_max_size}"
    )


def _load_latest_pool_entries_into_agents(
    pool: OpponentPoolState,
    ctx: TrainingContext,
) -> None:
    for agent_id in _agent_ids(ctx.args):
        entries = pool.entries_for(agent_id)
        if not entries:
            continue
        _load_agent_state(
            entries[-1].checkpoint_path,
            agent_id,
            ctx.agents[agent_id],
            ctx.device,
        )
        print(f"[Pool Init] current agent_{agent_id} <- {entries[-1].checkpoint_path}")


def _phase_update_count(args: IppoPoolArgs, phase_timesteps: int) -> tuple[int, int]:
    n_agents = len(args.agent_configs)
    if args.count_steps_by == "env_steps":
        denom = args.num_game_envs * args.num_steps
        step_increment = args.num_game_envs
    elif args.count_steps_by == "agent_steps":
        denom = args.num_game_envs * n_agents * args.num_steps
        step_increment = args.num_envs
    else:
        raise ValueError(
            f"Unknown count_steps_by={args.count_steps_by!r}; expected env_steps/agent_steps."
        )
    return max(1, math.ceil(int(phase_timesteps) / denom)), step_increment


def _load_opponent_group(
    group: list[PoolEntry],
    ctx: TrainingContext,
) -> None:
    for entry in group:
        _load_agent_state(
            entry.checkpoint_path,
            entry.agent_id,
            ctx.agents[entry.agent_id],
            ctx.device,
        )


def _sample_and_load_opponents(
    pool: OpponentPoolState,
    main_agent_id: int,
    ctx: TrainingContext,
    rng: random.Random,
) -> tuple[int, list[PoolEntry]]:
    opponent_ids = [agent_id for agent_id in _agent_ids(ctx.args) if agent_id != main_agent_id]
    slot_index, group, probs = pool.sample_group(main_agent_id, opponent_ids, rng)
    _load_opponent_group(group, ctx)
    opponent_desc = " ".join(
        f"agent_{entry.agent_id}[{entry.slot_index}]" for entry in group
    )
    print(
        f"[Pool Sample] main=agent_{main_agent_id} slot={slot_index} "
        f"opponents={opponent_desc} prob={probs[slot_index]:.3f}"
    )
    return slot_index, group


def train_pool_phase(
    ctx: TrainingContext,
    pool: OpponentPoolState,
    current_agent_states: list[dict[str, torch.Tensor]],
    main_agent_id: int,
    phase_timesteps: int,
    phase_name: str,
    rng: random.Random,
) -> None:
    args = ctx.args
    n_agents = len(args.agent_configs)
    n_game_envs = args.num_game_envs

    _restore_agent_state(ctx.agents[main_agent_id], current_agent_states[main_agent_id], ctx.device)
    args.agent_configs = _with_train_flags(
        args,
        train_ids={main_agent_id},
        policy_opponent_ids=set(_agent_ids(args)) - {main_agent_id},
    )
    phase_updates, step_increment = _phase_update_count(args, phase_timesteps)
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, ctx.device) for agent in ctx.agents]
    _, current_group = _sample_and_load_opponents(pool, main_agent_id, ctx, rng)

    start_time = time.time()
    last_pool_save_step = ctx.global_step

    for local_update in range(1, phase_updates + 1):
        if args.anneal_lr:
            progress = 1.0 - (local_update - 1.0) / phase_updates
            cfg = args.agent_configs[main_agent_id]
            ctx.optimizers[main_agent_id].param_groups[0]["lr"] = progress * cfg.learning_rate

        rollouts, ctx.global_step, rnn_states, new_episode_returns = collect_parallel_rollout_ippo(
            args.agent_configs,
            ctx.agents,
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
        ctx.next_obs = torch.stack([r.next_obs for r in rollouts], dim=1).reshape(args.num_envs, -1)
        ctx.next_done = torch.stack([r.next_done for r in rollouts], dim=1).reshape(args.num_envs)
        ctx.episode_count += _count_completed_episodes(rollouts)

        completed_returns = new_episode_returns[main_agent_id]
        if completed_returns:
            mean_reward = float(np.mean(np.asarray(completed_returns, dtype=np.float64)))
            pool.record_result(
                main_agent_id,
                current_group,
                mean_reward,
                n_games=len(completed_returns),
            )
            _, current_group = _sample_and_load_opponents(pool, main_agent_id, ctx, rng)
            rnn_states = [
                agent.get_initial_state(n_game_envs, ctx.device)
                for agent in ctx.agents
            ]

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

        if ctx.global_step - last_pool_save_step >= args.pool_save_interval:
            current_agent_states[main_agent_id] = _capture_agent_states(ctx.agents)[main_agent_id]
            save_main_agent_to_pool(pool, main_agent_id, ctx, phase_name)
            last_pool_save_step = ctx.global_step

    current_agent_states[main_agent_id] = _capture_agent_states(ctx.agents)[main_agent_id]
    save_main_agent_to_pool(pool, main_agent_id, ctx, f"{phase_name}_end")


def run_pool_cycle(args: IppoPoolArgs) -> None:
    pool = build_initial_pool(args)
    setup_args = copy.deepcopy(args)
    setup_args.use_opponent_pool = True
    setup_args.agent_configs = _with_train_flags(setup_args, set(_agent_ids(setup_args)))

    ctx = None
    try:
        ctx = setup_training_context(setup_args)
        if not any(ctx.args.ppo_model_paths) and not ctx.args.resume_from and not ctx.args.load_model_path:
            _load_latest_pool_entries_into_agents(pool, ctx)
        current_agent_states = _capture_agent_states(ctx.agents)
        rng = random.Random(ctx.args.seed)

        for round_idx in range(int(ctx.args.pool_rounds)):
            for main_agent_id in ctx.args.pool_main_agent_order:
                phase_name = f"round{round_idx + 1}_agent{main_agent_id}"
                print(
                    f"[Pool Phase] {phase_name}: "
                    f"{ctx.args.pool_phase_timesteps} steps"
                )
                train_pool_phase(
                    ctx,
                    pool,
                    current_agent_states,
                    int(main_agent_id),
                    int(ctx.args.pool_phase_timesteps),
                    phase_name,
                    rng,
                )

        final_id = int(ctx.args.pool_final_agent_id)
        print(
            f"[Pool Final] agent_{final_id}: "
            f"{ctx.args.pool_final_timesteps} steps"
        )
        train_pool_phase(
            ctx,
            pool,
            current_agent_states,
            final_id,
            int(ctx.args.pool_final_timesteps),
            f"final_agent{final_id}",
            rng,
        )

        for agent_id, state in enumerate(current_agent_states):
            _restore_agent_state(ctx.agents[agent_id], state, ctx.device)
        final_state = _build_train_state(
            ctx.global_step,
            ctx.start_update,
            ctx.episode_count,
            ctx.optimizers,
            ctx.episode_returns,
        )
        final_ids = {int(agent_id) for agent_id in ctx.args.pool_final_save_agent_ids}
        save_selected_agents(
            ctx.args.save_model_path,
            final_ids,
            ctx.agents,
            ctx.optimizers,
            ctx.reward_normalizers,
            ctx.args,
            extra={**final_state, "pool_rows": pool.to_rows()},
        )
    finally:
        close_context(ctx)


def _build_eval_agent(
    path: str,
    agent_id: int,
    args: IppoPoolArgs,
    n_actions: int,
    seg: Any,
    device: torch.device,
) -> IPPOAgent:
    cfg = args.agent_configs[agent_id]
    agent = IPPOAgent(n_actions, seg, cfg).to(device)
    _load_agent_state(path, agent_id, agent, device)
    agent.eval()
    return agent


def _select_action(
    agent: IPPOAgent,
    obs: torch.Tensor,
    rnn_state: Optional[torch.Tensor],
    deterministic: bool,
) -> tuple[int, Optional[torch.Tensor]]:
    with torch.no_grad():
        if deterministic:
            features, next_state = agent._forward_features(obs, rnn_state)
            logits = agent.actor(features)
            action = logits.argmax(dim=1)
            return int(action.item()), next_state
        action, _, _, _, next_state = agent.get_action_and_value(
            obs,
            rnn_state=rnn_state,
            return_state=True,
        )
        return int(action.item()), next_state


def _write_eval_rows(path: Optional[str], rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    output_path = pathlib.Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_label", "group_id", "episode", "agent_id", "reward"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Eval] rows saved to {output_path}")


def run_evaluation(args: IppoPoolArgs) -> None:
    eval_dir = (
        args.eval_opponent_checkpoint_dir
        or args.pool_initial_checkpoint_dir
        or args.pool_checkpoint_dir
    )
    if eval_dir is None:
        raise ValueError("Evaluation needs eval_opponent_checkpoint_dir or a pool checkpoint dir.")

    eval_args = copy.deepcopy(args)
    eval_args.n_parallel = int(args.pool_eval_groups)
    eval_args.pool_initial_checkpoint_dir = eval_dir
    eval_args.agent_configs = _with_train_flags(
        eval_args,
        train_ids=set(),
        policy_opponent_ids=set(_agent_ids(eval_args)),
    )
    writer, device, envs, seg, _ = init_training_setup(eval_args)
    try:
        _configure_runtime_args(eval_args, envs, seg)
        n_actions = int(envs.single_action_space.n)
        pool = build_initial_pool(eval_args)
        rng = random.Random(eval_args.seed)
        opponent_ids = [agent_id for agent_id in _agent_ids(eval_args) if agent_id != 0]
        group_count = pool.group_count(opponent_ids)
        if group_count >= int(eval_args.pool_eval_groups):
            slot_indices = rng.sample(range(group_count), int(eval_args.pool_eval_groups))
        else:
            slot_indices = [
                rng.randrange(group_count) for _ in range(int(eval_args.pool_eval_groups))
            ]
        opponent_groups = [
            [pool.entries_by_agent[agent_id][slot] for agent_id in opponent_ids]
            for slot in slot_indices
        ]

        model_paths = {
            "direct_ippo": eval_args.eval_ippo_agent0_path,
            "opponent_pool": eval_args.eval_pool_agent0_path,
        }
        rows: list[dict[str, Any]] = []
        for model_label, model_path in model_paths.items():
            if model_path is None:
                continue
            policies: list[list[IPPOAgent]] = []
            for group in opponent_groups:
                group_agents: list[Optional[IPPOAgent]] = [None for _ in _agent_ids(eval_args)]
                group_agents[0] = _build_eval_agent(
                    model_path, 0, eval_args, n_actions, seg, device
                )
                for entry in group:
                    group_agents[entry.agent_id] = _build_eval_agent(
                        entry.checkpoint_path,
                        entry.agent_id,
                        eval_args,
                        n_actions,
                        seg,
                        device,
                    )
                policies.append(group_agents)  # type: ignore[arg-type]
            rows.extend(
                _evaluate_policy_groups(
                    model_label,
                    policies,
                    eval_args,
                    envs,
                    device,
                )
            )

        _write_eval_rows(eval_args.pool_eval_output_path, rows)
        _print_eval_summary(rows)
    finally:
        envs.close()
        writer.close()


def _evaluate_policy_groups(
    model_label: str,
    policies: list[list[IPPOAgent]],
    args: IppoPoolArgs,
    envs: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    n_groups = len(policies)
    n_agents = len(args.agent_configs)
    obs_raw, _ = envs.reset(seed=args.seed)
    next_obs = np.asarray(obs_raw, dtype=np.float32)
    episode_rewards = np.zeros((n_groups, n_agents), dtype=np.float64)
    episode_counts = np.zeros(n_groups, dtype=np.int64)
    rnn_states: list[list[Optional[torch.Tensor]]] = []
    for group in policies:
        rnn_states.append([
            agent.get_initial_state(1, device) if agent.is_recurrent else None
            for agent in group
        ])

    rows: list[dict[str, Any]] = []
    while np.any(episode_counts < args.pool_eval_episodes_per_group):
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        obs_by_env = obs_t.view(n_groups, n_agents, -1)
        actions_by_env = np.zeros((n_groups, n_agents), dtype=np.int64)
        for group_idx in range(n_groups):
            for agent_idx in range(n_agents):
                action, next_state = _select_action(
                    policies[group_idx][agent_idx],
                    obs_by_env[group_idx, agent_idx].unsqueeze(0),
                    rnn_states[group_idx][agent_idx],
                    args.eval_deterministic,
                )
                actions_by_env[group_idx, agent_idx] = action
                rnn_states[group_idx][agent_idx] = next_state

        next_obs, rewards, terms, truncs, _ = envs.step(actions_by_env.reshape(-1))
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards_by_env = np.asarray(rewards, dtype=np.float32).reshape(n_groups, n_agents)
        dones_by_env = np.logical_or(terms, truncs).reshape(n_groups, n_agents)
        episode_rewards += rewards_by_env

        for group_idx in range(n_groups):
            if episode_counts[group_idx] >= args.pool_eval_episodes_per_group:
                continue
            if np.any(dones_by_env[group_idx]):
                episode = int(episode_counts[group_idx])
                for agent_idx in range(n_agents):
                    rows.append(
                        {
                            "model_label": model_label,
                            "group_id": group_idx,
                            "episode": episode,
                            "agent_id": args.agent_configs[agent_idx].agent_id,
                            "reward": float(episode_rewards[group_idx, agent_idx]),
                        }
                    )
                episode_counts[group_idx] += 1
                episode_rewards[group_idx, :] = 0.0
                rnn_states[group_idx] = [
                    agent.get_initial_state(1, device) if agent.is_recurrent else None
                    for agent in policies[group_idx]
                ]
    return rows


def _print_eval_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[Eval] no rows collected.")
        return
    labels = sorted({row["model_label"] for row in rows})
    for label in labels:
        rewards = np.asarray(
            [
                row["reward"] for row in rows
                if row["model_label"] == label and int(row["agent_id"]) == 0
            ],
            dtype=np.float64,
        )
        if rewards.size == 0:
            continue
        print(
            f"[Eval] {label}: agent0 mean={rewards.mean():.3f} "
            f"std={rewards.std(ddof=0):.3f} n={rewards.size}"
        )


def run_full_plan(args: IppoPoolArgs) -> None:
    run_ippo_bootstrap(args)
    pool_args = copy.deepcopy(args)
    pool_args.load_model_path = None
    pool_args.resume_from = None
    pool_args.ppo_model_paths = [None for _ in pool_args.agent_configs]
    if pool_args.pool_initial_checkpoint_dir is None:
        pool_args.pool_initial_checkpoint_dir = (
            str(pathlib.Path(args.bootstrap_save_model_path).parent)
            if args.bootstrap_save_model_path
            else None
        )
    run_pool_cycle(pool_args)


def main() -> None:
    args = _parse_cli(IppoPoolArgs())
    if args.run_mode == "ippo_bootstrap":
        run_ippo_bootstrap(args)
    elif args.run_mode == "pool_cycle":
        run_pool_cycle(args)
    elif args.run_mode == "evaluate":
        run_evaluation(args)
    elif args.run_mode == "full":
        run_full_plan(args)
    else:
        raise ValueError(f"Unknown run_mode={args.run_mode!r}")

if __name__ == "__main__":
    main()
