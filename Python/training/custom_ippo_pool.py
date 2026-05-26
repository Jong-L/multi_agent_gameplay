"""
Opponent-pool IPPO training for Tiny Swords.

This script intentionally reuses the existing custom_ippo.py PPO/IPPO pieces:
rollout collection, PPO updates, logging, checkpoint save/load, and Godot setup.

Supported modes:
  - ippo_bootstrap: load four single-agent PPO checkpoints, train all four
    agents together, save intermediate IPPO checkpoints, then continue direct
    IPPO training and save selected final agents.
  - bootstrap_checkpoint: Phase 1 only — train jointly, save episode checkpoints.
  - bootstrap_direct: Phase 2 only — resume from Phase 1 checkpoint, train direct IPPO.
  - pool_cycle: build opponent_pool[agent_id][slot_index] from recent IPPO
    checkpoints, train one main agent per phase, and sample frozen opponents
    with higher probability for groups that produced lower main-agent returns.
  - evaluate: compare two agent0 checkpoints against the same sampled opponent
    groups and write a CSV summary.
  - full: run bootstrap_checkpoint + bootstrap_direct + pool_cycle sequentially.
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

    run_mode: str = "evaluate"
    """ippo_bootstrap / bootstrap_checkpoint / bootstrap_direct / pool_cycle / evaluate / full"""

    bootstrap_save_model_path: Optional[str] = "saved_models/ippo_bootstrap"
    """Base path for the first direct-IPPO stage and its episode checkpoints."""

    bootstrap_final_save_model_path: Optional[str] = "saved_models/ippo_direct"
    """Base path for the second direct-IPPO stage final checkpoint."""

    eval_ippo_agent0_path: Optional[str] = "saved_models/ippo_direct_agent0.pt"
    """Direct-IPPO agent0 checkpoint used by evaluate mode."""

    eval_pool_agent0_path: Optional[str] = "saved_models/agent0_extra_pool_step102400_agent0.pt"
    """Opponent-pool-trained agent0 checkpoint used by evaluate mode."""

    eval_opponent_checkpoint_dir: Optional[str] = "saved_models/ippo_pool_checkpoints"
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
    reward_ema: float#指数平均奖励
    n_games: int = 0#对战场次
    last_reward: float = 0.0#最后奖励

class OpponentPoolState:
    """Queue-backed opponent_pool[agent_id][slot_index]."""
    def __init__(
        self,
        agent_ids: list[int],
        per_agent_max_size: int,
        epsilon: float,
        temperature: float,
        reward_ema_coef: float,
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
        self.reward_ema_coef = float(reward_ema_coef)
        self.default_reward_score = float(default_reward_score)
        self.delete_replaced_checkpoints = bool(delete_replaced_checkpoints)
        self.stats: dict[tuple[int, tuple[str, ...]], OpponentGroupStat] = {}

    def add_entry(self, entry: PoolEntry) -> Optional[PoolEntry]:
        """在对手池中加入记录"""
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
        """获取指定智能体的所有检查点"""
        return self.entries_by_agent.get(agent_id, [])

    def group_count(self, opponent_ids: list[int]) -> int:
        """获取可用的对手group数量"""
        if not opponent_ids:
            return 0
        return min(len(self.entries_for(agent_id)) for agent_id in opponent_ids)

    def sample_group(
        self,
        main_agent_id: int,
        opponent_ids: list[int],
        rng: random.Random,
        force_uniform: bool = False,#是否强制均匀采样
    ) -> tuple[int, list[PoolEntry], np.ndarray]:
        """从对手池中采样一个group"""
        count = self.group_count(opponent_ids)
        if count <= 0:
            raise RuntimeError(f"Opponent pool has no complete group for opponents={opponent_ids}.")

        candidates = [
            [self.entries_by_agent[agent_id][slot] for agent_id in opponent_ids]
            for slot in range(count)
        ]#candidates[opponent_count][count]
        if force_uniform or rng.random() < self.epsilon:
            probs = np.full(count, 1.0 / count, dtype=np.float64)
        else:
            # 计算所有候选对手group的奖励
            rewards = np.asarray(
                [self._group_reward(main_agent_id, group) for group in candidates],
                dtype=np.float64,
            )
            temp = max(self.temperature, 1e-6)
            logits = -(rewards - rewards.min()) / temp
            logits -= logits.max()#减去最大值以避免数值不稳定
            #softmax计算采样概率
            weights = np.exp(logits)
            probs = weights / weights.sum()

        # 从候选对手group中采样一个
        slot_index = int(rng.choices(range(count), weights=probs.tolist(), k=1)[0])
        return slot_index, candidates[slot_index], probs

    def record_result(
        self,
        main_agent_id: int,
        group: list[PoolEntry],
        mean_reward: float,
        n_games: int,
    ) -> None:
        """记录对战结果"""
        key = self._group_key(main_agent_id, group)
        stat = self.stats.get(key)
        if stat is None:
            stat = OpponentGroupStat(reward_ema=float(mean_reward))
            self.stats[key] = stat
        else:
            # 更新指数平均奖励
            stat.reward_ema = (
                self.reward_ema_coef * stat.reward_ema
                + (1.0 - self.reward_ema_coef) * float(mean_reward)
            )
        stat.n_games += int(n_games)
        stat.last_reward = float(mean_reward)

        # 更新组中每个检查点的数据
        for entry in group:
            entry.main_reward_ema = stat.reward_ema
            entry.n_games += int(n_games)

    def to_rows(self) -> list[dict[str, Any]]:
        """将对手池数据转换为字典列表"""
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
        """刷新指定智能体的快照索引"""
        for slot_index, entry in enumerate(self.entries_by_agent.get(agent_id, [])):
            entry.slot_index = slot_index

    def _group_reward(self, main_agent_id: int, group: list[PoolEntry]) -> float:
        """获取对战group的指数平均奖励"""
        stat = self.stats.get(self._group_key(main_agent_id, group))
        if stat is None:
            return self.default_reward_score
        return stat.reward_ema

    @staticmethod
    def _group_key(main_agent_id: int, group: list[PoolEntry]) -> tuple[int, tuple[str, ...]]:
        """获取主智能体查找group状态的key"""
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
    parser.add_argument("--run-mode", choices=["ippo_bootstrap", "bootstrap_checkpoint", "bootstrap_direct", "pool_cycle", "evaluate", "full"])
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
    parser.add_argument("--run-name")
    parser.add_argument("--pool-initial-checkpoint-dir")
    parser.add_argument("--pool-checkpoint-dir")
    parser.add_argument("--pool-slots-per-agent", type=int)
    parser.add_argument("--pool-initial-keep-per-agent", type=int)
    parser.add_argument("--pool-phase-timesteps", type=int)
    parser.add_argument("--pool-rounds", type=int)
    parser.add_argument("--pool-final-timesteps", type=int)
    parser.add_argument("--pool-save-interval", type=int)
    parser.add_argument("--port-offset", type=int)
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
        "run_name": "run_name",
        "pool_initial_checkpoint_dir": "pool_initial_checkpoint_dir",
        "pool_checkpoint_dir": "pool_checkpoint_dir",
        "pool_slots_per_agent": "pool_slots_per_agent",
        "pool_initial_keep_per_agent": "pool_initial_keep_per_agent",
        "pool_phase_timesteps": "pool_phase_timesteps",
        "pool_rounds": "pool_rounds",
        "pool_final_timesteps": "pool_final_timesteps",
        "pool_save_interval": "pool_save_interval",
        "port_offset": "port_offset",
        "eval_ippo_agent0_path": "eval_ippo_agent0_path",
        "eval_pool_agent0_path": "eval_pool_agent0_path",
        "eval_opponent_checkpoint_dir": "eval_opponent_checkpoint_dir",
    }
    for cli_name, field_name in mapping.items():
        value = getattr(parsed, cli_name)
        if value is not None:
            setattr(args, field_name, value)
    if parsed.ppo_model_paths is not None:
        args.ppo_model_paths = [None if isinstance(p, str) and p.lower() == "none" else p for p in parsed.ppo_model_paths]
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
    """创建IPPO智能体、优化器和奖励归一化器"""
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
    """设置训练上下文，初始化训练环境和智能体"""
    writer, device, envs, seg, _ = init_training_setup(args)
    _configure_runtime_args(args, envs, seg)
    n_actions = int(envs.single_action_space.n)
    agents, optimizers, reward_normalizers = _make_agents(args, n_actions, seg, device)

    #加载PPO模型参数，单独进行pool cycle时应该保证args.ppo_model_paths为None
    load_ppo_models_if_requested(args.ppo_model_paths, agents, device)
    resume_path = args.resume_from or args.load_model_path
    is_resume = bool(args.resume_from)
    #加载检查点，从四个模型开始ippo同时训练或单独跑pool cycle时应该保持检查点路径为None
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
    """将智能体网络参数加载到指定设备上"""
    agent.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def _load_agent_state(
    path: str,
    agent_id: int,
    agent: IPPOAgent,
    device: torch.device,
) -> dict:
    """从检查点中加载智能体网络参数"""
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
    """保存指定的智能体的检查点"""
    if save_path is None:
        return []
    original_configs = args.agent_configs
    args.agent_configs = _with_train_flags(args, selected_agent_ids)
    try:
        #不训练的智能体不会保存，直接复用保存函数
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


def run_ippo_bootstrap_checkpoint(args: IppoPoolArgs) -> None:
    """Phase 1 only: train all agents jointly, save episode checkpoints + final model."""
    all_ids = set(_agent_ids(args))
    base_name = args.run_name or args.exp_name

    phase1 = copy.deepcopy(args)
    phase1.run_name = f"{base_name}_checkpoint"
    phase1.use_opponent_pool = False
    phase1.agent_configs = _with_train_flags(phase1, all_ids)
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
    run_ippo_training_job(phase1, all_ids)


def run_ippo_bootstrap_direct(args: IppoPoolArgs) -> None:
    """Phase 2 only: continue training from Phase 1 checkpoint, save final direct-IPPO model."""
    all_ids = set(_agent_ids(args))
    base_name = args.run_name or args.exp_name

    phase2 = copy.deepcopy(args)
    phase2.run_name = f"{base_name}_direct"
    phase2.use_opponent_pool = False
    phase2.agent_configs = _with_train_flags(phase2, all_ids)
    phase2.ppo_model_paths = [None for _ in phase2.agent_configs]
    phase2.resume_from = args.bootstrap_save_model_path
    phase2.load_model_path = None
    phase2.total_timesteps = int(args.pool_bootstrap_checkpoint_timesteps + args.pool_bootstrap_extra_timesteps)
    phase2.save_checkpoint = False
    phase2.save_model_path = args.bootstrap_final_save_model_path
    final_ids = {int(agent_id) for agent_id in args.pool_final_save_agent_ids}
    print(
        "[Bootstrap] stage 2: "
        f"continue to {phase2.total_timesteps} total steps, final ids={sorted(final_ids)}"
    )
    run_ippo_training_job(phase2, final_ids)


def run_ippo_bootstrap(args: IppoPoolArgs) -> None:
    """Convenience: run both Phase 1 and Phase 2 sequentially."""
    run_ippo_bootstrap_checkpoint(args)
    run_ippo_bootstrap_direct(args)


_AGENT_RE_TEMPLATE = r"_agent{agent_id}(?:\.pt|$)"
_EPISODE_RE = re.compile(r"_episode(\d+)")
_STEP_RE = re.compile(r"_step(\d+)")


def _checkpoint_sort_key(path: pathlib.Path) -> tuple[int, float, str]:
    """
    生成检查点文件的排序键，用于对检查点文件进行排序。
    
    排序优先级：
    1. episode/step编号（主要排序依据）
    2. 文件修改时间（次要排序依据，处理相同编号的情况）
    3. 文件名（最终排序依据，确保确定性）
    Returns:
        三元组 (episode/step编号, 修改时间戳, 文件名)
    """
    name = path.name
    # 从文件名中提取episode或step编号
    # 优先匹配_episodeN，如果没有则匹配_stepN
    match = _EPISODE_RE.search(name) or _STEP_RE.search(name)
    # 如果找到匹配，提取数字；否则默认为0
    number = int(match.group(1)) if match else 0
    try:
        # 获取文件的最后修改时间戳
        mtime = path.stat().st_mtime
    except OSError:
        # 如果无法获取文件状态（如文件被删除），使用默认值0.0
        mtime = 0.0
    # 返回排序键：先按编号排序，再按时间排序，最后按文件名排序
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
    #查找所有匹配的检查点文件
    candidates = [
        path for path in root.rglob("*.pt")
        if agent_re.search(path.name)
    ]
    # 对检查点文件进行排序
    candidates.sort(key=_checkpoint_sort_key)
    # 保留最新k个检查点
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
        reward_ema_coef=args.pool_reward_ema,
        default_reward_score=args.pool_default_reward_score,
        delete_replaced_checkpoints=args.pool_delete_replaced_checkpoints,
    )
    #加载中断点，初始化对手池
    for agent_id in _agent_ids(args):
        #查找最近的检查点文件
        paths = find_recent_agent_checkpoints(
            checkpoint_dir,
            agent_id,
            args.pool_initial_keep_per_agent,
        )
        #将检查点文件添加到对手池中
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
    """更新主智能体的对手池"""
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
    """从对手池中加载最新检查点到智能体"""
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
    """计算阶段更新次数和步进增量"""
    n_agents = len(args.agent_configs)
    if args.count_steps_by == "env_steps":
        denom = args.num_game_envs * args.num_steps#每次采样后的总样本数
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
    """从对手池中采样并加载对手智能体的模型"""
    opponent_ids = [agent_id for agent_id in _agent_ids(ctx.args) if agent_id != main_agent_id]
    #从对手池中采样一个对手智能体组
    slot_index, group, probs = pool.sample_group(main_agent_id, opponent_ids, rng)
    #加载采样的对手智能体组的模型
    _load_opponent_group(group, ctx)
    opponent_desc = " ".join(f"agent_{entry.agent_id}[{entry.slot_index}]" for entry in group)
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
    """对手池迭代轮回训练"""
    args = ctx.args
    n_agents = len(args.agent_configs)
    n_game_envs = args.num_game_envs

    _restore_agent_state(ctx.agents[main_agent_id], current_agent_states[main_agent_id], ctx.device)
    #划分训练智能体和非训练智能体
    args.agent_configs = _with_train_flags(
        args,
        train_ids={main_agent_id},
        policy_opponent_ids=set(_agent_ids(args)) - {main_agent_id},
    )

    phase_updates, step_increment = _phase_update_count(args, phase_timesteps)
    #每个智能体在不同并行环境中的累计奖励不同
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, ctx.device) for agent in ctx.agents]
    _, current_group = _sample_and_load_opponents(pool, main_agent_id, ctx, rng)

    start_time = time.time()
    #记录上次保存检查点的步数
    last_pool_save_step = ctx.global_step

    for local_update in range(1, phase_updates + 1):
        if args.anneal_lr:#退火学习率
            progress = 1.0 - (local_update - 1.0) / phase_updates
            cfg = args.agent_configs[main_agent_id]
            ctx.optimizers[main_agent_id].param_groups[0]["lr"] = progress * cfg.learning_rate

        #收集数据
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

        completed_returns = new_episode_returns[main_agent_id]#收集数据时所有回合的奖励
        if completed_returns:#如果经历过回合结束
            mean_reward = float(np.mean(np.asarray(completed_returns, dtype=np.float64)))
            pool.record_result(
                main_agent_id,
                current_group,
                mean_reward,
                n_games=len(completed_returns),
            )
            #重新采样，current_group仅用于统计
            _, current_group = _sample_and_load_opponents(pool, main_agent_id, ctx, rng)
            #重置RNN状态
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
    setup_args.run_name = f"{(args.run_name or args.exp_name)}_pool"
    setup_args.use_opponent_pool = True
    #所有智能体都进行训练
    setup_args.agent_configs = _with_train_flags(setup_args, set(_agent_ids(setup_args)))

    ctx = None
    try:
        ctx = setup_training_context(setup_args)
        if not any(ctx.args.ppo_model_paths) and not ctx.args.resume_from and not ctx.args.load_model_path:
            _load_latest_pool_entries_into_agents(pool, ctx)#用最近的检查点模型
        #所有智能体的网络参数，shape(n_agents)
        current_agent_states: list[dict[str, torch.Tensor]] = _capture_agent_states(ctx.agents)
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

        #结束轮训后对主智能体进行额外训练
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
        #保存主智能体模型
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
    """将指定id的智能体设置为评估模型"""
    cfg = args.agent_configs[agent_id]
    agent = IPPOAgent(n_actions, seg, cfg).to(device)
    _load_agent_state(path, agent_id, agent, device)
    #设置为评估模式
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
    """评估模型"""
    #寻找评估模型的目录
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
    #都进行推理不参与训练
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
        #智能体0的对手
        opponent_ids = [agent_id for agent_id in _agent_ids(eval_args) if agent_id != 0]
        group_count = pool.group_count(opponent_ids)
        if group_count >= int(eval_args.pool_eval_groups):
            #不放回采样对手组
            slot_indices = rng.sample(range(group_count), int(eval_args.pool_eval_groups))
        else:#放回采样对手组
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
        for model_label, model_path in model_paths.items():#待评估的模型
            if model_path is None:
                continue
            #参与评估的所有模型策略
            policies: list[list[IPPOAgent]] = []
            #对于采样到的每个对手组
            for group in opponent_groups:
                #组内智能体
                group_agents: list[Optional[IPPOAgent]] = [None for _ in _agent_ids(eval_args)]
                #智能体0为待评估模型
                group_agents[0] = _build_eval_agent(model_path, 0, eval_args, n_actions, seg, device)
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
    """
    评估多个智能体组的性能
    
    Args:
        model_label: 模型标签，用于标识被评估的模型类型（如'direct_ippo'或'opponent_pool'）
        policies: 策略列表，每个元素是一个智能体组，包含该组中所有智能体的策略网络
        args: IPPO池化训练的参数配置对象
        envs: 向量化环境实例，支持并行执行多个episode
        device: PyTorch设备（CPU/GPU）
    
    Returns:
        包含评估结果的字典列表，每个字典记录一个episode的奖励信息
    """
    n_groups = len(policies)  # 智能体组的数量（即并行评估的episode数）
    n_agents = len(args.agent_configs)  # 每组中智能体的数量
    
    # 重置环境，获取初始观测值
    obs_raw, _ = envs.reset(seed=args.seed)
    next_obs = np.asarray(obs_raw, dtype=np.float32)  # 转换为numpy数组并指定数据类型
    
    # 初始化奖励累加器和episode计数器
    episode_rewards = np.zeros((n_groups, n_agents), dtype=np.float64)  # 记录每组每个智能体的累计奖励
    episode_counts = np.zeros(n_groups, dtype=np.int64)  # 记录每组已完成的episode数量
    
    # 初始化RNN隐藏状态（如果智能体使用循环神经网络）
    rnn_states: list[list[Optional[torch.Tensor]]] = []
    for group in policies:
        # 为每个智能体创建初始RNN状态，如果不是循环网络则为None
        rnn_states.append([
            agent.get_initial_state(1, device) if agent.is_recurrent else None
            for agent in group
        ])

    rows: list[dict[str, Any]] = []  # 存储评估结果的列表
    
    # 主评估循环：直到所有组都完成指定的episode数量
    while np.any(episode_counts < args.pool_eval_episodes_per_group):
        # 将观测值转换为PyTorch张量并重塑为(group, agent, feature)格式
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        obs_by_env = obs_t.view(n_groups, n_agents, -1)  # 重塑为[n_groups, n_agents, obs_dim]
        
        # 初始化动作数组
        actions_by_env = np.zeros((n_groups, n_agents), dtype=np.int64)
        
        # 对每个组和每个智能体进行动作选择
        for group_idx in range(n_groups):
            for agent_idx in range(n_agents):
                # 根据当前策略选择动作
                action, next_state = _select_action(
                    policies[group_idx][agent_idx],  # 当前智能体的策略
                    obs_by_env[group_idx, agent_idx].unsqueeze(0),  # 添加batch维度
                    rnn_states[group_idx][agent_idx],  # 当前RNN状态
                    args.eval_deterministic,  # 是否采用确定性策略
                )
                actions_by_env[group_idx, agent_idx] = action  # 保存选择的动作
                rnn_states[group_idx][agent_idx] = next_state  # 更新RNN状态

        # 执行动作并获取环境反馈
        next_obs, rewards, terms, truncs, _ = envs.step(actions_by_env.reshape(-1))
        next_obs = np.asarray(next_obs, dtype=np.float32)  # 转换新观测值
        
        # 处理奖励和终止信号
        rewards_by_env = np.asarray(rewards, dtype=np.float32).reshape(n_groups, n_agents)
        dones_by_env = np.logical_or(terms, truncs).reshape(n_groups, n_agents)  # 合并终止和截断信号
        episode_rewards += rewards_by_env  # 累加奖励

        # 检查每个组是否有episode结束
        for group_idx in range(n_groups):
            # 如果该组已达到目标episode数，则跳过
            if episode_counts[group_idx] >= args.pool_eval_episodes_per_group:
                continue
            
            # 如果该组中有任何智能体完成episode
            if np.any(dones_by_env[group_idx]):
                episode = int(episode_counts[group_idx])  # 当前episode编号
                
                # 为该组中每个智能体记录结果
                for agent_idx in range(n_agents):
                    rows.append(
                        {
                            "model_label": model_label,  # 模型标签
                            "group_id": group_idx,  # 组ID
                            "episode": episode,  # episode编号
                            "agent_id": args.agent_configs[agent_idx].agent_id,  # 智能体ID
                            "reward": float(episode_rewards[group_idx, agent_idx]),  # 累计奖励
                        }
                    )
                
                # 更新计数器并重置该组的奖励和RNN状态
                episode_counts[group_idx] += 1
                episode_rewards[group_idx, :] = 0.0  # 重置奖励累加器
                # 重置RNN状态以开始新的episode
                rnn_states[group_idx] = [
                    agent.get_initial_state(1, device) if agent.is_recurrent else None
                    for agent in policies[group_idx]
                ]
    
    return rows  # 返回所有评估结果


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
    print("[Full Plan] Starting full plan...")
    run_ippo_bootstrap(args)
    pool_args = copy.deepcopy(args)
    pool_args.load_model_path = None
    pool_args.resume_from = None
    pool_args.ppo_model_paths = [None for _ in pool_args.agent_configs]
    # Bootstrap checkpoints are saved as flat files in the parent directory
    # of bootstrap_save_model_path (e.g. saved_models/ippo_bootstrap_episode{N}_agent{id}.pt).
    # Override pool_initial_checkpoint_dir to point to the actual location.
    if args.bootstrap_save_model_path:
        pool_args.pool_initial_checkpoint_dir = str(
            pathlib.Path(args.bootstrap_save_model_path).parent
        )
    run_pool_cycle(pool_args)


def main() -> None:
    args = _parse_cli(IppoPoolArgs())
    if args.run_mode == "ippo_bootstrap":
        run_ippo_bootstrap(args)
    elif args.run_mode == "bootstrap_checkpoint":
        run_ippo_bootstrap_checkpoint(args)
    elif args.run_mode == "bootstrap_direct":
        run_ippo_bootstrap_direct(args)
    elif args.run_mode == "pool_cycle":
        run_pool_cycle(args)
<<<<<<< HEAD
    # 评估方法已经抽离到 pool_evaluate.py
=======
>>>>>>> 397661c67e2ddd0684da121c21254fcb707366c6
    elif args.run_mode == "evaluate":
        run_evaluation(args)
    elif args.run_mode == "full":
        run_full_plan(args)
    else:
        raise ValueError(f"Unknown run_mode={args.run_mode!r}")

if __name__ == "__main__":
    main()
