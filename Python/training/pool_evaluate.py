"""
Evaluate one or more agent_0 checkpoints against recent opponent-pool policies.

Since different agents have distinct reward functions (class-specific scoring),
agent_0's own reward is the only valid comparison metric.
All statistics (bootstrap CI, paired t-test, Cohen's d) are computed on
agent_0's per-group mean rewards.

Output:
  - CSV with per-episode rows (model_label, group_id, episode, agent_id, reward)
  - JSON with per-model statistics, pairwise comparisons, per-group breakdown
  - Terminal summary table
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from scipy import stats as scipy_stats

from custom_ppo_dataclass import PoolEntry
from custom_ippo import IPPOAgent
from custom_ippo_pool import (
    IppoPoolArgs,
    OpponentPoolState,
    _agent_ids,
    _build_eval_agent,
    _configure_runtime_args,
    _evaluate_policy_groups,
    _select_action,
    _with_train_flags,
)
from continue_ippo_pool_agent import (
    _checkpoint_sort_key,
    _checkpoint_source,
    find_recent_pool_checkpoints,
)
from godot_env_wrapper import init_training_setup


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PoolEvaluateArgs(IppoPoolArgs):
    """Configuration for standalone opponent-pool evaluation."""

    main_agent_id: int = 0
    eval_model_paths: tuple[str, ...] = (
        "saved_models/ippo_direct_agent0.pt",
        "saved_models/agent0_extra_pool_step8089600_agent0.pt",
        "saved_models/agent0_vs_average_opponents_step8089600_agent0.pt",
        "__random__",
    )
    """待评估的模型路径. '__random__' 表示随机动作基线."""
    eval_model_labels: Optional[tuple[str, ...]] = ("Direct", "Pool", "Average", "Random")
    opponent_checkpoint_dir: str = "saved_models/ippo_pool_checkpoints"
    """对手池检查点目录"""
    opponent_keep_per_agent: int = 20
    """对手池中每个智能体保留的检查点数量"""
    strict_opponent_count: bool = True
    """是否严格要求对手数量"""
    pool_eval_groups: int = 20
    """评估组数"""
    pool_eval_episodes_per_group: int = 5
    """每组评估 episode 数"""
    run_name: Optional[str] = "pool_evaluate"
    """运行名称"""
    pool_eval_output_path: Optional[str] = "logs/ippo_pool_eval.csv"
    """评估结果输出路径 (CSV)"""
    stats_json_path: Optional[str] = field(default=None)
    """统计结果 JSON 路径 (None 则自动从 CSV 路径推导)"""
    bootstrap_samples: int = 10000
    """Bootstrap 重采样次数"""
    ppo_model_paths: list[Optional[str]] = None

    def __post_init__(self) -> None:
        if self.ppo_model_paths is None:
            self.ppo_model_paths = [None for _ in self.agent_configs]
        if self.stats_json_path is None and self.pool_eval_output_path is not None:
            csv_path = pathlib.Path(self.pool_eval_output_path)
            self.stats_json_path = str(csv_path.with_name(csv_path.stem + "_stats.json"))


# ═══════════════════════════════════════════════════════════════════════
#  CLI parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_model_paths(raw_paths: list[str]) -> tuple[str, ...]:
    if len(raw_paths) == 1:
        text = raw_paths[0].strip()
        if text.startswith(("[", "(")):
            value = ast.literal_eval(text)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError("--model-paths literal must be a list/tuple of strings.")
            return tuple(value)
    return tuple(raw_paths)


def _parse_labels(raw_labels: Optional[list[str]]) -> Optional[tuple[str, ...]]:
    if raw_labels is None:
        return None
    if len(raw_labels) == 1:
        text = raw_labels[0].strip()
        if text.startswith(("[", "(")):
            value = ast.literal_eval(text)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError("--model-labels literal must be a list/tuple of strings.")
            return tuple(value)
    return tuple(raw_labels)


def _parse_args() -> PoolEvaluateArgs:
    args = PoolEvaluateArgs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-agent-id", type=int, default=args.main_agent_id)
    parser.add_argument("--model-paths", nargs="+", default=list(args.eval_model_paths))
    parser.add_argument("--model-labels", nargs="+")
    parser.add_argument("--opponent-checkpoint-dir", default=args.opponent_checkpoint_dir)
    parser.add_argument("--opponent-keep-per-agent", type=int, default=args.opponent_keep_per_agent)
    parser.add_argument("--allow-fewer-opponents", action="store_true")
    parser.add_argument("--eval-groups", type=int, default=args.pool_eval_groups)
    parser.add_argument("--episodes-per-group", type=int, default=args.pool_eval_episodes_per_group)
    parser.add_argument("--output-path", default=args.pool_eval_output_path)
    parser.add_argument("--stats-json-path", default=args.stats_json_path)
    parser.add_argument("--bootstrap-samples", type=int, default=args.bootstrap_samples)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--env-path", default=args.env_path)
    parser.add_argument("--config-path", default=args.config_path)
    parser.add_argument("--seed", type=int, default=args.seed)
    parser.add_argument("--port-offset", type=int, default=args.port_offset)
    parser.add_argument("--run-name", default=args.run_name)
    parser.add_argument("--track", action="store_true")
    parsed = parser.parse_args()

    args.main_agent_id = int(parsed.main_agent_id)
    args.eval_model_paths = _parse_model_paths(parsed.model_paths)
    parsed_labels = _parse_labels(parsed.model_labels)
    if parsed_labels is not None:
        args.eval_model_labels = parsed_labels
    args.opponent_checkpoint_dir = parsed.opponent_checkpoint_dir
    args.opponent_keep_per_agent = int(parsed.opponent_keep_per_agent)
    args.strict_opponent_count = not bool(parsed.allow_fewer_opponents)
    args.pool_eval_groups = int(parsed.eval_groups)
    args.pool_eval_episodes_per_group = int(parsed.episodes_per_group)
    args.pool_eval_output_path = parsed.output_path
    args.stats_json_path = parsed.stats_json_path
    args.bootstrap_samples = int(parsed.bootstrap_samples)
    args.eval_deterministic = not bool(parsed.stochastic)
    args.env_path = parsed.env_path
    args.config_path = parsed.config_path
    args.seed = int(parsed.seed)
    args.port_offset = int(parsed.port_offset)
    args.run_name = parsed.run_name
    args.track = bool(parsed.track)
    args.n_parallel = int(args.pool_eval_groups)
    args.use_opponent_pool = False
    args.save_checkpoint = False
    args.resume_from = None
    args.load_model_path = None
    args.ppo_model_paths = [None for _ in args.agent_configs]
    return args


# ═══════════════════════════════════════════════════════════════════════
#  Model label helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_model_labels(
    model_paths: tuple[str, ...],
    labels: Optional[tuple[str, ...]],
) -> list[str]:
    if labels is not None:
        if len(labels) != len(model_paths):
            raise ValueError(
                f"model label count ({len(labels)}) must match model path count ({len(model_paths)})."
            )
        return list(labels)

    seen: dict[str, int] = {}
    result: list[str] = []
    for model_path in model_paths:
        stem = pathlib.Path(model_path).stem
        count = seen.get(stem, 0) + 1
        seen[stem] = count
        result.append(stem if count == 1 else f"{stem}_{count}")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Opponent pool construction
# ═══════════════════════════════════════════════════════════════════════

def _build_recent_opponent_pool(args: PoolEvaluateArgs) -> OpponentPoolState:
    opponent_ids = [
        agent_id for agent_id in _agent_ids(args)
        if agent_id != int(args.main_agent_id)
    ]
    pool = OpponentPoolState(
        agent_ids=opponent_ids,
        per_agent_max_size=max(args.pool_slots_per_agent, args.opponent_keep_per_agent),
        epsilon=args.pool_epsilon,
        temperature=args.pool_pfsp_temperature,
        reward_ema_coef=args.pool_reward_ema,
        default_reward_score=args.pool_default_reward_score,
        delete_replaced_checkpoints=False,
    )

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
                "Pass --allow-fewer-opponents to evaluate with the available complete slots."
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
        print(
            f"[Eval Pool] agent_{agent_id}: loaded {len(paths)} checkpoints "
            f"from {pathlib.Path(paths[0]).name} -> {pathlib.Path(paths[-1]).name}"
        )

    return pool


def _sample_opponent_groups(
    pool: OpponentPoolState,
    opponent_ids: list[int],
    n_groups: int,
    seed: int,
) -> list[list[PoolEntry]]:
    group_count = pool.group_count(opponent_ids)
    if group_count <= 0:
        raise RuntimeError("No complete opponent checkpoint group was found.")

    rng = random.Random(seed)
    if group_count >= n_groups:
        slot_indices = rng.sample(range(group_count), n_groups)
    else:
        slot_indices = [rng.randrange(group_count) for _ in range(n_groups)]

    print(f"[Eval Pool] sampled opponent slots: {slot_indices}")
    return [
        [pool.entries_by_agent[agent_id][slot] for agent_id in opponent_ids]
        for slot in slot_indices
    ]


def _build_policy_groups(
    model_path: str,
    main_agent_id: int,
    opponent_groups: list[list[PoolEntry]],
    args: PoolEvaluateArgs,
    n_actions: int,
    seg: Any,
    device: torch.device,
) -> list[list[IPPOAgent]]:
    policies: list[list[IPPOAgent]] = []
    for group in opponent_groups:
        group_agents: list[Optional[IPPOAgent]] = [None for _ in _agent_ids(args)]
        group_agents[main_agent_id] = _build_eval_agent(
            model_path, main_agent_id, args, n_actions, seg, device,
        )
        for entry in group:
            group_agents[entry.agent_id] = _build_eval_agent(
                entry.checkpoint_path, entry.agent_id, args, n_actions, seg, device,
            )
        missing = [idx for idx, agent in enumerate(group_agents) if agent is None]
        if missing:
            raise RuntimeError(f"Missing policies for agent indices: {missing}")
        policies.append(group_agents)  # type: ignore[arg-type]
    return policies


# ═══════════════════════════════════════════════════════════════════════
#  Random baseline evaluation (agent_0 takes random actions)
# ═══════════════════════════════════════════════════════════════════════

def _evaluate_random_baseline(
    model_label: str,
    opponent_groups: list[list[PoolEntry]],
    eval_args: PoolEvaluateArgs,
    envs: Any,
    n_actions: int,
    seg: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run evaluation with agent_0 taking uniformly random actions.

    Opponent policies (agent_1/2/3) are loaded from opponent_groups as usual.
    This provides a pure random baseline — any trained model should beat it.
    """
    n_groups = len(opponent_groups)
    n_agents = len(eval_args.agent_configs)

    # Build policies for opponents only
    policies: list[list[Any]] = []
    for group in opponent_groups:
        group_agents: list[Any] = [None for _ in _agent_ids(eval_args)]
        for entry in group:
            group_agents[entry.agent_id] = _build_eval_agent(
                entry.checkpoint_path, entry.agent_id, eval_args, n_actions, seg, device,
            )
        policies.append(group_agents)

    # Init RNN states for opponents
    rnn_states: list[list[Optional[Any]]] = []
    for group in policies:
        rnn_states.append([
            group[agent_idx].get_initial_state(1, device)
            if group[agent_idx] is not None and group[agent_idx].is_recurrent else None
            for agent_idx in range(n_agents)
        ])

    obs_raw, _ = envs.reset(seed=eval_args.seed)
    next_obs = np.asarray(obs_raw, dtype=np.float32)
    episode_rewards = np.zeros((n_groups, n_agents), dtype=np.float64)
    episode_counts = np.zeros(n_groups, dtype=np.int64)
    rows: list[dict[str, Any]] = []

    main_id = int(eval_args.main_agent_id)
    while np.any(episode_counts < eval_args.pool_eval_episodes_per_group):
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        obs_by_env = obs_t.view(n_groups, n_agents, -1)
        actions_by_env = np.zeros((n_groups, n_agents), dtype=np.int64)

        for group_idx in range(n_groups):
            for agent_idx in range(n_agents):
                if agent_idx == main_id:
                    # Random baseline: uniform random
                    actions_by_env[group_idx, agent_idx] = np.random.randint(0, n_actions)
                elif policies[group_idx][agent_idx] is not None:
                    action, next_state = _select_action(
                        policies[group_idx][agent_idx],
                        obs_by_env[group_idx, agent_idx].unsqueeze(0),
                        rnn_states[group_idx][agent_idx],
                        eval_args.eval_deterministic,
                    )
                    actions_by_env[group_idx, agent_idx] = action
                    rnn_states[group_idx][agent_idx] = next_state
                else:
                    actions_by_env[group_idx, agent_idx] = np.random.randint(0, n_actions)

        next_obs, rewards, terms, truncs, _ = envs.step(actions_by_env.reshape(-1))
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards_by_env = np.asarray(rewards, dtype=np.float32).reshape(n_groups, n_agents)
        dones_by_env = np.logical_or(terms, truncs).reshape(n_groups, n_agents)
        episode_rewards += rewards_by_env

        for group_idx in range(n_groups):
            if episode_counts[group_idx] >= eval_args.pool_eval_episodes_per_group:
                continue
            if np.any(dones_by_env[group_idx]):
                episode = int(episode_counts[group_idx])
                for agent_idx in range(n_agents):
                    rows.append({
                        "model_label": model_label,
                        "group_id": group_idx,
                        "episode": episode,
                        "agent_id": eval_args.agent_configs[agent_idx].agent_id,
                        "reward": float(episode_rewards[group_idx, agent_idx]),
                    })
                episode_counts[group_idx] += 1
                episode_rewards[group_idx, :] = 0.0
                rnn_states[group_idx] = [
                    policies[group_idx][agent_idx].get_initial_state(1, device)
                    if policies[group_idx][agent_idx] is not None
                       and policies[group_idx][agent_idx].is_recurrent
                    else None
                    for agent_idx in range(n_agents)
                ]

    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════

def _bootstrap_ci(
    data: np.ndarray,
    stat_fn: Any = np.mean,
    n_bootstrap: int = 10000,
    ci: int = 95,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    data = np.asarray(data)
    if rng is None:
        rng = np.random.default_rng()
    stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (100 - ci) / 2
    return {
        "value": float(stat_fn(data)),
        "lower": float(np.percentile(stats, alpha)),
        "upper": float(np.percentile(stats, 100 - alpha)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-group reward aggregation
# ═══════════════════════════════════════════════════════════════════════

def _per_group_rewards(
    rows: list[dict[str, Any]],
    model_label: str,
    main_agent_id: int,
) -> np.ndarray:
    """Return per-group mean rewards (shape: n_groups,) for one model on agent_0."""
    main_rows = [
        r for r in rows
        if r["model_label"] == model_label and int(r["agent_id"]) == main_agent_id
    ]
    group_ids = sorted({int(r["group_id"]) for r in main_rows})
    return np.array([
        np.mean([r["reward"] for r in main_rows if int(r["group_id"]) == gid])
        for gid in group_ids
    ])


# ═══════════════════════════════════════════════════════════════════════
#  Per-model statistics (reward only)
# ═══════════════════════════════════════════════════════════════════════

def _compute_model_stats(
    rows: list[dict[str, Any]],
    model_labels: list[str],
    main_agent_id: int,
    n_bootstrap: int = 10000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng()
    results = []

    for label in model_labels:
        group_rewards = _per_group_rewards(rows, label, main_agent_id)
        reward_ci = _bootstrap_ci(group_rewards, np.mean, n_bootstrap, rng=rng)

        main_rows = [
            r for r in rows
            if r["model_label"] == label and int(r["agent_id"]) == main_agent_id
        ]
        raw_rewards = np.array([r["reward"] for r in main_rows])

        results.append({
            "model_label": label,
            "n_groups": len(group_rewards),
            "n_episodes": len(main_rows),
            "mean_reward": reward_ci,
            "raw_mean_reward_episode": float(raw_rewards.mean()),
            "raw_std_reward_episode": float(raw_rewards.std(ddof=0)),
            "median_reward": float(np.median(raw_rewards)),
            "per_group_mean_rewards": {str(i): float(v) for i, v in enumerate(group_rewards)},
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Pairwise statistical comparison (reward only)
# ═══════════════════════════════════════════════════════════════════════

def _compute_pairwise(
    rows: list[dict[str, Any]],
    model_labels: list[str],
    main_agent_id: int,
) -> list[dict[str, Any]]:
    group_rewards = {
        label: _per_group_rewards(rows, label, main_agent_id)
        for label in model_labels
    }

    results = []
    for i, label_a in enumerate(model_labels):
        for label_b in model_labels[i + 1:]:
            r_a = group_rewards[label_a]
            r_b = group_rewards[label_b]
            delta = r_a - r_b
            delta_ci = _bootstrap_ci(delta, np.mean, 10000)
            t = scipy_stats.ttest_rel(r_a, r_b)
            d = _cohens_d(r_a, r_b)

            results.append({
                "models": (label_a, label_b),
                "delta_reward": delta_ci,
                "reward_a": float(r_a.mean()),
                "reward_b": float(r_b.mean()),
                "ttest_reward": {"statistic": float(t.statistic), "p_value": float(t.pvalue)},
                "cohens_d_reward": float(d),
                "significant_05": bool(t.pvalue < 0.05),
                "significant_01": bool(t.pvalue < 0.01),
            })

    return results


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    if len(diff) < 2:
        return 0.0
    std = np.std(diff, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(diff) / std)


# ═══════════════════════════════════════════════════════════════════════
#  Output: CSV
# ═══════════════════════════════════════════════════════════════════════

def _write_csv(path: Optional[str], rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    output_path = pathlib.Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_label", "group_id", "episode", "agent_id", "reward"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Eval] CSV saved to {output_path}")


def _save_stats_json(
    path: Optional[str],
    model_stats: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> None:
    if path is None:
        return
    output_path = pathlib.Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": model_stats, "pairwise_comparisons": pairwise}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Eval] Stats JSON saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Terminal summary
# ═══════════════════════════════════════════════════════════════════════

def _print_summary(
    model_stats: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    model_labels: list[str],
) -> None:
    SEP = "=" * 72
    SUB = "-" * 72

    print(f"\n{SEP}")
    print("[Eval Summary]  (agent_0 reward only — different agents have different reward functions)")
    print(f"{SEP}")

    # ---- Reward table ----
    header = f"{'Model':<10} {'Mean Reward [95% CI]':>32} {'Std':>10}"
    print(f"\n{header}")
    print(SUB)

    for stat in model_stats:
        mrw = stat["mean_reward"]
        rw_str = f"{mrw['value']:7.1f} [{mrw['lower']:6.1f}, {mrw['upper']:6.1f}]"
        std_str = f"{stat['raw_std_reward_episode']:7.1f}"
        print(f"{stat['model_label']:<10} {rw_str:>32} {std_str:>10}")

    print(f"\n  Groups: {model_stats[0]['n_groups']}  |  Episodes/group: {model_stats[0]['n_episodes'] // model_stats[0]['n_groups']}")
    print(f"  CI:     95% bootstrap ({10000} resamples at group level)")

    # ---- Pairwise ----
    if len(pairwise) > 0:
        print(f"\n{SEP}")
        print("[Pairwise Comparison]  (paired t-test on per-group mean rewards, n=20)")
        print(f"{SEP}")
        header = f"{'Pair':<16} {'Δ Reward [95% CI]':>28} {'p':>10} {'Cohen d':>10} {'Sig':>6}"
        print(header)
        print(SUB)

        for pw in pairwise:
            a, b = pw["models"]
            pair_str = f"{a} vs {b}"
            dr = pw["delta_reward"]
            dr_str = f"{dr['value']:+.1f} [{dr['lower']:+.1f}, {dr['upper']:+.1f}]"
            p_str = f"{pw['ttest_reward']['p_value']:.4f}"
            d_str = f"{pw['cohens_d_reward']:.2f}"
            sig = "**" if pw["significant_01"] else ("*" if pw["significant_05"] else "")
            print(f"{pair_str:<16} {dr_str:>28} {p_str:>10} {d_str:>10} {sig:>6}")

        print(f"\n  * p < 0.05,  ** p < 0.01")

    # ---- Per-group rewards ----
    print(f"\n{SEP}")
    print("[Per-Group Mean Rewards]")
    print(f"{SEP}")

    group_header = f"{'Group':<7}"
    for label in model_labels:
        group_header += f" {label:>10}"
    print(group_header)
    print(SUB)

    n_groups = model_stats[0]["n_groups"]
    for g in range(n_groups):
        row_str = f"{g:<7}"
        for stat in model_stats:
            rw = stat["per_group_mean_rewards"].get(str(g), 0.0)
            row_str += f" {rw:>10.1f}"
        print(row_str)

    # ---- Interpretation ----
    print(f"\n{SEP}")
    print("[Interpretation]")
    print(f"{SEP}")
    if len(model_stats) >= 2:
        best = max(model_stats, key=lambda s: s["mean_reward"]["value"])
        print(f"  Highest mean reward: {best['model_label']} ({best['mean_reward']['value']:.1f})")
        most_stable = min(model_stats, key=lambda s: s["raw_std_reward_episode"])
        print(f"  Lowest reward std:   {most_stable['model_label']} ({most_stable['raw_std_reward_episode']:.1f})")

        if len(pairwise) > 0:
            best_pair = max(pairwise, key=lambda p: abs(p["cohens_d_reward"]))
            print(f"  Largest effect size: {best_pair['models'][0]} vs {best_pair['models'][1]} "
                  f"(d={best_pair['cohens_d_reward']:.2f})")
            sig_pairs = [p for p in pairwise if p["significant_05"]]
            if sig_pairs:
                names = [" vs ".join(p["models"]) for p in sig_pairs]
                print(f"  Significant (p<0.05): {', '.join(names)}")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  Main evaluation driver
# ═══════════════════════════════════════════════════════════════════════

def run_pool_evaluation(args: PoolEvaluateArgs) -> None:
    if args.main_agent_id not in _agent_ids(args):
        raise ValueError(f"main_agent_id={args.main_agent_id} not in {_agent_ids(args)}")
    if not args.eval_model_paths:
        raise ValueError("At least one evaluated model path is required.")

    eval_args = copy.deepcopy(args)
    eval_args.n_parallel = int(args.pool_eval_groups)
    eval_args.agent_configs = _with_train_flags(
        eval_args, train_ids=set(), policy_opponent_ids=set(_agent_ids(eval_args)),
    )
    eval_args.ppo_model_paths = [None for _ in eval_args.agent_configs]
    model_labels = _make_model_labels(eval_args.eval_model_paths, eval_args.eval_model_labels)

    writer, device, envs, seg, _ = init_training_setup(eval_args)
    try:
        _configure_runtime_args(eval_args, envs, seg)
        n_actions = int(envs.single_action_space.n)
        pool = _build_recent_opponent_pool(eval_args)
        opponent_ids = [
            aid for aid in _agent_ids(eval_args) if aid != int(eval_args.main_agent_id)
        ]
        opponent_groups = _sample_opponent_groups(
            pool, opponent_ids, int(eval_args.pool_eval_groups), int(eval_args.seed),
        )

        # Phase 1: run all evaluations
        rows: list[dict[str, Any]] = []
        for model_label, model_path in zip(model_labels, eval_args.eval_model_paths):
            print(f"[Eval] {model_label}: {model_path}")
            if model_path == "__random__":
                rows.extend(
                    _evaluate_random_baseline(
                        model_label, opponent_groups, eval_args, envs,
                        n_actions, seg, device,
                    )
                )
            else:
                policies = _build_policy_groups(
                    model_path, int(eval_args.main_agent_id), opponent_groups,
                    eval_args, n_actions, seg, device,
                )
                rows.extend(
                    _evaluate_policy_groups(model_label, policies, eval_args, envs, device)
                )

        # Phase 2: compute statistics
        model_stats = _compute_model_stats(
            rows, model_labels, int(eval_args.main_agent_id), int(eval_args.bootstrap_samples),
        )
        pairwise = _compute_pairwise(rows, model_labels, int(eval_args.main_agent_id))

        # Phase 3: save outputs
        _write_csv(eval_args.pool_eval_output_path, rows)
        _save_stats_json(eval_args.stats_json_path, model_stats, pairwise)

        # Phase 4: print summary
        _print_summary(model_stats, pairwise, model_labels)

    finally:
        envs.close()
        writer.close()


def main() -> None:
    run_pool_evaluation(_parse_args())


if __name__ == "__main__":
    main()
