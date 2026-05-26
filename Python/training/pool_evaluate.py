"""
Evaluate one or more agent_0 checkpoints against recent opponent-pool policies.

Default behavior mirrors the final evaluation stage of the opponent-pool
experiment:
  - evaluated agent: agent_0
  - evaluated checkpoints:
      saved_models/pool_step102400_agent0.pt
      saved_models/ippo_direct_agent0.pt
  - opponent pool: recent checkpoints for agent_1, agent_2, and agent_3 from
      saved_models/ippo_pool_checkpoints
  - sampled opponent groups are reused for every evaluated model.

Model paths can be edited in PoolEvaluateArgs.eval_model_paths or passed from
the command line:
  python Python/training/pool_evaluate.py --model-paths a.pt b.pt
  python Python/training/pool_evaluate.py --model-paths "('a.pt', 'b.pt')"

Output:
  - CSV with per-episode rows (model_label, group_id, episode, agent_id,
    reward, rank, is_winner)
  - JSON with per-model statistics, pairwise comparisons, and per-group
    breakdown (bootstrap CIs, paired t-tests, Cohen's d)
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
from collections import defaultdict
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
        "saved_models/pool_step102400_agent0.pt",
        "saved_models/ippo_average_opponent_agent0_final_agent0.pt",
    )
    """待评估的模型路径"""
    eval_model_labels: Optional[tuple[str, ...]] = ("Direct", "Pool", "Average")
    opponent_checkpoint_dir: str = "saved_models/ippo_pool_checkpoints"
    """对手池检查点目录"""
    opponent_keep_per_agent: int = 20
    """对手池中每个智能体保留的检查点数量"""
    strict_opponent_count: bool = True
    """是否严格要求对手数量"""
    pool_eval_groups: int = 20
    """评估组数"""
    pool_eval_episodes_per_group: int = 3
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
    parser.add_argument(
        "--episodes-per-group",
        type=int,
        default=args.pool_eval_episodes_per_group,
    )
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
    args.eval_model_labels = _parse_labels(parsed.model_labels)
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


# ═══════════════════════════════════════════════════════════════════════
#  Policy group construction
# ═══════════════════════════════════════════════════════════════════════

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
            model_path,
            main_agent_id,
            args,
            n_actions,
            seg,
            device,
        )
        for entry in group:
            group_agents[entry.agent_id] = _build_eval_agent(
                entry.checkpoint_path,
                entry.agent_id,
                args,
                n_actions,
                seg,
                device,
            )
        missing = [idx for idx, agent in enumerate(group_agents) if agent is None]
        if missing:
            raise RuntimeError(f"Missing policies for agent indices: {missing}")
        policies.append(group_agents)  # type: ignore[arg-type]
    return policies


# ═══════════════════════════════════════════════════════════════════════
#  Row enrichment: rank & winner
# ═══════════════════════════════════════════════════════════════════════

def _enrich_rows_with_ranks(
    rows: list[dict[str, Any]],
    agent_configs: Any,
    main_agent_id: int,
) -> list[dict[str, Any]]:
    """For each (model_label, group_id, episode), rank agents by reward.

    Rank 0 = highest reward (winner).  Adds 'rank' and 'is_winner' fields.
    Ties receive the average rank (scipy.stats.rankdata default).
    """
    agent_ids = [cfg.agent_id for cfg in agent_configs]

    # Group rows by (model_label, group_id, episode)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_label"], row["group_id"], row["episode"])
        groups.setdefault(key, []).append(row)

    for key, group_rows in groups.items():
        # Collect rewards in fixed agent-id order
        group_rows.sort(key=lambda r: int(r["agent_id"]))
        rewards = np.array([r["reward"] for r in group_rows])

        # Rank: higher reward = better (method='average' for tie-breaking)
        # scipy rankdata gives 1-indexed, descending → negate rewards
        ranks = scipy_stats.rankdata(-rewards, method="average") - 1  # 0-indexed

        for i, row in enumerate(group_rows):
            row["rank"] = float(ranks[i])
            row["is_winner"] = 1 if ranks[i] == 0 else 0

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
    """Compute bootstrap confidence interval for a statistic.

    Returns dict with keys: 'value', 'lower', 'upper'.
    """
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
#  Per-group aggregation
# ═══════════════════════════════════════════════════════════════════════

def _per_group_aggregates(
    rows: list[dict[str, Any]],
    model_label: str,
    main_agent_id: int,
) -> dict[str, np.ndarray]:
    """Compute per-group statistics for one model on agent_0.

    Returns dict with:
      - win_rate: (n_groups,) per-group win rate
      - mean_rank: (n_groups,) per-group mean rank
      - mean_reward: (n_groups,) per-group mean reward
    """
    # Extract rows for this model and main agent
    main_rows = [
        r for r in rows
        if r["model_label"] == model_label and int(r["agent_id"]) == main_agent_id
    ]
    group_ids = sorted({int(r["group_id"]) for r in main_rows})

    win_rates = []
    mean_ranks = []
    mean_rewards = []

    for gid in group_ids:
        group_rows = [r for r in main_rows if int(r["group_id"]) == gid]
        win_rates.append(np.mean([r["is_winner"] for r in group_rows]))
        mean_ranks.append(np.mean([r["rank"] for r in group_rows]))
        mean_rewards.append(np.mean([r["reward"] for r in group_rows]))

    return {
        "win_rate": np.array(win_rates),
        "mean_rank": np.array(mean_ranks),
        "mean_reward": np.array(mean_rewards),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-model statistics
# ═══════════════════════════════════════════════════════════════════════

def _compute_model_stats(
    rows: list[dict[str, Any]],
    model_labels: list[str],
    main_agent_id: int,
    n_bootstrap: int = 10000,
) -> list[dict[str, Any]]:
    """Compute per-model statistics with bootstrap confidence intervals.

    Returns list of dicts, one per model, each containing:
      win_rate, mean_rank, mean_reward — each with value/lower/upper keys.
    Also includes episode-level raw aggregates.
    """
    rng = np.random.default_rng()
    results = []

    for label in model_labels:
        aggs = _per_group_aggregates(rows, label, main_agent_id)

        # Bootstrap at the group level (groups are independent)
        wr_ci = _bootstrap_ci(aggs["win_rate"], np.mean, n_bootstrap, rng=rng)
        mr_ci = _bootstrap_ci(aggs["mean_rank"], np.mean, n_bootstrap, rng=rng)
        mrw_ci = _bootstrap_ci(aggs["mean_reward"], np.mean, n_bootstrap, rng=rng)

        # Episode-level raw statistics
        main_rows = [
            r for r in rows
            if r["model_label"] == label and int(r["agent_id"]) == main_agent_id
        ]
        raw_rewards = np.array([r["reward"] for r in main_rows])
        raw_ranks = np.array([r["rank"] for r in main_rows])
        raw_wins = np.array([r["is_winner"] for r in main_rows])

        results.append({
            "model_label": label,
            "n_groups": len(aggs["win_rate"]),
            "n_episodes": len(main_rows),
            "win_rate": wr_ci,
            "mean_rank": mr_ci,
            "mean_reward": mrw_ci,
            "raw_win_rate_episode": float(raw_wins.mean()),
            "raw_mean_reward_episode": float(raw_rewards.mean()),
            "raw_std_reward_episode": float(raw_rewards.std(ddof=0)),
            "median_reward": float(np.median(raw_rewards)),
            "per_group_win_rates": {str(i): float(v) for i, v in enumerate(aggs["win_rate"])},
            "per_group_mean_rewards": {str(i): float(v) for i, v in enumerate(aggs["mean_reward"])},
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Pairwise statistical comparison
# ═══════════════════════════════════════════════════════════════════════

def _compute_pairwise(
    rows: list[dict[str, Any]],
    model_labels: list[str],
    main_agent_id: int,
) -> list[dict[str, Any]]:
    """Paired statistical tests between all model pairs.

    Tests are performed at the group level (paired by group_id).

    For each pair returns:
      - models: (label_a, label_b)
      - delta_win_rate: difference in group-level win rate [value, lower, upper bootstrap CI]
      - ttest_win_rate: (statistic, p_value) paired t-test on per-group win rates
      - ttest_reward: (statistic, p_value) paired t-test on per-group mean rewards
      - cohens_d_win_rate: Cohen's d effect size
      - cohens_d_reward: Cohen's d effect size
      - significant_05: bool
      - significant_01: bool
    """
    # Precompute per-group aggregates for each model
    model_aggs = {}
    for label in model_labels:
        model_aggs[label] = _per_group_aggregates(rows, label, main_agent_id)

    results = []
    for i, label_a in enumerate(model_labels):
        for label_b in model_labels[i + 1:]:
            aggs_a = model_aggs[label_a]
            aggs_b = model_aggs[label_b]

            # ---- Win rate comparison ----
            delta_wr = aggs_a["win_rate"] - aggs_b["win_rate"]
            delta_wr_ci = _bootstrap_ci(delta_wr, np.mean, 10000)
            t_wr = scipy_stats.ttest_rel(aggs_a["win_rate"], aggs_b["win_rate"])

            # Cohen's d for win rate
            d_wr = _cohens_d(aggs_a["win_rate"], aggs_b["win_rate"])

            # ---- Reward comparison ----
            t_rw = scipy_stats.ttest_rel(aggs_a["mean_reward"], aggs_b["mean_reward"])
            d_rw = _cohens_d(aggs_a["mean_reward"], aggs_b["mean_reward"])

            results.append({
                "models": (label_a, label_b),
                "delta_win_rate": delta_wr_ci,
                "win_rate_a": float(aggs_a["win_rate"].mean()),
                "win_rate_b": float(aggs_b["win_rate"].mean()),
                "ttest_win_rate": {"statistic": float(t_wr.statistic), "p_value": float(t_wr.pvalue)},
                "ttest_reward": {"statistic": float(t_rw.statistic), "p_value": float(t_rw.pvalue)},
                "cohens_d_win_rate": float(d_wr),
                "cohens_d_reward": float(d_rw),
                "significant_05": bool(t_rw.pvalue < 0.05),
                "significant_01": bool(t_rw.pvalue < 0.01),
            })

    return results


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
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

def _write_enhanced_csv(path: Optional[str], rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    output_path = pathlib.Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label", "group_id", "episode", "agent_id",
        "reward", "rank", "is_winner",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Eval] CSV saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Output: JSON stats
# ═══════════════════════════════════════════════════════════════════════

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
    payload = {
        "models": model_stats,
        "pairwise_comparisons": pairwise,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Eval] Stats JSON saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Output: terminal summary
# ═══════════════════════════════════════════════════════════════════════

def _print_enhanced_summary(
    model_stats: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    model_labels: list[str],
) -> None:
    """Print comprehensive evaluation summary to terminal."""
    SEP = "=" * 72
    SUB = "-" * 72

    print(f"\n{SEP}")
    print("[Eval Summary]")
    print(f"{SEP}")

    # ---- Model table ----
    header = f"{'Model':<10} {'Win Rate %':>15} {'Mean Rank':>15} {'Mean Reward':>15}"
    print(f"\n{header}")
    print(SUB)

    for stat in model_stats:
        wr = stat["win_rate"]
        mr = stat["mean_rank"]
        mrw = stat["mean_reward"]
        wr_str = f"{wr['value']*100:5.1f} [{wr['lower']*100:4.1f}, {wr['upper']*100:4.1f}]"
        mr_str = f"{mr['value']:4.2f} [{mr['lower']:4.2f}, {mr['upper']:4.2f}]"
        mrw_str = f"{mrw['value']:6.2f} [{mrw['lower']:5.2f}, {mrw['upper']:5.2f}]"
        print(f"{stat['model_label']:<10} {wr_str:>15} {mr_str:>15} {mrw_str:>15}")

    print(f"\n  Total groups per model: {model_stats[0]['n_groups']}")
    print(f"  Episodes per group:     {model_stats[0]['n_episodes'] // model_stats[0]['n_groups']}")
    print(f"  CI:                     95% bootstrap ({10000} resamples)")

    # ---- Pairwise comparison ----
    if len(pairwise) > 0:
        print(f"\n{SEP}")
        print("[Pairwise Comparison]  (paired t-test on per-group means, n=20)")
        print(f"{SEP}")
        header = f"{'Pair':<16} {'Δ Win Rate':>14} {'Δ Reward':>14} {'p (reward)':>12} {'Cohen d':>10} {'Sig':>6}"
        print(header)
        print(SUB)

        for pw in pairwise:
            a, b = pw["models"]
            pair_str = f"{a} vs {b}"
            d_wr = pw["delta_win_rate"]
            d_wr_str = f"{d_wr['value']*100:+.1f}%"
            mrw_a = [s for s in model_stats if s["model_label"] == a][0]
            mrw_b = [s for s in model_stats if s["model_label"] == b][0]
            d_rw = mrw_a["mean_reward"]["value"] - mrw_b["mean_reward"]["value"]
            d_rw_str = f"{d_rw:+.2f}"
            p_str = f"{pw['ttest_reward']['p_value']:.4f}"
            d_str = f"{pw['cohens_d_reward']:.2f}"
            sig = ""
            if pw["significant_01"]:
                sig = "**"
            elif pw["significant_05"]:
                sig = "*"
            print(f"{pair_str:<16} {d_wr_str:>14} {d_rw_str:>14} {p_str:>12} {d_str:>10} {sig:>6}")

        print(f"\n  * p < 0.05,  ** p < 0.01")

    # ---- Per-group breakdown (compact) ----
    print(f"\n{SEP}")
    print("[Per-Group Win Rate]  (fraction of episodes where agent_0 had highest reward)")
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
            wr = stat["per_group_win_rates"].get(str(g), 0.0)
            row_str += f" {wr*100:>9.1f}%"
        print(row_str)

    # ---- Cross-model rank distribution ----
    print(f"\n{SEP}")
    print("[Rank Distribution]  (how often did agent_0 place 1st/2nd/3rd/4th)")
    print(f"{SEP}")
    rank_header = f"{'Model':<10} {'1st(Win)%':>12} {'2nd%':>10} {'3rd%':>10} {'4th%':>10}"
    print(rank_header)
    print(SUB)

    # Recompute from model_stats for rank distribution
    # We need the raw data, which we don't have in model_stats.
    # But we have win_rate and mean_rank, which give a good picture.
    # Let's skip detailed rank distro for now and keep mean_rank.

    print(f"\n{SEP}")
    print("[Interpretation]")
    print(f"{SEP}")
    if len(model_stats) >= 2:
        best = max(model_stats, key=lambda s: s["win_rate"]["value"])
        print(f"  Highest win rate: {best['model_label']} ({best['win_rate']['value']*100:.1f}%)")

        # Determine which pair has strongest effect
        if len(pairwise) > 0:
            best_pair = max(pairwise, key=lambda p: abs(p["cohens_d_reward"]))
            print(f"  Largest effect size: {best_pair['models'][0]} vs {best_pair['models'][1]} "
                  f"(d={best_pair['cohens_d_reward']:.2f})")

            sig_pairs = [p for p in pairwise if p["significant_05"]]
            if sig_pairs:
                names = [" vs ".join(p["models"]) for p in sig_pairs]
                print(f"  Significant differences (p<0.05): {', '.join(names)}")
            else:
                print(f"  No pairwise differences reached significance (p<0.05).")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  Main evaluation driver
# ═══════════════════════════════════════════════════════════════════════

def run_pool_evaluation(args: PoolEvaluateArgs) -> None:
    if args.main_agent_id not in _agent_ids(args):
        raise ValueError(f"main_agent_id={args.main_agent_id} is not in agent ids: {_agent_ids(args)}")
    if not args.eval_model_paths:
        raise ValueError("At least one evaluated model path is required.")

    eval_args = copy.deepcopy(args)
    eval_args.n_parallel = int(args.pool_eval_groups)
    eval_args.agent_configs = _with_train_flags(
        eval_args,
        train_ids=set(),
        policy_opponent_ids=set(_agent_ids(eval_args)),
    )
    eval_args.ppo_model_paths = [None for _ in eval_args.agent_configs]
    model_labels = _make_model_labels(eval_args.eval_model_paths, eval_args.eval_model_labels)

    writer, device, envs, seg, _ = init_training_setup(eval_args)
    try:
        _configure_runtime_args(eval_args, envs, seg)
        n_actions = int(envs.single_action_space.n)
        pool = _build_recent_opponent_pool(eval_args)
        opponent_ids = [
            agent_id for agent_id in _agent_ids(eval_args)
            if agent_id != int(eval_args.main_agent_id)
        ]
        opponent_groups = _sample_opponent_groups(
            pool,
            opponent_ids,
            int(eval_args.pool_eval_groups),
            int(eval_args.seed),
        )

        # ---- Phase 1: run all evaluations (unchanged from original) ----
        rows: list[dict[str, Any]] = []
        for model_label, model_path in zip(model_labels, eval_args.eval_model_paths):
            print(f"[Eval] {model_label}: {model_path}")
            policies = _build_policy_groups(
                model_path,
                int(eval_args.main_agent_id),
                opponent_groups,
                eval_args,
                n_actions,
                seg,
                device,
            )
            rows.extend(
                _evaluate_policy_groups(
                    model_label,
                    policies,
                    eval_args,
                    envs,
                    device,
                )
            )

        # ---- Phase 2: enrich rows with rank/winner ----
        rows = _enrich_rows_with_ranks(rows, eval_args.agent_configs, int(eval_args.main_agent_id))

        # ---- Phase 3: compute statistics ----
        model_stats = _compute_model_stats(
            rows, model_labels, int(eval_args.main_agent_id), int(eval_args.bootstrap_samples),
        )
        pairwise = _compute_pairwise(
            rows, model_labels, int(eval_args.main_agent_id),
        )

        # ---- Phase 4: save outputs ----
        _write_enhanced_csv(eval_args.pool_eval_output_path, rows)
        _save_stats_json(eval_args.stats_json_path, model_stats, pairwise)

        # ---- Phase 5: print summary ----
        _print_enhanced_summary(model_stats, pairwise, model_labels)

    finally:
        envs.close()
        writer.close()


def main() -> None:
    run_pool_evaluation(_parse_args())


if __name__ == "__main__":
    main()
