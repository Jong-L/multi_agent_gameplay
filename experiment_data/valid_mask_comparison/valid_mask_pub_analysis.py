"""
Valid Mask Comparison — Publication-Style Analysis
===================================================
Compares 2 observation mask strategies:
  1. no_valid_mask  — raw missing data (no validity signal)
  2. use_valid_mask — explicit validity mask for missing slot entries

Metrics (preserved from valid_mask_analysis.py):
  - Total score = sum of ALL values
  - Ball score = collect_ball_A + collect_ball_B only

Data processing: per-file → mean ± SEM across 5 runs (real error bands)
Drawing: publication_plot_utils pipeline (Gaussian filter → B-spline → fill_between)
Colors: Wong 2011 colorblind-friendly DEFAULT_PALETTE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data_analyze"))
from publication_plot_utils import (
    setup_style, prepare_curve, plot_with_fill,
    style_axes, save_figure, add_stats_box, DEFAULT_PALETTE
)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "summary"
OUTPUT_DIR.mkdir(exist_ok=True)

MASK_NAMES = {
    'no_valid_mask': 'No Valid Mask',
    'use_valid_mask': 'Use Valid Mask',
}

MASK_COLORS = {
    'no_valid_mask': DEFAULT_PALETTE[1],  # #DE8F05 orange
    'use_valid_mask': DEFAULT_PALETTE[0],  # #0173B2 blue
}

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
PLAYER_IDS = [0, 1, 2, 3]
SMOOTH_WINDOW = 5  # Gaussian sigma = 5 × 0.35 = 1.75


# ============================================================
# Data loading
# ============================================================
def discover_files():
    """Group CSVs by mask type."""
    result = {'no_valid_mask': [], 'use_valid_mask': []}
    for fp in DATA_DIR.glob("*.csv"):
        name = fp.name
        if name.startswith("no_valid_mask"):
            result['no_valid_mask'].append(str(fp))
        elif name.startswith("use_valid_mask"):
            result['use_valid_mask'].append(str(fp))
    for k in result:
        result[k].sort()
    return result


def process_one_file(fp, max_eps):
    """
    Process a single CSV file.
    Returns: dict {ep: {p0_total, p0_ball, p1_total, ..., total, ball}}
    """
    df = pd.read_csv(fp)
    result = {}
    for ep in range(1, max_eps + 1):
        ep_df = df[df['episode_id'] == ep]
        row = {}
        for pid in PLAYER_IDS:
            p_df = ep_df[ep_df['player_id'] == pid]
            row[f'p{pid}_total'] = p_df['value'].sum()
            ball_df = p_df[p_df['source'].isin(BALL_SOURCES)]
            row[f'p{pid}_ball'] = ball_df['value'].sum()
        row['total'] = sum(row[f'p{p}_total'] for p in PLAYER_IDS)
        row['ball']  = sum(row[f'p{p}_ball'] for p in PLAYER_IDS)
        result[ep] = row
    return result


def aggregate_group(file_list, max_eps):
    """
    Process all 5 runs, compute per-episode mean ± SEM.
    Returns: dict with *_mean and *_sem arrays.
    """
    n = len(file_list)
    all_results = [process_one_file(fp, max_eps) for fp in file_list]

    metrics = ['total', 'ball'] + [f'p{p}_total' for p in PLAYER_IDS] + [f'p{p}_ball' for p in PLAYER_IDS]
    arrs = {m: np.zeros((n, max_eps)) for m in metrics}
    for run_i, res in enumerate(all_results):
        for ep in range(max_eps):
            r = res.get(ep + 1, {})
            for m in metrics:
                arrs[m][run_i, ep] = r.get(m, 0.0)

    result = {'episodes': np.arange(1, max_eps + 1), 'num_eps': max_eps}
    for m in metrics:
        result[f'{m}_mean'] = arrs[m].mean(axis=0)
        result[f'{m}_sem'] = arrs[m].std(axis=0, ddof=1) / np.sqrt(n)
    return result


# ============================================================
# Plotting
# ============================================================
MASK_ORDER = ['no_valid_mask', 'use_valid_mask']

def plot_player_total(group_stats):
    """4 subplots: per-player total score."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    for idx, pid in enumerate(PLAYER_IDS):
        ax = axes[idx]; col = f'p{pid}_total'
        for mask in MASK_ORDER:
            if mask not in group_stats: continue
            gs = group_stats[mask]; x = gs['episodes']
            xs, ys, yl, yu = prepare_curve(x, gs[f'{col}_mean'], gs[f'{col}_sem'], SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu, color=MASK_COLORS[mask], label=MASK_NAMES[mask])
        style_axes(ax, xlabel='Episode', ylabel='Average Total Score',
                   title=f'Player {pid}',
                   legend_kwargs={'loc': 'upper left', 'fontsize': 9})
    fig.suptitle('Player Average Total Score vs Episode (Valid Mask Comparison)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "valid_mask_comparison_player_total_score.png")


def plot_total_total(group_stats):
    """1 plot: sum of all players total score."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = ["Mean Total Score:"]
    for mask in MASK_ORDER:
        if mask not in group_stats: continue
        gs = group_stats[mask]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs['total_mean'], gs['total_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=MASK_COLORS[mask], label=MASK_NAMES[mask])
        lines.append(f"  {MASK_NAMES[mask]}: {gs['total_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='Episode', ylabel='Total Score (Sum of 4 Players)',
               title='Total Score Comparison (Valid Mask)',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': 'Mask Type'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "valid_mask_comparison_total_score.png")


def plot_player_ball(group_stats):
    """4 subplots: per-player ball score."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    for idx, pid in enumerate(PLAYER_IDS):
        ax = axes[idx]; col = f'p{pid}_ball'
        for mask in MASK_ORDER:
            if mask not in group_stats: continue
            gs = group_stats[mask]; x = gs['episodes']
            xs, ys, yl, yu = prepare_curve(x, gs[f'{col}_mean'], gs[f'{col}_sem'], SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu, color=MASK_COLORS[mask], label=MASK_NAMES[mask])
        style_axes(ax, xlabel='Episode', ylabel='Average Ball Score',
                   title=f'Player {pid}',
                   legend_kwargs={'loc': 'upper left', 'fontsize': 9})
        ax.set_ylim(bottom=0)
    fig.suptitle('Player Average Ball Score vs Episode (Valid Mask Comparison)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "valid_mask_comparison_player_ball_score.png")


def plot_total_ball(group_stats):
    """1 plot: sum of all players ball score."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = ["Mean Ball Score:"]
    for mask in MASK_ORDER:
        if mask not in group_stats: continue
        gs = group_stats[mask]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs['ball_mean'], gs['ball_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=MASK_COLORS[mask], label=MASK_NAMES[mask])
        lines.append(f"  {MASK_NAMES[mask]}: {gs['ball_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='Episode', ylabel='Total Ball Score (Sum of 4 Players)',
               title='Ball Collection Score Comparison (Valid Mask)',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': 'Mask Type'})
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "valid_mask_comparison_total_ball_score.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Valid Mask Comparison — Publication-Style Analysis")
    print("=" * 70)
    setup_style()

    # 1. Discover
    print("\n[1/3] Discovering files...")
    data_files = discover_files()
    max_ep = None
    for mask, files in data_files.items():
        print(f"  {MASK_NAMES[mask]}: {len(files)} files")
        for fp in files:
            df = pd.read_csv(fp, usecols=['episode_id'])
            me = int(df['episode_id'].max())
            if max_ep is None or me < max_ep:
                max_ep = me
    print(f"  Max episodes: {max_ep}")

    # 2. Process
    print("\n[2/3] Processing per-file → mean ± SEM...")
    group_stats = {}
    for mask in MASK_ORDER:
        if len(data_files[mask]) == 0:
            print(f"  SKIP {MASK_NAMES[mask]}: no files")
            continue
        group_stats[mask] = aggregate_group(data_files[mask], max_ep)
        print(f"  {MASK_NAMES[mask]}: total={group_stats[mask]['total_mean'].mean():.1f}, "
              f"ball={group_stats[mask]['ball_mean'].mean():.1f}")

    # 3. Save CSVs + plots
    print("\n[3/3] Saving CSVs and generating plots...")
    for mask, gs in group_stats.items():
        rows = [{'episode_id': ep + 1,
                 'total_score': gs['total_mean'][ep], 'total_sem': gs['total_sem'][ep],
                 'ball_score': gs['ball_mean'][ep], 'ball_sem': gs['ball_sem'][ep]}
                for ep in range(gs['num_eps'])]
        for pid in PLAYER_IDS:
            for ep in range(gs['num_eps']):
                rows[ep][f'p{pid}_total'] = gs[f'p{pid}_total_mean'][ep]
                rows[ep][f'p{pid}_ball'] = gs[f'p{pid}_ball_mean'][ep]
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"valid_mask_comparison_{mask}_stats.csv", index=False)

    # Overall summary
    summary_rows = []
    for mask in MASK_ORDER:
        if mask not in group_stats: continue
        gs = group_stats[mask]
        row = {'mask_type': mask, 'mask_name': MASK_NAMES[mask],
               'num_episodes': gs['num_eps'],
               'mean_total_score': gs['total_mean'].mean(),
               'mean_ball_score': gs['ball_mean'].mean()}
        for pid in PLAYER_IDS:
            row[f'p{pid}_mean_total'] = gs[f'p{pid}_total_mean'].mean()
            row[f'p{pid}_mean_ball'] = gs[f'p{pid}_ball_mean'].mean()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "valid_mask_comparison_summary.csv", index=False)

    plot_player_total(group_stats)
    plot_total_total(group_stats)
    plot_player_ball(group_stats)
    plot_total_ball(group_stats)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mask in MASK_ORDER:
        if mask not in group_stats: continue
        gs = group_stats[mask]
        print(f"\n{MASK_NAMES[mask]} ({gs['num_eps']} episodes):")
        print(f"  Mean Total Score: {gs['total_mean'].mean():.1f}")
        print(f"  Mean Ball Score:  {gs['ball_mean'].mean():.1f}")
        for pid in PLAYER_IDS:
            print(f"    Player {pid}: Total={gs[f'p{pid}_total_mean'].mean():.1f}, "
                  f"Ball={gs[f'p{pid}_ball_mean'].mean():.1f}")

    print(f"\n{'=' * 70}")
    print(f"Done. Output: {OUTPUT_DIR}")
    print(f"{len(list(OUTPUT_DIR.glob('valid_mask_comparison_*.png')))} PNGs + "
          f"{len(list(OUTPUT_DIR.glob('valid_mask_comparison_*.csv')))} CSVs")
    print("=" * 70)


if __name__ == "__main__":
    main()