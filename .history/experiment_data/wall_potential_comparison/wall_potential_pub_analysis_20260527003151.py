"""
Wall Potential Comparison — Publication-Style Analysis
======================================================
Compares 4 wall avoidance strategies:
  1. no_shaping  — Sparse penalty (0.5) only
  2. linear      — Linear potential function (lrrs)
  3. inverse     — Inverse proportional potential (invprs)
  4. distance    — Dense penalty based on distance to wall

Data processing: identical to wall_shaping_comparison.py
  - Total score = sum of ALL values per player per episode
  - Wall collision = count of wall_collision events only
  - Per-file processing → mean ± SEM across 5 runs

Drawing: publication_plot_utils pipeline
  Gaussian filter → B-spline spline → fill_between error bands
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
# 全局中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Add data_analyze to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data_analyze"))
from publication_plot_utils import (
    setup_style, prepare_curve, plot_with_fill,
    style_axes, save_figure, DEFAULT_PALETTE
)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "summary"
OUTPUT_DIR.mkdir(exist_ok=True)

TYPE_NAMES = {
    'no_shaping': '稀疏惩罚(0.5)',
    'linear':     '线性势函数',
    'inverse':    '反比势函数',
    'distance':   '距离惩罚',
}

TYPE_COLORS = {
    'no_shaping': DEFAULT_PALETTE[0],  # #0173B2 blue
    'linear':     DEFAULT_PALETTE[1],  # #DE8F05 orange
    'inverse':    DEFAULT_PALETTE[2],  # #029E73 green
    'distance':   DEFAULT_PALETTE[3],  # #CC78BC pink
}

PLAYERS = [0, 1, 2, 3]
N_RUNS = 5
SMOOTH_WINDOW = 5  # Gaussian sigma = 5 × 0.35 = 1.75


# ============================================================
# Data loading
# ============================================================
def load_data_files():
    """Group CSV files by wall shaping type."""
    data_files = {
        'no_shaping': [],
        'linear': [],
        'inverse': [],
        'distance': [],
    }
    for fp in DATA_DIR.glob("*.csv"):
        fname = fp.name
        if "LiDAR_no_shaping" in fname:
            data_files['no_shaping'].append(str(fp))
        elif "lrrs_" in fname and "invprs" not in fname:
            data_files['linear'].append(str(fp))
        elif "invprs_" in fname:
            data_files['inverse'].append(str(fp))
        elif "distance_penalty_" in fname:
            data_files['distance'].append(str(fp))
    for k in data_files:
        data_files[k].sort()
    return data_files


# ============================================================
# Data processing (preserving original logic)
# ============================================================
def process_one_file(fp):
    """
    Process a single CSV file.
    Returns:
        result: dict {episode_id: {p0_score, p0_walls, p1_score, ...}}
        max_ep: int
    """
    df = pd.read_csv(fp)
    max_ep = int(df['episode_id'].max())
    result = {}
    for ep in range(1, max_ep + 1):
        ep_df = df[df['episode_id'] == ep]
        row = {}
        for pid in PLAYERS:
            p_df = ep_df[ep_df['player_id'] == pid]
            # Total score = sum of ALL values
            total_score = p_df['value'].sum()
            # Wall collision = count of wall_collision events only
            wall_count = len(p_df[p_df['source'] == 'wall_collision'])
            row[f'p{pid}_score'] = total_score
            row[f'p{pid}_walls'] = wall_count
        row['total_score'] = row['p0_score'] + row['p1_score'] + row['p2_score'] + row['p3_score']
        row['total_walls'] = row['p0_walls'] + row['p1_walls'] + row['p2_walls'] + row['p3_walls']
        result[ep] = row
    return result, max_ep


def aggregate_group(file_list, group_name):
    """
    Process all 5 runs in a group, compute per-episode mean ± SEM.
    Returns: dict with per-player and total arrays indexed by episode.
    """
    n = len(file_list)
    print(f"  Processing {TYPE_NAMES[group_name]} ({n} files)...")

    # Collect per-file results
    all_file_results = []
    max_eps = None
    for fp in file_list:
        res, mep = process_one_file(fp)
        all_file_results.append(res)
        if max_eps is None or mep > max_eps:
            max_eps = mep

    # Build per-episode arrays [n_runs, n_eps] for each metric
    metrics = {}
    for pid in PLAYERS:
        metrics[f'p{pid}_score'] = np.zeros((n, max_eps))
        metrics[f'p{pid}_walls'] = np.zeros((n, max_eps))
    metrics['total_score'] = np.zeros((n, max_eps))
    metrics['total_walls'] = np.zeros((n, max_eps))

    for run_i, res in enumerate(all_file_results):
        for ep in range(1, max_eps + 1):
            r = res.get(ep, {})
            for key in metrics:
                metrics[key][run_i, ep - 1] = r.get(key, 0.0)

    # Compute mean ± SEM
    result = {}
    for key, arr in metrics.items():
        result[f'{key}_mean'] = arr.mean(axis=0)
        result[f'{key}_sem'] = arr.std(axis=0, ddof=1) / np.sqrt(n)

    result['episodes'] = np.arange(1, max_eps + 1)
    result['num_eps'] = max_eps
    return result


# ============================================================
# Plotting
# ============================================================
def plot_player_scores(group_stats):
    """4 subplots: each player's score mean ± SEM across 4 shaping types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    for idx, pid in enumerate(PLAYERS):
        ax = axes[idx]
        col = f'p{pid}_score'
        for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
            if type_name not in group_stats:
                continue
            gs = group_stats[type_name]
            x = gs['episodes']
            y = gs[f'{col}_mean']
            e = gs[f'{col}_sem']
            xs, ys, yl, yu = prepare_curve(x, y, e, smooth_window=SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu,
                          color=TYPE_COLORS[type_name],
                          label=TYPE_NAMES[type_name])
        style_axes(ax, xlabel='回合', ylabel='平均总得分',
                   title=f'智能体 {pid}',
                   legend_kwargs={'loc': 'upper left', 'fontsize': 8})

    fig.suptitle('各智能体平均总得分 vs 回合（墙壁势函数对比）', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "wall_potential_comparison_player_score.png")


def plot_total_scores(group_stats):
    """1 plot: total score mean ± SEM across 4 shaping types."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    stats_lines = ["Mean Total Score:"]
    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if type_name not in group_stats:
            continue
        gs = group_stats[type_name]
        x = gs['episodes']
        y = gs['total_score_mean']
        e = gs['total_score_sem']
        xs, ys, yl, yu = prepare_curve(x, y, e, smooth_window=SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu,
                      color=TYPE_COLORS[type_name],
                      label=TYPE_NAMES[type_name])
        stats_lines.append(f"  {TYPE_NAMES[type_name]}: {y.mean():.1f}")

    # Stats box (upper-left to not overlap legend on upper-right)
    from publication_plot_utils import add_stats_box
    add_stats_box(ax, "\n".join(stats_lines), loc='upper left', fontsize=9)

    style_axes(ax, xlabel='Episode', ylabel='Total Average Score',
               title='Total Average Score vs Episode (Wall Shaping Comparison)',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10,
                              'title': 'Wall Shaping Method'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "wall_potential_comparison_total_score.png")


def plot_player_walls(group_stats):
    """4 subplots: each player's wall collision count mean ± SEM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    for idx, pid in enumerate(PLAYERS):
        ax = axes[idx]
        col = f'p{pid}_walls'
        for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
            if type_name not in group_stats:
                continue
            gs = group_stats[type_name]
            x = gs['episodes']
            y = gs[f'{col}_mean']
            e = gs[f'{col}_sem']
            xs, ys, yl, yu = prepare_curve(x, y, e, smooth_window=SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu,
                          color=TYPE_COLORS[type_name],
                          label=TYPE_NAMES[type_name])
        style_axes(ax, xlabel='Episode', ylabel='Average Wall Collision Count',
                   title=f'Player {pid}',
                   legend_kwargs={'loc': 'upper right', 'fontsize': 8})
        ax.set_ylim(bottom=0)

    fig.suptitle('Player Wall Collision Count vs Episode (Wall Shaping Comparison)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "wall_potential_comparison_player_walls.png")


def plot_total_walls(group_stats):
    """1 plot: total wall collision count mean ± SEM."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    stats_lines = ["Mean Wall Collisions:"]
    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if type_name not in group_stats:
            continue
        gs = group_stats[type_name]
        x = gs['episodes']
        y = gs['total_walls_mean']
        e = gs['total_walls_sem']
        xs, ys, yl, yu = prepare_curve(x, y, e, smooth_window=SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu,
                      color=TYPE_COLORS[type_name],
                      label=TYPE_NAMES[type_name])
        stats_lines.append(f"  {TYPE_NAMES[type_name]}: {y.mean():.1f}")

    from publication_plot_utils import add_stats_box
    add_stats_box(ax, "\n".join(stats_lines), loc='upper left', fontsize=9)

    style_axes(ax, xlabel='Episode', ylabel='Total Wall Collision Count',
               title='Total Wall Collision Count vs Episode (Wall Shaping Comparison)',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10,
                              'title': 'Wall Shaping Method'})
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "wall_potential_comparison_total_walls.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Wall Potential Comparison — Publication-Style Analysis")
    print("=" * 70)

    setup_style()

    # 1. Load files
    print("\n[1/4] Loading data files...")
    data_files = load_data_files()
    for tn, files in data_files.items():
        print(f"  {TYPE_NAMES[tn]}: {len(files)} files")

    # 2. Process each group
    print("\n[2/4] Processing per-file → mean ± SEM...")
    group_stats = {}
    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if len(data_files[type_name]) == 0:
            print(f"  [!] Skipping {TYPE_NAMES[type_name]}: no files")
            continue
        group_stats[type_name] = aggregate_group(data_files[type_name], type_name)

    # 3. Save summary CSVs
    print("\n[3/4] Saving summary CSVs...")
    for type_name, gs in group_stats.items():
        rows = []
        for ep in range(gs['num_eps']):
            row = {'episode_id': ep + 1}
            for pid in PLAYERS:
                row[f'player_{pid}_score'] = gs[f'p{pid}_score_mean'][ep]
                row[f'player_{pid}_score_sem'] = gs[f'p{pid}_score_sem'][ep]
                row[f'player_{pid}_wall_count'] = gs[f'p{pid}_walls_mean'][ep]
                row[f'player_{pid}_wall_count_sem'] = gs[f'p{pid}_walls_sem'][ep]
            row['total_score'] = gs['total_score_mean'][ep]
            row['total_score_sem'] = gs['total_score_sem'][ep]
            row['total_wall_count'] = gs['total_walls_mean'][ep]
            row['total_wall_count_sem'] = gs['total_walls_sem'][ep]
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"wall_potential_comparison_{type_name}_stats.csv", index=False)

    # Overall summary
    summary_rows = []
    for type_name, gs in group_stats.items():
        row = {'type': type_name, 'type_name': TYPE_NAMES[type_name],
               'num_episodes': gs['num_eps'],
               'mean_total_score': gs['total_score_mean'].mean(),
               'mean_total_walls': gs['total_walls_mean'].mean()}
        for pid in PLAYERS:
            row[f'p{pid}_mean_score'] = gs[f'p{pid}_score_mean'].mean()
            row[f'p{pid}_mean_walls'] = gs[f'p{pid}_walls_mean'].mean()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "wall_potential_comparison_summary.csv", index=False)
    print(f"  Saved CSVs to {OUTPUT_DIR}")

    # 4. Generate plots
    print("\n[4/4] Generating publication-quality plots...")
    plot_player_scores(group_stats)
    plot_total_scores(group_stats)
    plot_player_walls(group_stats)
    plot_total_walls(group_stats)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for type_name, gs in group_stats.items():
        print(f"\n{TYPE_NAMES[type_name]} ({gs['num_eps']} episodes):")
        print(f"  Mean Total Score: {gs['total_score_mean'].mean():.1f}")
        print(f"  Mean Total Walls: {gs['total_walls_mean'].mean():.1f}")
        for pid in PLAYERS:
            print(f"    Player {pid}: Score={gs[f'p{pid}_score_mean'].mean():.1f}, "
                  f"Walls={gs[f'p{pid}_walls_mean'].mean():.1f}")

    print(f"\nOutput: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()