"""
Network Architecture Comparison — Publication-Style Analysis
=============================================================
Compares 4 network architectures across 3 sub-experiments:
  1. S1   (s1_*)   — curriculum stage 1 (ball collection)
  2. S4   (s4_*)   — curriculum stage 4 (full combat, multi-agent)
  3. Sarl (sarl_*) — single-agent full game

Metrics:
  - S1:   total score (all values) + ball score (collect_ball_A/B only)
  - S4:   total score (all values)
  - Sarl: total score + 6 event counts (attack/damage/wall/kill/death)

Data processing: per-file → mean ± SEM across 4 runs (real error bands)
Drawing: publication_plot_utils pipeline (Gaussian filter → B-spline → fill_between)
Colors: Wong 2011 colorblind-friendly DEFAULT_PALETTE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
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

NETWORK_ORDER = ['MLP', '分段MLP', 'GRU-MLP 96', 'GRU-MLP 128']
NETWORK_COLORS = {
    'MLP':            DEFAULT_PALETTE[0],  # #0173B2 blue
    '分段MLP':        DEFAULT_PALETTE[1],  # #DE8F05 orange
    'GRU-MLP 96':     DEFAULT_PALETTE[2],  # #029E73 green
    'GRU-MLP 128':    DEFAULT_PALETTE[3],  # #CC78BC pink
}
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
PLAYER_IDS = [0, 1, 2, 3]
SMOOTH_WINDOW = 5  # Gaussian sigma = 5 × 0.35 = 1.75


# ============================================================
# File discovery
# ============================================================
def discover_files(prefix):
    """Discover files grouped by network type for a given prefix (s1/s4/sarl)."""
    patterns = {
        'MLP':            f'{prefix}_mlp_*.csv',
        '分段MLP':        f'{prefix}_seg_mlp_*.csv',
        'GRU-MLP 96':     f'{prefix}_96_gru_mlp_*.csv',
        'GRU-MLP 128':    f'{prefix}_128_gru_mlp_*.csv',
    }
    result = {}
    for net, pat in patterns.items():
        files = sorted(glob.glob(str(DATA_DIR / pat)))
        if files:
            result[net] = files
    return result


def find_max_episodes(all_network_files):
    """Find minimum max-episode across all files for truncation."""
    min_ep = None
    for net, files in all_network_files.items():
        for fp in files:
            df = pd.read_csv(fp, usecols=['episode_id'])
            me = int(df['episode_id'].max())
            if min_ep is None or me < min_ep:
                min_ep = me
    return min_ep


# ============================================================
# S1 & S4: Multi-agent processing
# ============================================================
def process_multi_file(fp, max_eps, ball_only=False):
    """
    Process one multi-agent file (S1/S4).
    Returns dict: {ep: {total: float, player_{0..3}: float, [ball_*]}}
    """
    df = pd.read_csv(fp)
    result = {}
    for ep in range(1, max_eps + 1):
        ep_df = df[df['episode_id'] == ep]
        row = {}
        for pid in PLAYER_IDS:
            p_df = ep_df[ep_df['player_id'] == pid]
            # Total score
            row[f'p{pid}_total'] = p_df['value'].sum()
            # Ball score
            ball_df = p_df[p_df['source'].isin(BALL_SOURCES)]
            row[f'p{pid}_ball'] = ball_df['value'].sum()
        row['total'] = sum(row[f'p{p}_total'] for p in PLAYER_IDS)
        row['ball'] = sum(row[f'p{p}_ball'] for p in PLAYER_IDS)
        result[ep] = row
    return result


def aggregate_multi_group(file_list, max_eps, metrics):
    """
    Aggregate multi-agent group. metrics: list of ['total', 'ball', 'p0_total', ...]
    Returns: dict with *_mean and *_sem arrays
    """
    n = len(file_list)
    all_results = [process_multi_file(fp, max_eps) for fp in file_list]

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
# Sarl: Single-agent processing
# ============================================================
EVENT_SOURCES = ['attack', 'bear_damage', 'cause_damage_to_enemy',
                 'wall_collision', 'kill_enemy', 'died']

def process_sarl_file(fp, max_eps):
    """Process one single-agent file. Returns dict {ep: {total_score, attack, ...}}"""
    df = pd.read_csv(fp)
    result = {}
    for ep in range(1, max_eps + 1):
        ep_df = df[df['episode_id'] == ep]
        row = {'total_score': ep_df['value'].sum()}
        for src in EVENT_SOURCES:
            row[src] = len(ep_df[ep_df['source'] == src])
        result[ep] = row
    return result


def aggregate_sarl_group(file_list, max_eps):
    """Aggregate single-agent group."""
    n = len(file_list)
    all_results = [process_sarl_file(fp, max_eps) for fp in file_list]

    metrics = ['total_score'] + EVENT_SOURCES
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
# Plotting: S1
# ============================================================
PLOT_ORDER = NETWORK_ORDER

def plot_s1_player_total(group_stats):
    """4 subplots: per-player total score."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    for idx, pid in enumerate(PLAYER_IDS):
        ax = axes[idx]; col = f'p{pid}_total'
        for net in PLOT_ORDER:
            if net not in group_stats: continue
            gs = group_stats[net]; x = gs['episodes']
            xs, ys, yl, yu = prepare_curve(x, gs[f'{col}_mean'], gs[f'{col}_sem'], SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        style_axes(ax, xlabel='回合', ylabel='平均总得分',
                   title=f'智能体 {pid}', legend_kwargs={'loc': 'upper left', 'fontsize': 8})
    fig.suptitle('S1：各智能体总得分——网络结构对比',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s1_player_total_score.png")


def plot_s1_total(group_stats):
    """1 plot: sum of all players total score."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = ["平均总得分："]
    for net in PLOT_ORDER:
        if net not in group_stats: continue
        gs = group_stats[net]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs['total_mean'], gs['total_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        lines.append(f"  {net}: {gs['total_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='回合', ylabel='总得分（4智能体之和）',
               title='S1：总得分——网络结构对比',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': '网络结构'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s1_total_score.png")


def plot_s1_player_ball(group_stats):
    """4 subplots: per-player ball score."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    for idx, pid in enumerate(PLAYER_IDS):
        ax = axes[idx]; col = f'p{pid}_ball'
        for net in PLOT_ORDER:
            if net not in group_stats: continue
            gs = group_stats[net]; x = gs['episodes']
            xs, ys, yl, yu = prepare_curve(x, gs[f'{col}_mean'], gs[f'{col}_sem'], SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        style_axes(ax, xlabel='回合', ylabel='平均采球得分',
                   title=f'智能体 {pid}', legend_kwargs={'loc': 'upper left', 'fontsize': 8})
    fig.suptitle('S1：各智能体采球得分——网络结构对比',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s1_player_ball_score.png")


def plot_s1_ball_total(group_stats):
    """1 plot: sum of all players ball score."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = ["平均采球得分："]
    for net in PLOT_ORDER:
        if net not in group_stats: continue
        gs = group_stats[net]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs['ball_mean'], gs['ball_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        lines.append(f"  {net}: {gs['ball_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='回合', ylabel='采球总得分（4智能体之和）',
               title='S1：采球得分——网络结构对比',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': '网络结构'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s1_total_ball_score.png")


# ============================================================
# Plotting: S4
# ============================================================
def plot_s4_player_total(group_stats):
    """4 subplots: per-player total score."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    for idx, pid in enumerate(PLAYER_IDS):
        ax = axes[idx]; col = f'p{pid}_total'
        for net in PLOT_ORDER:
            if net not in group_stats: continue
            gs = group_stats[net]; x = gs['episodes']
            xs, ys, yl, yu = prepare_curve(x, gs[f'{col}_mean'], gs[f'{col}_sem'], SMOOTH_WINDOW)
            plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        style_axes(ax, xlabel='回合', ylabel='平均总得分',
                   title=f'智能体 {pid}', legend_kwargs={'loc': 'upper left', 'fontsize': 8})
    fig.suptitle('S4：各智能体总得分——网络结构对比',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s4_player_total_score.png")


def plot_s4_total(group_stats):
    """1 plot: sum of all players total score."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = ["平均总得分："]
    for net in PLOT_ORDER:
        if net not in group_stats: continue
        gs = group_stats[net]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs['total_mean'], gs['total_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        lines.append(f"  {net}: {gs['total_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='回合', ylabel='总得分（4智能体之和）',
               title='S4：总得分——网络结构对比',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': '网络结构'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "s4_total_score.png")


# ============================================================
# Plotting: Sarl
# ============================================================
EVENT_LABELS = {
    'total_score':          ('总得分', 'sarl_total_score.png'),
    'attack':               ('攻击次数', 'sarl_attack_count.png'),
    'bear_damage':          ('承受伤害', 'sarl_damage_taken.png'),
    'cause_damage_to_enemy':('造成伤害', 'sarl_damage_dealt.png'),
    'wall_collision':       ('撞墙次数', 'sarl_wall_collision.png'),
    'kill_enemy':           ('击杀数', 'sarl_kill_count.png'),
    'died':                 ('死亡数', 'sarl_death_count.png'),
}

def plot_sarl_metric(group_stats, metric, y_label, fname):
    """1 plot: single-agent metric."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    lines = [f"平均 {y_label}："]
    for net in PLOT_ORDER:
        if net not in group_stats: continue
        gs = group_stats[net]; x = gs['episodes']
        xs, ys, yl, yu = prepare_curve(x, gs[f'{metric}_mean'], gs[f'{metric}_sem'], SMOOTH_WINDOW)
        plot_with_fill(ax, xs, ys, yl, yu, color=NETWORK_COLORS[net], label=net)
        lines.append(f"  {net}: {gs[f'{metric}_mean'].mean():.1f}")
    add_stats_box(ax, "\n".join(lines), loc='upper left', fontsize=9)
    style_axes(ax, xlabel='回合', ylabel=y_label,
               title=f'Sarl：{y_label}——网络结构对比',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': '网络结构'})
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / fname)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Network Architecture Comparison — Publication-Style Analysis")
    print("=" * 70)
    setup_style()

    # ── S1 ──
    print("\n" + "─" * 50)
    print("S1: Curriculum Stage 1 (Ball Collection)")
    print("─" * 50)
    s1_files = discover_files('s1')
    if s1_files:
        max_ep = find_max_episodes(s1_files)
        print(f"  Files per network: {[len(v) for v in s1_files.values()]}, max_ep={max_ep}")
        s1_metrics = ['total', 'ball'] + [f'p{p}_total' for p in PLAYER_IDS] + [f'p{p}_ball' for p in PLAYER_IDS]
        s1_stats = {}
        for net in PLOT_ORDER:
            if net in s1_files:
                s1_stats[net] = aggregate_multi_group(s1_files[net], max_ep, s1_metrics)
                print(f"  {net}: total_mean={s1_stats[net]['total_mean'].mean():.1f}, ball_mean={s1_stats[net]['ball_mean'].mean():.1f}")
        plot_s1_player_total(s1_stats)
        plot_s1_total(s1_stats)
        plot_s1_player_ball(s1_stats)
        plot_s1_ball_total(s1_stats)
        # Save CSVs
        for net, gs in s1_stats.items():
            rows = [{'episode_id': ep+1,
                     'total_score': gs['total_mean'][ep], 'total_sem': gs['total_sem'][ep],
                     'ball_score': gs['ball_mean'][ep], 'ball_sem': gs['ball_sem'][ep]}
                    for ep in range(gs['num_eps'])]
            for pid in PLAYER_IDS:
                for ep in range(gs['num_eps']):
                    rows[ep][f'p{pid}_total'] = gs[f'p{pid}_total_mean'][ep]
                    rows[ep][f'p{pid}_ball'] = gs[f'p{pid}_ball_mean'][ep]
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"s1_{net.replace(' ','_').replace('-','_')}_stats.csv", index=False)
    else:
        print("  SKIP: No S1 files found")

    # ── S4 ──
    print("\n" + "─" * 50)
    print("S4: Curriculum Stage 4 (Full Combat)")
    print("─" * 50)
    s4_files = discover_files('s4')
    if s4_files:
        max_ep = find_max_episodes(s4_files)
        print(f"  Files per network: {[len(v) for v in s4_files.values()]}, max_ep={max_ep}")
        s4_metrics = ['total'] + [f'p{p}_total' for p in PLAYER_IDS]
        s4_stats = {}
        for net in PLOT_ORDER:
            if net in s4_files:
                s4_stats[net] = aggregate_multi_group(s4_files[net], max_ep, s4_metrics)
                print(f"  {net}: total_mean={s4_stats[net]['total_mean'].mean():.1f}")
        plot_s4_player_total(s4_stats)
        plot_s4_total(s4_stats)
        for net, gs in s4_stats.items():
            rows = [{'episode_id': ep+1, 'total_score': gs['total_mean'][ep],
                     'total_sem': gs['total_sem'][ep]} for ep in range(gs['num_eps'])]
            for pid in PLAYER_IDS:
                for ep in range(gs['num_eps']):
                    rows[ep][f'p{pid}_total'] = gs[f'p{pid}_total_mean'][ep]
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"s4_{net.replace(' ','_').replace('-','_')}_stats.csv", index=False)
    else:
        print("  SKIP: No S4 files found")

    # ── Sarl ──
    print("\n" + "─" * 50)
    print("Sarl: Single-Agent Full Game")
    print("─" * 50)
    sarl_files = discover_files('sarl')
    if sarl_files:
        max_ep = find_max_episodes(sarl_files)
        print(f"  Files per network: {[len(v) for v in sarl_files.values()]}, max_ep={max_ep}")
        sarl_stats = {}
        for net in PLOT_ORDER:
            if net in sarl_files:
                sarl_stats[net] = aggregate_sarl_group(sarl_files[net], max_ep)
                print(f"  {net}: total_mean={sarl_stats[net]['total_score_mean'].mean():.1f}")
        for metric, (label, fname) in EVENT_LABELS.items():
            plot_sarl_metric(sarl_stats, metric, label, fname)
        # CSVs
        for net, gs in sarl_stats.items():
            rows = []
            for ep in range(gs['num_eps']):
                row = {'episode_id': ep + 1}
                for m in ['total_score'] + EVENT_SOURCES:
                    row[f'{m}'] = gs[f'{m}_mean'][ep]
                    row[f'{m}_sem'] = gs[f'{m}_sem'][ep]
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"sarl_{net.replace(' ','_').replace('-','_')}_stats.csv", index=False)
    else:
        print("  SKIP: No Sarl files found")

    print(f"\n{'=' * 70}")
    print(f"Done. Output: {OUTPUT_DIR}")
    print(f"Generated {len(list(OUTPUT_DIR.glob('*.png')))} PNGs + {len(list(OUTPUT_DIR.glob('*_stats.csv')))} CSVs")
    print("=" * 70)


if __name__ == "__main__":
    main()