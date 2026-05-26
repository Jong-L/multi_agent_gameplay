"""
不同网络结构得分对比分析
Compare average scores across different network architectures
(MLP, Segmented MLP, GRU-MLP 96, GRU-MLP 128)

生成4张图表:
  1. 每个玩家平均所有类型得分 vs 回合数 (2x2子图, 4条曲线)
  2. 所有玩家平均所有类型得分总和 vs 回合数 (1图, 4条曲线)
  3. 每个玩家平均吃球得分 vs 回合数 (2x2子图, 4条曲线)
  4. 所有玩家平均吃球得分总和 vs 回合数 (1图, 4条曲线)
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import sys

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = Path(__file__).parent  # D:/.../experiment_data/different-network/
SUMMARY_DIR = DATA_DIR / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 5  # moving average window

# Seaborn publication style (consistent with data_analyze/)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Colorblind-friendly palette
NETWORK_COLORS = {
    'MLP':            '#0173B2',   # Blue
    '分段MLP':  '#DE8F05',   # Orange
    'GRU-MLP 96':     '#029E73',   # Green
    'GRU-MLP 128':    '#D55E00',   # Vermillion
}

NETWORK_DISPLAY_ORDER = ['MLP', '分段MLP', 'GRU-MLP 96', 'GRU-MLP 128']

# File pattern → display name mapping
NETWORK_PATTERNS = {
    'MLP':            's1_mlp_*.csv',
    '分段MLP':  's1_seg_mlp_*.csv',
    'GRU-MLP 96':     's1_96_gru_mlp_*.csv',
    'GRU-MLP 128':    's1_128_gru_mlp_*.csv',
}

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
PLAYER_IDS = [0, 1, 2, 3]

# ==============================================================================
# Utility Functions
# ==============================================================================

def smooth_curve(y, window=SMOOTH_WINDOW):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def _episodes_from_files(file_paths):
    """Extract max episode per file for validation"""
    ep_counts = {}
    for fp in file_paths:
        df = pd.read_csv(fp, usecols=['episode_id'])
        ep_counts[Path(fp).name] = df['episode_id'].max()
    return ep_counts


def load_and_aggregate_network(network_name, file_paths):
    """
    Load all PID files for one network type and aggregate.
    
    For each PID file:
      1. Compute per-episode per-player total score (sum of all values)
      2. Compute per-episode per-player ball score (sum of BALL_SOURCES)
    
    Then average across all PID files (mean of PIDs per episode×player).
    
    Returns:
        total_df: DataFrame with columns [episode_id, player_0, player_1, player_2, player_3, total]
        ball_df:  DataFrame with same structure for ball scores only
    """
    all_total_scores = []  # list of DataFrames, one per PID
    all_ball_scores = []
    
    for fp in file_paths:
        df = pd.read_csv(fp)
        
        # Per-episode per-player ALL scores
        pid_total = df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        pid_total.columns = ['episode_id', 'player_id', 'total_score']
        
        # Per-episode per-player BALL scores
        ball_df = df[df['source'].isin(BALL_SOURCES)]
        pid_ball = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        pid_ball.columns = ['episode_id', 'player_id', 'ball_score']
        
        all_total_scores.append(pid_total)
        all_ball_scores.append(pid_ball)
    
    # --- Aggregate TOTAL scores across PIDs ---
    # Build multi-index (episode_id × player_id)
    episodes = sorted(set().union(*[set(t['episode_id'].unique()) for t in all_total_scores]))
    
    # For each PID, build a full episode×player pivot, fill missing with 0
    pid_total_pivots = []
    for pid_df in all_total_scores:
        pivot = pid_df.pivot_table(
            index='episode_id', columns='player_id', 
            values='total_score', fill_value=0
        )
        # Ensure all players 0-3 exist
        for p in PLAYER_IDS:
            if p not in pivot.columns:
                pivot[p] = 0
        pivot = pivot.reindex(episodes, fill_value=0)
        pivot = pivot[sorted(pivot.columns)]
        pid_total_pivots.append(pivot)
    
    # Mean across PIDs
    total_mean = sum(pid_total_pivots) / len(pid_total_pivots)
    total_mean = total_mean.reset_index()
    total_mean.columns = ['episode_id'] + [f'player_{p}' for p in sorted(total_mean.columns[1:])]
    # Compute total across all players
    player_cols = [c for c in total_mean.columns if c.startswith('player_')]
    total_mean['total'] = total_mean[player_cols].sum(axis=1)
    
    # --- Aggregate BALL scores across PIDs ---
    pid_ball_pivots = []
    for pid_df in all_ball_scores:
        pivot = pid_df.pivot_table(
            index='episode_id', columns='player_id',
            values='ball_score', fill_value=0
        )
        for p in PLAYER_IDS:
            if p not in pivot.columns:
                pivot[p] = 0
        pivot = pivot.reindex(episodes, fill_value=0)
        pivot = pivot[sorted(pivot.columns)]
        pid_ball_pivots.append(pivot)
    
    ball_mean = sum(pid_ball_pivots) / len(pid_ball_pivots)
    ball_mean = ball_mean.reset_index()
    ball_mean.columns = ['episode_id'] + [f'player_{p}' for p in sorted(ball_mean.columns[1:])]
    player_cols_b = [c for c in ball_mean.columns if c.startswith('player_')]
    ball_mean['total'] = ball_mean[player_cols_b].sum(axis=1)
    
    return total_mean, ball_mean


def validate_episode_counts(all_network_files):
    """
    Check that every file across all networks has the same max episode.
    Print warning if mismatch detected.
    
    Returns:
        (is_valid, common_episodes) tuple
    """
    all_counts = {}
    for net_name, files in all_network_files.items():
        for fp in files:
            df = pd.read_csv(fp, usecols=['episode_id'])
            all_counts[Path(fp).name] = df['episode_id'].max()
    
    unique_counts = set(all_counts.values())
    if len(unique_counts) > 1:
        print("\n" + "!" * 60)
        print("WARNING: Episode count mismatch detected!")
        print("!" * 60)
        for fname, ec in sorted(all_counts.items()):
            print(f"  {fname}: {ec} episodes")
        print("!" * 60)
        return False, min(unique_counts)
    
    print(f"\nAll files have consistent episode count: {list(unique_counts)[0]}")
    return True, list(unique_counts)[0]


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_per_player_all_scores(network_data, save_path):
    """
    Plot 1: 2x2 subplots, one per player.
    Each subplot shows 4 curves (4 network types).
    Y-axis: mean ALL score types per episode.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('各智能体平均总得分随回合变化\n（所有奖励类型）',
                 fontsize=16, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for p_idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[p_idx]
        player_col = f'player_{player_id}'
        
        for net_name in NETWORK_DISPLAY_ORDER:
            if net_name not in network_data:
                continue
            df = network_data[net_name]['total']
            episodes = df['episode_id'].values
            scores = df[player_col].values
            
            if SMOOTH_WINDOW > 1 and len(scores) > SMOOTH_WINDOW:
                smoothed = smooth_curve(scores, SMOOTH_WINDOW)
                smoothed_ep = episodes[SMOOTH_WINDOW - 1:]
            else:
                smoothed = scores
                smoothed_ep = episodes
            
            sns.lineplot(x=smoothed_ep, y=smoothed,
                         label=net_name,
                         color=NETWORK_COLORS[net_name],
                         linewidth=1.5, ax=ax)
            
            # Semi-transparent fill (band = 4% of data range)
            data_range = smoothed.max() - smoothed.min()
            band = max(data_range * 0.04, 1.0)
            ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                            alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)
        
        ax.set_xlabel('回合', fontsize=11)
        ax.set_ylabel('平均总得分', fontsize=11)
        ax.set_title(f'智能体 {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        ax.set_xlim(0, None)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig


def plot_total_all_scores(network_data, save_path):
    """
    Plot 2: 1 subplot, 4 curves.
    Y-axis: sum of all players' average all-type scores.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    stats_parts = []
    
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]['total']
        episodes = df['episode_id'].values
        total_scores = df['total'].values  # sum across all 4 players
        
        mean_val = total_scores.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")
        
        if SMOOTH_WINDOW > 1 and len(total_scores) > SMOOTH_WINDOW:
            smoothed = smooth_curve(total_scores, SMOOTH_WINDOW)
            smoothed_ep = episodes[SMOOTH_WINDOW - 1:]
        else:
            smoothed = total_scores
            smoothed_ep = episodes
        
        sns.lineplot(x=smoothed_ep, y=smoothed,
                     label=net_name,
                     color=NETWORK_COLORS[net_name],
                     linewidth=1.5, ax=ax)
        
        # Semi-transparent fill (band = 4% of data range)
        data_range = smoothed.max() - smoothed.min()
        band = max(data_range * 0.04, 1.0)
        ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                        alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)
    
    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel('总得分（所有智能体之和）', fontsize=12)
    ax.set_title('总得分对比——网络结构\n（所有奖励类型）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='网络结构')
    ax.set_xlim(0, None)
    
    # Statistics annotation
    stats_text = "平均总得分：\n" + "\n".join(stats_parts)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig


def plot_per_player_ball_scores(network_data, save_path):
    """
    Plot 3: 2x2 subplots, one per player.
    Each subplot shows 4 curves.
    Y-axis: mean BALL score per episode.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('各智能体平均采球得分随回合变化',
                 fontsize=16, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for p_idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[p_idx]
        player_col = f'player_{player_id}'
        
        for net_name in NETWORK_DISPLAY_ORDER:
            if net_name not in network_data:
                continue
            df = network_data[net_name]['ball']
            episodes = df['episode_id'].values
            scores = df[player_col].values
            
            if SMOOTH_WINDOW > 1 and len(scores) > SMOOTH_WINDOW:
                smoothed = smooth_curve(scores, SMOOTH_WINDOW)
                smoothed_ep = episodes[SMOOTH_WINDOW - 1:]
            else:
                smoothed = scores
                smoothed_ep = episodes
            
            sns.lineplot(x=smoothed_ep, y=smoothed,
                         label=net_name,
                         color=NETWORK_COLORS[net_name],
                         linewidth=1.5, ax=ax)
            
            # Semi-transparent fill (band = 4% of data range)
            data_range = smoothed.max() - smoothed.min()
            band = max(data_range * 0.04, 1.0)
            ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                            alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)
        
        ax.set_xlabel('回合', fontsize=11)
        ax.set_ylabel('平均采球得分', fontsize=11)
        ax.set_title(f'智能体 {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        ax.set_xlim(0, None)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig


def plot_total_ball_scores(network_data, save_path):
    """
    Plot 4: 1 subplot, 4 curves.
    Y-axis: sum of all players' average ball scores.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    stats_parts = []
    
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]['ball']
        episodes = df['episode_id'].values
        total_scores = df['total'].values
        
        mean_val = total_scores.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")
        
        if SMOOTH_WINDOW > 1 and len(total_scores) > SMOOTH_WINDOW:
            smoothed = smooth_curve(total_scores, SMOOTH_WINDOW)
            smoothed_ep = episodes[SMOOTH_WINDOW - 1:]
        else:
            smoothed = total_scores
            smoothed_ep = episodes
        
        sns.lineplot(x=smoothed_ep, y=smoothed,
                     label=net_name,
                     color=NETWORK_COLORS[net_name],
                     linewidth=1.5, ax=ax)
        
        # Semi-transparent fill (band = 4% of data range)
        data_range = smoothed.max() - smoothed.min()
        band = max(data_range * 0.04, 1.0)
        ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                        alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)
    
    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel('采球总得分（所有智能体之和）', fontsize=12)
    ax.set_title('采球总得分对比——网络结构',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='网络结构')
    ax.set_xlim(0, None)
    
    # Statistics annotation
    stats_text = "平均采球得分：\n" + "\n".join(stats_parts)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# Summary Statistics
# ==============================================================================

def print_summary(network_data):
    """Print summary statistics table"""
    print("\n" + "=" * 70)
    print("SUMMARY: Network Architecture Score Comparison")
    print("=" * 70)
    
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        total_df = network_data[net_name]['total']
        ball_df = network_data[net_name]['ball']
        
        print(f"\n{'─' * 50}")
        print(f"  {net_name}")
        print(f"{'─' * 50}")
        print(f"  Episodes: {len(total_df)}")
        print(f"  {'Player':<10} {'Total Score Avg':>15} {'Ball Score Avg':>15}")
        print(f"  {'─' * 40}")
        
        for p in PLAYER_IDS:
            col = f'player_{p}'
            total_avg = total_df[col].mean()
            ball_avg = ball_df[col].mean()
            print(f"  Player {p:<5} {total_avg:>15.2f} {ball_avg:>15.2f}")
        
        print(f"  {'─' * 40}")
        print(f"  {'ALL SUM':<10} {total_df['total'].mean():>15.2f} {ball_df['total'].mean():>15.2f}")
    
    # Best network
    best_total = max(
        (network_data[n]['total']['total'].mean(), n)
        for n in NETWORK_DISPLAY_ORDER if n in network_data
    )
    best_ball = max(
        (network_data[n]['ball']['total'].mean(), n)
        for n in NETWORK_DISPLAY_ORDER if n in network_data
    )
    
    print(f"\n{'=' * 70}")
    print(f"BEST total score:  {best_total[1]} ({best_total[0]:.1f})")
    print(f"BEST ball score:   {best_ball[1]} ({best_ball[0]:.1f})")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Network Architecture Comparison Analysis")
    print("=" * 70)
    
    # ── Step 1: Discover files ──
    print("\n[Step 1] Discovering data files...")
    all_network_files = {}
    for net_name, pattern in NETWORK_PATTERNS.items():
        files = sorted(glob.glob(str(DATA_DIR / pattern)))
        if files:
            all_network_files[net_name] = files
            print(f"  {net_name}: {len(files)} files found")
            for f in files:
                print(f"    - {Path(f).name}")
        else:
            print(f"  {net_name}: NO FILES FOUND!")
    
    if not all_network_files:
        print("\nERROR: No data files found!")
        sys.exit(1)
    
    # ── Step 2: Validate episode consistency ──
    print("\n[Step 2] Validating episode consistency...")
    is_valid, common_eps = validate_episode_counts(all_network_files)
    if not is_valid:
        print("\nProceeding with analysis using minimum common episodes...")
    
    # ── Step 3: Load and aggregate data ──
    print("\n[Step 3] Loading and aggregating data...")
    network_data = {}
    for net_name, files in all_network_files.items():
        print(f"\n  Processing {net_name} ({len(files)} files)...")
        total_df, ball_df = load_and_aggregate_network(net_name, files)
        network_data[net_name] = {'total': total_df, 'ball': ball_df}
        print(f"    Total score range: [{total_df['total'].min():.1f}, {total_df['total'].max():.1f}]")
        print(f"    Ball score range:  [{ball_df['total'].min():.1f}, {ball_df['total'].max():.1f}]")
    
    # ── Step 4: Print summary ──
    print_summary(network_data)
    
    # ── Step 5: Generate plots ──
    print("\n[Step 4] Generating plots...")
    
    plot_per_player_all_scores(
        network_data,
        str(SUMMARY_DIR / "01_per_player_all_scores.png")
    )
    
    plot_total_all_scores(
        network_data,
        str(SUMMARY_DIR / "02_total_all_scores.png")
    )
    
    plot_per_player_ball_scores(
        network_data,
        str(SUMMARY_DIR / "03_per_player_ball_scores.png")
    )
    
    plot_total_ball_scores(
        network_data,
        str(SUMMARY_DIR / "04_total_ball_scores.png")
    )
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"Output directory: {SUMMARY_DIR}")
    print(f"Generated {len(list(SUMMARY_DIR.glob('*.png')))} plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
