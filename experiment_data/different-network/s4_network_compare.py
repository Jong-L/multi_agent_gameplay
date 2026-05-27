"""
S4 环境网络结构得分对比分析
S4 Environment: Compare average scores across different network architectures
(MLP, Segmented MLP, GRU-MLP 96, GRU-MLP 128)

生成图表:
  1. 每个玩家平均所有类型得分 vs 回合数 (2x2子图, 4条曲线)
  2. 所有玩家平均所有类型得分总和 vs 回合数 (1图, 4条曲线)

数据文件要求:
  - 仅分析 s4_ 前缀文件
  - 回合数不一致 → 警告并使用最小回合数截断
  - 玩家缺失 → 该回合该玩家得分为0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import sys


def _set_chinese_font():
    """
    动态检测并设置中文字体。
    在 sns.set_* / setup_style() 之后调用，防止被覆盖。
    """
    import matplotlib
    import matplotlib.font_manager as fm
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'PingFang SC',
        'KaiTi', 'FangSong', 'STKaiti', 'SimSun', 'YouYuan',
    ]
    available = set(f.name for f in fm.fontManager.ttflist)
    for c in candidates:
        if c in available:
            matplotlib.rcParams['font.sans-serif'] = [c, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = Path(__file__).parent  # D:/.../experiment_data/different-network/
SUMMARY_DIR = DATA_DIR / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW_RATIO = 0.05  # 5% of total episodes

# Seaborn publication style (consistent with data_analyze/)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
_set_chinese_font()

# Colorblind-friendly palette
NETWORK_COLORS = {
    'MLP':            '#0173B2',   # Blue
    '分段MLP':  '#DE8F05',   # Orange
    'GRU-MLP 96':     '#029E73',   # Green
    'GRU-MLP 128':    '#D55E00',   # Vermillion
}

NETWORK_DISPLAY_ORDER = ['MLP', '分段MLP', 'GRU-MLP 96', 'GRU-MLP 128']

# File pattern → display name mapping (s4 only)
NETWORK_PATTERNS = {
    'MLP':            's4_mlp_*.csv',
    '分段MLP':  's4_seg_mlp_*.csv',
    'GRU-MLP 96':     's4_96_gru_mlp_*.csv',
    'GRU-MLP 128':    's4_128_gru_mlp_*.csv',
}

PLAYER_IDS = [0, 1, 2, 3]

# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_smooth_window(n_episodes):
    """Compute smoothing window as 5% of total episodes, minimum 1."""
    return max(1, round(n_episodes * SMOOTH_WINDOW_RATIO))


def smooth_curve(y, window=5):
    """Apply moving average smoothing"""
    if len(y) < window or window <= 1:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def validate_episode_counts(all_network_files):
    """
    Check that every file across all networks has the same max episode.
    Print warning if mismatch detected. Never aborts analysis.
    
    Returns:
        (is_valid, common_episodes) tuple
    """
    all_counts = {}
    for net_name, files in all_network_files.items():
        for fp in files:
            df = pd.read_csv(fp, usecols=['episode_id'])
            all_counts[Path(fp).name] = int(df['episode_id'].max())
    
    unique_counts = sorted(set(all_counts.values()))
    
    if len(unique_counts) > 1:
        print("\n" + "!" * 60)
        print("WARNING: Episode count mismatch detected!")
        print("!" * 60)
        for fname in sorted(all_counts.keys()):
            ec = all_counts[fname]
            print(f"  {fname}: {ec} episodes")
        min_ep = unique_counts[0]
        print("-" * 60)
        print(f"  → Analysis will be TRUNCATED to {min_ep} episodes")
        print(f"    (minimum across all files)")
        print("!" * 60)
        return False, min_ep
    
    print(f"\nAll files have consistent episode count: {unique_counts[0]}")
    return True, unique_counts[0]


def load_and_aggregate_network(network_name, file_paths, max_episodes=None):
    """
    Load all PID files for one network type and aggregate.
    
    For each PID file:
      1. Compute per-episode per-player total score (sum of ALL sources)
      2. Pivot to wide format (episode × player), fill missing players/episodes with 0
    
    Then average across all PID files (mean of PIDs per episode×player).
    
    Args:
        network_name: display name of the network
        file_paths: list of CSV paths for this network type
        max_episodes: optional cap on episode range (for cross-file truncation)
    
    Returns:
        total_df: DataFrame with columns [episode_id, player_0, ..., player_3, total]
    """
    all_pid_scores = []
    
    for fp in file_paths:
        df = pd.read_csv(fp)
        
        # Per-episode per-player total score (all reward types summed)
        pid_total = df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        pid_total.columns = ['episode_id', 'player_id', 'total_score']
        
        # Pivot to wide format: rows=episodes, cols=players
        pivot = pid_total.pivot_table(
            index='episode_id', columns='player_id',
            values='total_score', fill_value=0
        )
        # Ensure all players 0-3 exist
        for p in PLAYER_IDS:
            if p not in pivot.columns:
                pivot[p] = 0
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        pivot = pivot.sort_index()
        
        # Truncate if needed
        if max_episodes is not None:
            pivot = pivot.loc[pivot.index <= max_episodes]
        
        all_pid_scores.append(pivot)
    
    # Build common episode index
    common_episodes = sorted(set().union(*[set(p.index) for p in all_pid_scores]))
    
    # Reindex all PIDs to common episodes, fill missing with 0
    pid_pivots_aligned = []
    for pivot in all_pid_scores:
        pivot_aligned = pivot.reindex(common_episodes, fill_value=0)
        pid_pivots_aligned.append(pivot_aligned)
    
    # Mean across PIDs
    total_mean = sum(pid_pivots_aligned) / len(pid_pivots_aligned)
    total_mean = total_mean.reset_index()
    total_mean.columns = ['episode_id'] + [f'player_{p}' for p in sorted(total_mean.columns[1:])]
    
    # Compute total across all players
    player_cols = [c for c in total_mean.columns if c.startswith('player_')]
    total_mean['total'] = total_mean[player_cols].sum(axis=1)
    
    return total_mean


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_per_player_all_scores(network_data, smooth_window, save_path):
    """
    Plot 1: 2x2 subplots, one per player.
    Each subplot shows 4 curves (4 network types).
    Y-axis: mean ALL score types per episode.
    """
    _set_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('各智能体平均总得分随回合变化\n（所有奖励类型）',
                 fontsize=16, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for p_idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[p_idx]
        player_col = f'player_{player_id}'
        
        for net_name in NETWORK_DISPLAY_ORDER:
            if net_name not in network_data:
                continue
            df = network_data[net_name]
            episodes = df['episode_id'].values
            scores = df[player_col].values
            
            if smooth_window > 1 and len(scores) > smooth_window:
                smoothed = smooth_curve(scores, smooth_window)
                smoothed_ep = episodes[smooth_window - 1:]
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
    plt.close(fig)
    return fig


def plot_total_all_scores(network_data, smooth_window, save_path):
    """
    Plot 2: 1 subplot, 4 curves.
    Y-axis: sum of all players' average all-type scores.
    """
    _set_chinese_font()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    stats_parts = []
    
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]
        episodes = df['episode_id'].values
        total_scores = df['total'].values  # sum across all 4 players
        
        mean_val = total_scores.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")
        
        if smooth_window > 1 and len(total_scores) > smooth_window:
            smoothed = smooth_curve(total_scores, smooth_window)
            smoothed_ep = episodes[smooth_window - 1:]
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
    ax.set_title('总得分对比——网络结构\n（所有奖励类型）——S4环境',
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
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


# ==============================================================================
# Summary Statistics
# ==============================================================================

def print_summary(network_data):
    """Print summary statistics table"""
    print("\n" + "=" * 70)
    print("SUMMARY: S4 Network Architecture Score Comparison")
    print("=" * 70)
    
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]
        
        print(f"\n{'─' * 50}")
        print(f"  {net_name}")
        print(f"{'─' * 50}")
        print(f"  Episodes: {len(df)}")
        print(f"  {'Player':<10} {'Avg Total Score':>16}")
        print(f"  {'─' * 28}")
        
        for p in PLAYER_IDS:
            col = f'player_{p}'
            avg = df[col].mean()
            print(f"  Player {p:<5} {avg:>16.2f}")
        
        print(f"  {'─' * 28}")
        print(f"  {'ALL SUM':<10} {df['total'].mean():>16.2f}")
    
    # Best network
    best_total = max(
        (network_data[n]['total'].mean(), n)
        for n in NETWORK_DISPLAY_ORDER if n in network_data
    )
    
    print(f"\n{'=' * 70}")
    print(f"BEST total score:  {best_total[1]} ({best_total[0]:.1f})")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("S4 Environment: Network Architecture Comparison Analysis")
    print("=" * 70)
    
    # ── Step 1: Discover s4 files only ──
    print("\n[Step 1] Discovering S4 data files...")
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
        print("\nERROR: No S4 data files found!")
        sys.exit(1)
    
    # ── Step 2: Validate episode consistency ──
    print("\n[Step 2] Validating episode consistency...")
    is_valid, common_eps = validate_episode_counts(all_network_files)
    if not is_valid:
        print("\n  → Truncating all files to common episode range.")
    
    # ── Step 3: Determine smoothing window ──
    smooth_window = compute_smooth_window(common_eps)
    print(f"\n  Smoothing window: {smooth_window} episodes ({SMOOTH_WINDOW_RATIO*100:.0f}% of {common_eps})")
    
    # ── Step 4: Load and aggregate data ──
    print("\n[Step 3] Loading and aggregating data...")
    network_data = {}
    for net_name, files in all_network_files.items():
        print(f"\n  Processing {net_name} ({len(files)} files)...")
        total_df = load_and_aggregate_network(net_name, files, max_episodes=common_eps)
        network_data[net_name] = total_df
        print(f"    Episodes: {len(total_df)}")
        print(f"    Total score range: [{total_df['total'].min():.1f}, {total_df['total'].max():.1f}]")
    
    # ── Step 5: Print summary ──
    print_summary(network_data)
    
    # ── Step 6: Generate plots ──
    print("\n[Step 4] Generating plots...")
    
    plot_per_player_all_scores(
        network_data,
        smooth_window,
        str(SUMMARY_DIR / "s4_01_per_player_all_scores.png")
    )
    
    plot_total_all_scores(
        network_data,
        smooth_window,
        str(SUMMARY_DIR / "s4_02_total_all_scores.png")
    )
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"Output directory: {SUMMARY_DIR}")
    print(f"Generated {len(list(SUMMARY_DIR.glob('s4_*.png')))} S4 plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
