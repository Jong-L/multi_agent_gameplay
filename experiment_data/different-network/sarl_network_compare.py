"""
Sarl (Single-Agent) Environment: Network Architecture Comparison Analysis
单智能体环境不同网络结构得分与事件统计对比

生成图表:
  1. 智能体平均所有类型得分总和 vs 回合数 (1图, 4条曲线)
  2. 智能体攻击次数 vs 回合数 (1图, 4条曲线)
  3. 智能体受击次数 vs 回合数 (1图, 4条曲线)
  4. 智能体撞墙次数 vs 回合数 (1图, 4条曲线)
  5. 智能体击杀次数 vs 回合数 (1图, 4条曲线)
  6. 智能体死亡次数 vs 回合数 (1图, 4条曲线)

数据文件要求:
  - 仅分析 sarl_ 前缀文件
  - 回合数不一致 → 警告并使用最小回合数截断
  - sarl为单智能体(player_id=0)，无多玩家问题
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
DATA_DIR = Path(__file__).parent
SUMMARY_DIR = DATA_DIR / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW_RATIO = 0.05  # 5% of total episodes

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
_set_chinese_font()

NETWORK_COLORS = {
    'MLP':            '#0173B2',   # Blue
    '分段MLP':  '#DE8F05',   # Orange
    'GRU-MLP 96':     '#029E73',   # Green
    'GRU-MLP 128':    '#D55E00',   # Vermillion
}

NETWORK_DISPLAY_ORDER = ['MLP', '分段MLP', 'GRU-MLP 96', 'GRU-MLP 128']

NETWORK_PATTERNS = {
    'MLP':            'sarl_mlp_*.csv',
    '分段MLP':  'sarl_seg_mlp_*.csv',
    'GRU-MLP 96':     'sarl_96_gru_mlp_*.csv',
    'GRU-MLP 128':    'sarl_128_gru_mlp_*.csv',
}

# Event sources to count (each row = one event)
EVENT_SOURCES = {
    'attack':               '攻击次数',
    'bear_damage':          '承受伤害',
    'cause_damage_to_enemy':'造成伤害',
    'wall_collision':       '撞墙次数',
    'kill_enemy':           '击杀数',
    'died':                 '死亡数',
}

# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_smooth_window(n_episodes):
    return max(1, round(n_episodes * SMOOTH_WINDOW_RATIO))


def smooth_curve(y, window=5):
    if len(y) < window or window <= 1:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def validate_episode_counts(all_network_files):
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
        print("!" * 60)
        return False, min_ep

    print(f"\nAll files have consistent episode count: {unique_counts[0]}")
    return True, unique_counts[0]


def load_and_aggregate_network(network_name, file_paths, max_episodes=None):
    """
    Load all PID files for one network type and aggregate.
    sarl is single-agent (player_id=0 only).

    Returns:
        score_df: DataFrame [episode_id, total_score]
        event_dfs: dict of {source: DataFrame [episode_id, count]}
    """
    pid_scores = []
    pid_events = {src: [] for src in EVENT_SOURCES}

    for fp in file_paths:
        df = pd.read_csv(fp)

        # --- Total score per episode ---
        pid_total = df.groupby('episode_id')['value'].sum().reset_index()
        pid_total.columns = ['episode_id', 'total_score']
        pid_total = pid_total.set_index('episode_id').sort_index()

        if max_episodes is not None:
            pid_total = pid_total.loc[pid_total.index <= max_episodes]
        pid_scores.append(pid_total)

        # --- Event counts per episode ---
        for src in EVENT_SOURCES:
            pid_event = df[df['source'] == src].groupby('episode_id').size().reset_index(name='count')
            pid_event = pid_event.set_index('episode_id').sort_index()
            if max_episodes is not None:
                pid_event = pid_event.loc[pid_event.index <= max_episodes]
            pid_events[src].append(pid_event)

    # --- Aggregate scores ---
    common_episodes = sorted(set().union(*[set(p.index) for p in pid_scores]))
    pid_scores_aligned = [p.reindex(common_episodes, fill_value=0) for p in pid_scores]
    score_mean = sum(pid_scores_aligned) / len(pid_scores_aligned)
    score_df = score_mean.reset_index()
    score_df.columns = ['episode_id', 'total_score']

    # --- Aggregate event counts ---
    event_dfs = {}
    for src in EVENT_SOURCES:
        pid_ev_aligned = [e.reindex(common_episodes, fill_value=0) for e in pid_events[src]]
        ev_mean = sum(pid_ev_aligned) / len(pid_ev_aligned)
        ev_df = ev_mean.reset_index()
        ev_df.columns = ['episode_id', 'count']
        event_dfs[src] = ev_df

    return score_df, event_dfs


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_total_score(network_data, smooth_window, save_path):
    """Plot 1: Total score vs episode, 4 curves."""
    _set_chinese_font()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    stats_parts = []
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]['score']
        episodes = df['episode_id'].values
        scores = df['total_score'].values

        mean_val = scores.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")

        if smooth_window > 1 and len(scores) > smooth_window:
            smoothed = smooth_curve(scores, smooth_window)
            smoothed_ep = episodes[smooth_window - 1:]
        else:
            smoothed = scores
            smoothed_ep = episodes

        sns.lineplot(x=smoothed_ep, y=smoothed,
                     label=net_name, color=NETWORK_COLORS[net_name],
                     linewidth=1.5, ax=ax)

        data_range = smoothed.max() - smoothed.min()
        band = max(data_range * 0.04, 1.0)
        ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                        alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)

    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel('总得分', fontsize=12)
    ax.set_title('总得分对比——网络结构\n（单智能体环境）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='网络结构')
    ax.set_xlim(0, None)

    stats_text = "平均总得分：\n" + "\n".join(stats_parts)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_event_count(network_data, source_key, y_label, title, smooth_window, save_path):
    """Generic event count plot: 1 subplot, 4 curves."""
    _set_chinese_font()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    stats_parts = []
    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name]['events'][source_key]
        episodes = df['episode_id'].values
        counts = df['count'].values

        mean_val = counts.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")

        if smooth_window > 1 and len(counts) > smooth_window:
            smoothed = smooth_curve(counts, smooth_window)
            smoothed_ep = episodes[smooth_window - 1:]
        else:
            smoothed = counts
            smoothed_ep = episodes

        sns.lineplot(x=smoothed_ep, y=smoothed,
                     label=net_name, color=NETWORK_COLORS[net_name],
                     linewidth=1.5, ax=ax)

        data_range = smoothed.max() - smoothed.min()
        band = max(data_range * 0.04, 0.1)
        ax.fill_between(smoothed_ep, smoothed - band, smoothed + band,
                        alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)

    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{title}——单智能体环境',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='网络结构')
    ax.set_xlim(0, None)

    stats_text = f"平均{y_label}：\n" + "\n".join(stats_parts)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close(fig)


# ==============================================================================
# Summary Statistics
# ==============================================================================

def print_summary(network_data):
    print("\n" + "=" * 70)
    print("SUMMARY: Sarl Network Architecture Comparison")
    print("=" * 70)

    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        score_df = network_data[net_name]['score']
        event_dfs = network_data[net_name]['events']

        print(f"\n{'─' * 50}")
        print(f"  {net_name}")
        print(f"{'─' * 50}")
        print(f"  Episodes: {len(score_df)}")
        print(f"  Mean Total Score: {score_df['total_score'].mean():.2f}")
        print(f"  {'Event':<22} {'Mean Count':>12}")
        print(f"  {'─' * 35}")
        for src, label in EVENT_SOURCES.items():
            mean_count = event_dfs[src]['count'].mean()
            print(f"  {label:<22} {mean_count:>12.1f}")

    best = max((network_data[n]['score']['total_score'].mean(), n)
               for n in NETWORK_DISPLAY_ORDER if n in network_data)
    print(f"\n{'=' * 70}")
    print(f"BEST total score: {best[1]} ({best[0]:.1f})")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Sarl Environment: Network Architecture Comparison Analysis")
    print("=" * 70)

    # Step 1: Discover files
    print("\n[Step 1] Discovering Sarl data files...")
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
        print("\nERROR: No Sarl data files found!")
        sys.exit(1)

    # Step 2: Validate episode consistency
    print("\n[Step 2] Validating episode consistency...")
    is_valid, common_eps = validate_episode_counts(all_network_files)
    if not is_valid:
        print("\n  → Truncating all files to common episode range.")

    smooth_window = compute_smooth_window(common_eps)
    print(f"\n  Smoothing window: {smooth_window} episodes ({SMOOTH_WINDOW_RATIO*100:.0f}% of {common_eps})")

    # Step 3: Load and aggregate
    print("\n[Step 3] Loading and aggregating data...")
    network_data = {}
    for net_name, files in all_network_files.items():
        print(f"\n  Processing {net_name} ({len(files)} files)...")
        score_df, event_dfs = load_and_aggregate_network(net_name, files, max_episodes=common_eps)
        network_data[net_name] = {'score': score_df, 'events': event_dfs}
        print(f"    Episodes: {len(score_df)}")
        print(f"    Total score range: [{score_df['total_score'].min():.1f}, {score_df['total_score'].max():.1f}]")

    # Step 4: Summary
    print_summary(network_data)

    # Step 5: Plots
    print("\n[Step 4] Generating plots...")

    # 1. Total score
    plot_total_score(
        network_data, smooth_window,
        str(SUMMARY_DIR / "sarl_01_total_score.png")
    )

    # 2-6. Event counts
    event_plots = [
        ('attack',               '攻击次数',          '智能体攻击次数',           'sarl_02_attack_count.png'),
        ('bear_damage',          '承受伤害',          '智能体承受伤害',            'sarl_03_damage_taken.png'),
        ('cause_damage_to_enemy','造成伤害',          '智能体造成伤害',            'sarl_03b_damage_dealt.png'),
        ('wall_collision',       '撞墙次数',  '智能体撞墙次数',    'sarl_04_wall_collision.png'),
        ('kill_enemy',           '击杀数',            '智能体击杀数',              'sarl_05_kill_count.png'),
        ('died',                 '死亡数',           '智能体死亡数',             'sarl_06_death_count.png'),
    ]

    for src, y_label, title, filename in event_plots:
        plot_event_count(
            network_data, src, y_label, title, smooth_window,
            str(SUMMARY_DIR / filename)
        )

    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"Output directory: {SUMMARY_DIR}")
    print(f"Generated {len(list(SUMMARY_DIR.glob('sarl_*.png')))} Sarl plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
