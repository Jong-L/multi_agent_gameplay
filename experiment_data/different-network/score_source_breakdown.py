"""
得分来源拆解分析
按三类得分来源分析随回合数的变化趋势：

  1. 攻击与击杀得分（正向）
     - bear_damage (+10)
     - cause_damage_to_player (+10)
     - cause_damage_to_enemy (+7)
     - kill_enemy (+14)
     - kill_player (+20)
     - attack (-0.02，计入成本)

  2. 受到攻击与死亡惩罚（负向）
     - died (-1.5)
     - wall_collision (-0.5)  ← 单独拆出为第3类

  3. 撞墙惩罚
     - wall_collision (-0.5)

每类生成 2 张图：
  A. 每个玩家平均得分 vs 回合数（2×2 子图，4 条曲线）
  B. 所有玩家平均得分总和 vs 回合数（1 图，4 条曲线）
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import sys
import warnings
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = Path(__file__).parent
SUMMARY_DIR = DATA_DIR / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 8

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# --- Font setup for CJK characters ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

NETWORK_COLORS = {
    'MLP':            '#0173B2',
    '分段MLP':  '#DE8F05',
    'GRU-MLP 96':     '#029E73',
    'GRU-MLP 128':    '#D55E00',
}
NETWORK_DISPLAY_ORDER = ['MLP', '分段MLP', 'GRU-MLP 96', 'GRU-MLP 128']

NETWORK_PATTERNS = {
    'MLP':            's1_mlp_*.csv',
    '分段MLP':  's1_seg_mlp_*.csv',
    'GRU-MLP 96':     's1_96_gru_mlp_*.csv',
    'GRU-MLP 128':    's1_128_gru_mlp_*.csv',
}
PLAYER_IDS = [0, 1, 2, 3]

# --- 三类得分来源 ---
SOURCE_GROUPS = {
    'combat_kill': {
        'label': '战斗与击杀得分',
        'sources': ['bear_damage', 'cause_damage_to_player',
                    'cause_damage_to_enemy', 'kill_enemy', 'kill_player'],
        'filename_prefix': '01',
    },
    'death_penalty': {
        'label': '死亡与伤害惩罚',
        'sources': ['died'],
        'filename_prefix': '02',
    },
    'wall_penalty': {
        'label': '撞墙惩罚',
        'sources': ['wall_collision'],
        'filename_prefix': '03',
    },
}

# ==============================================================================
# Utilities
# ==============================================================================

def smooth_curve(y, window=SMOOTH_WINDOW):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def load_and_aggregate_by_source(file_paths):
    """
    为每个 PID 文件计算每回合每玩家的三类得分，
    再跨 PID 取均值。
    
    返回 dict:
      { group_key: DataFrame([episode_id, player_0..3, total]) }
    """
    # Collect per-PID per-group scores
    all_groups = {k: [] for k in SOURCE_GROUPS}

    for fp in file_paths:
        df = pd.read_csv(fp)

        for grp_key, grp_cfg in SOURCE_GROUPS.items():
            grp_df = df[df['source'].isin(grp_cfg['sources'])]
            scores = grp_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
            scores.columns = ['episode_id', 'player_id', 'score']
            all_groups[grp_key].append(scores)

    # Pivot and average across PIDs
    episodes = sorted(set().union(
        *[set(g['episode_id'].unique()) for gl in all_groups.values() for g in gl]
    ))

    result = {}
    for grp_key, pid_list in all_groups.items():
        pivots = []
        for pid_df in pid_list:
            pivot = pid_df.pivot_table(
                index='episode_id', columns='player_id',
                values='score', fill_value=0
            )
            for p in PLAYER_IDS:
                if p not in pivot.columns:
                    pivot[p] = 0
            pivot = pivot.reindex(episodes, fill_value=0)
            pivot = pivot[sorted(pivot.columns)]
            pivots.append(pivot)

        mean_pivot = sum(pivots) / len(pivots)
        mean_pivot = mean_pivot.reset_index()
        mean_pivot.columns = ['episode_id'] + [f'player_{p}' for p in sorted(mean_pivot.columns[1:])]
        player_cols = [c for c in mean_pivot.columns if c.startswith('player_')]
        mean_pivot['total'] = mean_pivot[player_cols].sum(axis=1)
        result[grp_key] = mean_pivot

    return result


def make_band(y_values):
    """Dynamic band = 4% of data range, minimum 1."""
    rng = y_values.max() - y_values.min()
    return max(rng * 0.04, 1.0)


# ==============================================================================
# Plotting
# ==============================================================================

def plot_per_player(grp_key, grp_cfg, network_data, save_path):
    """2×2 subplots, one per player, 4 network curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(
        f'各智能体平均{grp_cfg["label"]}随回合变化',
        fontsize=16, fontweight='bold', y=1.01
    )
    axes = axes.flatten()

    for p_idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[p_idx]
        player_col = f'player_{player_id}'

        for net_name in NETWORK_DISPLAY_ORDER:
            if net_name not in network_data:
                continue
            df = network_data[net_name][grp_key]
            episodes = df['episode_id'].values
            scores = df[player_col].values

            if SMOOTH_WINDOW > 1 and len(scores) > SMOOTH_WINDOW:
                sm = smooth_curve(scores, SMOOTH_WINDOW)
                sm_ep = episodes[SMOOTH_WINDOW - 1:]
            else:
                sm, sm_ep = scores, episodes

            sns.lineplot(x=sm_ep, y=sm, label=net_name,
                         color=NETWORK_COLORS[net_name],
                         linewidth=1.5, ax=ax)

            band = make_band(sm)
            ax.fill_between(sm_ep, sm - band, sm + band,
                            alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)

        ax.set_xlabel('回合', fontsize=11)
        ax.set_ylabel(grp_cfg['label'], fontsize=11)
        ax.set_title(f'智能体 {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        ax.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def plot_total(grp_key, grp_cfg, network_data, save_path):
    """1 subplot, 4 network curves, sum across all players."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    stats_parts = []

    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        df = network_data[net_name][grp_key]
        episodes = df['episode_id'].values
        totals = df['total'].values

        mean_val = totals.mean()
        stats_parts.append(f"{net_name}: {mean_val:.1f}")

        if SMOOTH_WINDOW > 1 and len(totals) > SMOOTH_WINDOW:
            sm = smooth_curve(totals, SMOOTH_WINDOW)
            sm_ep = episodes[SMOOTH_WINDOW - 1:]
        else:
            sm, sm_ep = totals, episodes

        sns.lineplot(x=sm_ep, y=sm, label=net_name,
                     color=NETWORK_COLORS[net_name],
                     linewidth=1.5, ax=ax)

        band = make_band(sm)
        ax.fill_between(sm_ep, sm - band, sm + band,
                        alpha=0.25, color=NETWORK_COLORS[net_name], zorder=1)

    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel(f'总体{grp_cfg["label"]}', fontsize=12)
    ax.set_title(
        f'总{grp_cfg["label"]}对比——网络结构',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='网络结构')
    ax.set_xlim(0, None)

    stats_text = f"平均{grp_cfg['label']}：\n" + "\n".join(stats_parts)
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# Summary
# ==============================================================================

def print_summary(network_data):
    print("\n" + "=" * 70)
    print("SUMMARY: Score Source Breakdown by Network")
    print("=" * 70)

    for net_name in NETWORK_DISPLAY_ORDER:
        if net_name not in network_data:
            continue
        print(f"\n{'─' * 60}")
        print(f"  {net_name}")
        print(f"{'─' * 60}")
        print(f"  {'Source':<30} {'Player Avg':>12} {'Total Avg':>12}")
        print(f"  {'-' * 55}")

        for grp_key, grp_cfg in SOURCE_GROUPS.items():
            df = network_data[net_name][grp_key]
            player_means = [df[f'player_{p}'].mean() for p in PLAYER_IDS]
            total_mean = df['total'].mean()
            avg_str = f"{np.mean(player_means):>12.2f}"
            print(f"  {grp_cfg['label']:<30} {avg_str}{total_mean:>12.2f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Score Source Breakdown Analysis")
    print("=" * 70)

    # Discover files
    print("\n[1/4] Discovering files...")
    all_network_files = {}
    for net_name, pattern in NETWORK_PATTERNS.items():
        files = sorted(glob.glob(str(DATA_DIR / pattern)))
        if files:
            all_network_files[net_name] = files
            print(f"  {net_name}: {len(files)} files")
        else:
            print(f"  {net_name}: NOT FOUND!")

    if not all_network_files:
        print("ERROR: No files found!")
        sys.exit(1)

    # Load & aggregate
    print("\n[2/4] Loading & aggregating by source group...")
    network_data = {}
    for net_name, files in all_network_files.items():
        print(f"  {net_name}...")
        network_data[net_name] = load_and_aggregate_by_source(files)

    # Print summary
    print_summary(network_data)

    # Generate plots
    print("\n[3/4] Generating plots...")

    for grp_key, grp_cfg in SOURCE_GROUPS.items():
        prefix = grp_cfg['filename_prefix']
        label_short = grp_cfg['label'].replace(' ', '_').replace('&', 'and').lower()

        # Per-player 2×2
        plot_per_player(
            grp_key, grp_cfg, network_data,
            str(SUMMARY_DIR / f"{prefix}_per_player_{label_short}.png")
        )

        # Total 1-plot
        plot_total(
            grp_key, grp_cfg, network_data,
            str(SUMMARY_DIR / f"{prefix}_total_{label_short}.png")
        )

    print(f"\n[4/4] Done! {len(list(SUMMARY_DIR.glob('*.png')))} PNG files in {SUMMARY_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
