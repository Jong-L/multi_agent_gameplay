"""
Wall Penalty vs Ball Reward Analysis
========================================
分析撞墙惩罚值对智能体采球得分和撞墙次数的影响。
三种惩罚力度：0.05 / 0.5 / 5.0

使用 data_analyze/publication_plot_utils.py 的统一绘图管线。
只统计 collect_ball_A 和 collect_ball_B 作为采球得分。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加 data_analyze 目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'data_analyze'))

from publication_plot_utils import (
    setup_style, save_figure,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
    COLORS,
)

# ── 配置 ────────────────────────────────────────────
PENALTY_COLORS = {
    '00p5': COLORS['blue'],     # #0173B2
    '0p5':  COLORS['orange'],   # #DE8F05
    '5':    COLORS['green'],    # #029E73
}

PENALTY_NAMES = {
    '00p5': '惩罚值=0.05',
    '0p5':  '惩罚值=0.5',
    '5':    '惩罚值=5.0',
}

PENALTY_VALUES = {
    '00p5': 0.05,
    '0p5':  0.5,
    '5':    5.0,
}

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']

# ── 数据加载与处理 ──────────────────────────────────

def load_data_files(base_dir):
    """按惩罚类型分组数据文件。"""
    data_files = {'00p5': [], '0p5': [], '5': []}
    base_path = Path(base_dir)

    for file in base_path.glob("*.csv"):
        fname = file.name
        if "wall_penalty_analysis.py" in fname:
            continue
        if "00p5penalty" in fname:
            data_files['00p5'].append(str(file))
        elif "0p5penalty" in fname:
            data_files['0p5'].append(str(file))
        elif "5penalty" in fname and "00p5" not in fname and "0p5" not in fname:
            data_files['5'].append(str(file))

    for key in data_files:
        data_files[key].sort()
    return data_files


def aggregate_data_by_penalty(data_files):
    """按惩罚类型聚合所有文件数据。"""
    aggregated = {}

    for penalty_type, file_list in data_files.items():
        print(f"\n处理 {PENALTY_NAMES[penalty_type]}（{len(file_list)} 个文件）...")

        if len(file_list) == 0:
            print(f"  [!] 警告：未找到 {penalty_type} 的文件")
            continue

        all_episodes = []
        episode_counts = {}

        for fp in file_list:
            df = pd.read_csv(fp)
            max_ep = df['episode_id'].max() if len(df) > 0 else 0
            episode_counts[Path(fp).name] = max_ep
            all_episodes.append(df)

        unique_counts = set(episode_counts.values())
        if len(unique_counts) > 1:
            print(f"  [!] 警告：回合数不一致: {episode_counts}")
            print("  [!] 分析中止——所有文件必须有相同的回合数")
            return None

        max_episodes = list(unique_counts)[0]
        print(f"  [OK] 一致：{max_episodes} 回合")

        combined_df = pd.concat(all_episodes, ignore_index=True)
        aggregated[penalty_type] = {
            'df': combined_df,
            'max_episodes': max_episodes,
            'num_files': len(file_list),
        }

    return aggregated


def compute_episode_stats(aggregated_data):
    """计算每个回合的统计量。"""
    stats = {}

    for penalty_type, data in aggregated_data.items():
        df = data['df']
        max_episodes = data['max_episodes']
        num_files = data['num_files']

        all_players = sorted(df['player_id'].unique())
        episode_stats = []

        for episode in range(1, max_episodes + 1):
            ep_data = {'episode_id': episode}
            ep_df = df[df['episode_id'] == episode]

            player_ball_scores = {}
            player_wall_counts = {}

            for player_id in all_players:
                player_df = ep_df[ep_df['player_id'] == player_id]

                ball_df = player_df[player_df['source'].isin(BALL_SOURCES)]
                ball_score = ball_df['value'].sum()
                player_ball_scores[player_id] = ball_score / num_files

                wall_df = player_df[player_df['source'] == 'wall_collision']
                wall_count = len(wall_df)
                player_wall_counts[player_id] = wall_count / num_files

                ep_data[f'player_{player_id}_ball_score'] = player_ball_scores[player_id]
                ep_data[f'player_{player_id}_wall_count'] = player_wall_counts[player_id]

            ep_data['total_ball_score'] = sum(player_ball_scores.values())
            ep_data['total_wall_count'] = sum(player_wall_counts.values())
            episode_stats.append(ep_data)

        stats_df = pd.DataFrame(episode_stats)
        stats[penalty_type] = {
            'df': stats_df,
            'players': all_players,
            'max_episodes': max_episodes,
        }

    return stats


# ── 绘图函数 ────────────────────────────────────────

def plot_player_ball_score(stats, save_path=None, smooth_window=5):
    """各智能体采球得分对比（4 子图）。"""
    setup_style()

    first_penalty = list(stats.keys())[0]
    players = stats[first_penalty]['players']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('各智能体平均采球得分 vs 撞墙惩罚',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_ball_score'

        for penalty_type in ['00p5', '0p5', '5']:
            if penalty_type not in stats:
                continue
            df = stats[penalty_type]['df']
            episodes = df['episode_id'].values
            scores = df[col_name].values

            half_band = np.max(np.abs(scores)) * 0.08
            errors = np.full_like(scores, half_band, dtype=float)

            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, scores, errors=errors,
                smooth_window=smooth_window,
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=PENALTY_COLORS[penalty_type],
                           label=PENALTY_NAMES[penalty_type])

        style_axes(ax, xlabel='训练回合', ylabel='平均采球得分',
                   title=f'智能体 {player_id}',
                   legend_kwargs={'title': '惩罚力度', 'fontsize': 8})

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total_ball_score(stats, save_path=None, smooth_window=5):
    """所有智能体平均采球得分对比（1 图）。"""
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    stats_text = "回合均分:\n"

    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        df = stats[penalty_type]['df']
        episodes = df['episode_id'].values
        total_scores = df['total_ball_score'].values

        half_band = np.max(np.abs(total_scores)) * 0.08
        errors = np.full_like(total_scores, half_band, dtype=float)

        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total_scores, errors=errors,
            smooth_window=smooth_window,
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=PENALTY_COLORS[penalty_type],
                       label=PENALTY_NAMES[penalty_type])

        avg_score = total_scores.mean()
        stats_text += f"  {PENALTY_NAMES[penalty_type]}: {avg_score:.2f}\n"

    style_axes(ax, xlabel='训练回合', ylabel='所有智能体平均采球得分',
               title='所有智能体平均采球得分对比 —— 撞墙惩罚',
               legend_kwargs={'title': '惩罚力度'})
    add_stats_box(ax, stats_text)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_player_wall_count(stats, save_path=None, smooth_window=5):
    """各智能体撞墙次数对比（4 子图）。"""
    setup_style()

    first_penalty = list(stats.keys())[0]
    players = stats[first_penalty]['players']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('各智能体平均撞墙次数 vs 惩罚力度',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_wall_count'

        for penalty_type in ['00p5', '0p5', '5']:
            if penalty_type not in stats:
                continue
            df = stats[penalty_type]['df']
            episodes = df['episode_id'].values
            wall_counts = df[col_name].values

            half_band = np.max(np.abs(wall_counts)) * 0.08
            errors = np.full_like(wall_counts, half_band, dtype=float)

            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, wall_counts, errors=errors,
                smooth_window=smooth_window,
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=PENALTY_COLORS[penalty_type],
                           label=PENALTY_NAMES[penalty_type])

        style_axes(ax, xlabel='训练回合', ylabel='平均撞墙次数',
                   title=f'智能体 {player_id}',
                   legend_kwargs={'title': '惩罚力度', 'fontsize': 8})

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total_wall_count(stats, save_path=None, smooth_window=5):
    """所有智能体平均撞墙次数对比（1 图）。"""
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    stats_text = "回合均次:\n"

    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        df = stats[penalty_type]['df']
        episodes = df['episode_id'].values
        total_wall_counts = df['total_wall_count'].values

        half_band = np.max(np.abs(total_wall_counts)) * 0.08
        errors = np.full_like(total_wall_counts, half_band, dtype=float)

        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total_wall_counts, errors=errors,
            smooth_window=smooth_window,
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=PENALTY_COLORS[penalty_type],
                       label=PENALTY_NAMES[penalty_type])

        avg_count = total_wall_counts.mean()
        stats_text += f"  {PENALTY_NAMES[penalty_type]}: {avg_count:.1f}\n"

    style_axes(ax, xlabel='训练回合', ylabel='所有智能体平均撞墙次数',
               title='所有智能体平均撞墙次数对比 —— 惩罚力度',
               legend_kwargs={'title': '惩罚力度'})
    add_stats_box(ax, stats_text, loc='lower right')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ── 辅助函数 ────────────────────────────────────────

def save_summary_csv(stats, output_dir):
    """保存汇总 CSV。"""
    summary_dir = Path(output_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)

    for penalty_type, data in stats.items():
        df = data['df']
        output_path = summary_dir / f"{penalty_type}_ball_episode_stats.csv"
        df.to_csv(output_path, index=False)
        print(f"已保存: {output_path}")

    summary_data = []
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        df = stats[penalty_type]['df']
        row = {
            'penalty_type': penalty_type,
            'penalty_value': PENALTY_VALUES[penalty_type],
            'mean_total_ball_score': df['total_ball_score'].mean(),
            'mean_total_wall_count': df['total_wall_count'].mean(),
            'max_episodes': stats[penalty_type]['max_episodes'],
        }
        for player_id in stats[penalty_type]['players']:
            row[f'player_{player_id}_mean_ball_score'] = \
                df[f'player_{player_id}_ball_score'].mean()
            row[f'player_{player_id}_mean_wall_count'] = \
                df[f'player_{player_id}_wall_count'].mean()
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = summary_dir / "ball_overall_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"已保存: {summary_path}")
    return summary_df


def print_summary(stats):
    """打印汇总统计。"""
    print("\n" + "=" * 70)
    print("采球得分汇总")
    print("=" * 70)

    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        data = stats[penalty_type]
        df = data['df']
        players = data['players']

        print(f"\n{PENALTY_NAMES[penalty_type]}:")
        print(f"  回合数: {data['max_episodes']}")
        print(f"  场均采球得分: {df['total_ball_score'].mean():.2f}")
        print(f"  场均撞墙次数: {df['total_wall_count'].mean():.2f}")
        print("  各智能体数据:")
        for player_id in players:
            sc = df[f'player_{player_id}_ball_score'].mean()
            wc = df[f'player_{player_id}_wall_count'].mean()
            print(f"    智能体 {player_id}: 采球得分={sc:.2f}, 撞墙={wc:.2f}")


# ── 主入口 ──────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("撞墙惩罚 vs 采球得分分析")
    print("=" * 70)
    print(f"统计的球得分来源: {BALL_SOURCES}")

    print("\n[1/6] 正在加载数据文件...")
    data_files = load_data_files(base_dir)
    for pt, files in data_files.items():
        print(f"  {PENALTY_NAMES[pt]}: {len(files)} 个文件")

    print("\n[2/6] 正在聚合数据...")
    aggregated = aggregate_data_by_penalty(data_files)
    if aggregated is None or len(aggregated) == 0:
        print("\n[错误] 数据加载失败，请检查文件路径。")
        return

    print("\n[3/6] 正在计算回合统计...")
    stats = compute_episode_stats(aggregated)

    print("\n[4/6] 正在保存汇总 CSV...")
    summary_df = save_summary_csv(stats, base_dir)

    print("\n[5/6] 正在生成图表...")
    plot_player_ball_score(stats,
                           save_path=summary_dir / "player_ball_score_comparison.png",
                           smooth_window=5)
    plot_total_ball_score(stats,
                          save_path=summary_dir / "total_ball_score_comparison.png",
                          smooth_window=5)
    plot_player_wall_count(stats,
                           save_path=summary_dir / "player_wall_count_comparison.png",
                           smooth_window=5)
    plot_total_wall_count(stats,
                          save_path=summary_dir / "total_wall_count_comparison.png",
                          smooth_window=5)

    print("\n[6/6] 打印汇总...")
    print_summary(stats)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n图表输出目录: {summary_dir}")
    print("  - player_ball_score_comparison.png")
    print("  - total_ball_score_comparison.png")
    print("  - player_wall_count_comparison.png")
    print("  - total_wall_count_comparison.png")
    print("\n所有图表均为 300 DPI，可直接用于论文。")

if __name__ == "__main__":
    main()
