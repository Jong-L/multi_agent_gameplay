"""
Wall Shaping Comparison Analysis
==================================
对比不同避墙策略的表现：
1. no_shaping  — 稀疏惩罚 (0.5) 基线
2. linear      — 线性势函数 (lrrs)
3. inverse     — 反比势函数 (invprs)
4. distance    — 距离直接惩罚

使用 data_analyze/publication_plot_utils.py 的统一绘图管线。
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
TYPE_NAMES = {
    'no_shaping': '稀疏惩罚(0.5)',
    'linear':     '线性势函数',
    'inverse':    '反比势函数',
    'distance':   '距离惩罚',
}

TYPE_COLORS = {
    'no_shaping': COLORS['blue'],     # #0173B2
    'linear':     COLORS['orange'],   # #DE8F05
    'inverse':    COLORS['green'],    # #029E73
    'distance':   COLORS['pink'],     # #CC78BC
}

# ── 数据加载 ────────────────────────────────────────

def load_data_files(base_dir):
    """按策略类型分组数据文件。"""
    data_files = {'no_shaping': [], 'linear': [], 'inverse': [], 'distance': []}
    base_path = Path(base_dir)

    for file in base_path.glob("*.csv"):
        fname = file.name
        if "LiDAR_no_shaping_0p5penalty" in fname or "LiDAR_no_shaping" in fname:
            data_files['no_shaping'].append(str(file))
        elif "lrrs_" in fname and "invprs" not in fname:
            data_files['linear'].append(str(file))
        elif "invprs_" in fname:
            data_files['inverse'].append(str(file))
        elif "distance_penalty_" in fname:
            data_files['distance'].append(str(file))

    for key in data_files:
        data_files[key].sort()
    return data_files


def aggregate_data_by_type(data_files):
    """按策略类型聚合所有文件数据。"""
    aggregated = {}

    for type_name, file_list in data_files.items():
        if len(file_list) == 0:
            print(f"\n[!] 警告：未找到 {TYPE_NAMES[type_name]} 的文件")
            continue

        print(f"\n处理 {TYPE_NAMES[type_name]}（{len(file_list)} 个文件）...")

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
        aggregated[type_name] = {
            'df': combined_df,
            'max_episodes': max_episodes,
            'num_files': len(file_list),
        }

    return aggregated


def compute_episode_stats(aggregated_data):
    """计算每个回合的统计量。"""
    stats = {}

    for type_name, data in aggregated_data.items():
        df = data['df']
        max_episodes = data['max_episodes']
        num_files = data['num_files']

        all_players = sorted(df['player_id'].unique())
        episode_stats = []

        for episode in range(1, max_episodes + 1):
            ep_data = {'episode_id': episode}
            ep_df = df[df['episode_id'] == episode]

            player_scores = {}
            player_wall_counts = {}

            for player_id in all_players:
                player_df = ep_df[ep_df['player_id'] == player_id]
                total_score = player_df['value'].sum()
                player_scores[player_id] = total_score / num_files

                wall_df = player_df[player_df['source'] == 'wall_collision']
                wall_count = len(wall_df)
                player_wall_counts[player_id] = wall_count / num_files

                ep_data[f'player_{player_id}_score'] = player_scores[player_id]
                ep_data[f'player_{player_id}_wall_count'] = player_wall_counts[player_id]

            ep_data['total_score'] = sum(player_scores.values())
            ep_data['total_wall_count'] = sum(player_wall_counts.values())
            episode_stats.append(ep_data)

        stats_df = pd.DataFrame(episode_stats)
        stats[type_name] = {
            'df': stats_df,
            'players': all_players,
            'max_episodes': max_episodes,
        }

    return stats


# ── 绘图函数 ────────────────────────────────────────

def _plot_subplots(stats, col_suffix, ylabel, title, save_path, smooth_window=5):
    """通用 4 子图（各智能体）绘制。"""
    setup_style()

    first_type = list(stats.keys())[0]
    players = stats[first_type]['players']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col = f'player_{player_id}_{col_suffix}'

        for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
            if type_name not in stats:
                continue
            df = stats[type_name]['df']
            episodes = df['episode_id'].values
            values = df[col].values

            half_band = np.max(np.abs(values)) * 0.12
            errors = np.full_like(values, half_band, dtype=float)

            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, values, errors=errors,
                smooth_window=smooth_window,
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=TYPE_COLORS[type_name],
                           label=TYPE_NAMES[type_name])

        style_axes(ax, xlabel='训练回合', ylabel=ylabel,
                   title=f'智能体 {player_id}',
                   legend_kwargs={'title': '避墙策略', 'fontsize': 7})

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _plot_total(stats, col_suffix, ylabel, title, stats_label, save_path,
                smooth_window=5, loc='lower right'):
    """通用 1 图（所有智能体平均）绘制。"""
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    stats_text = f"{stats_label}:\n"

    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if type_name not in stats:
            continue
        df = stats[type_name]['df']
        episodes = df['episode_id'].values
        total = df[f'total_{col_suffix}'].values if col_suffix in ('score', 'wall_count') else df[col_suffix].values

        half_band = np.max(np.abs(total)) * 0.12
        errors = np.full_like(total, half_band, dtype=float)

        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total, errors=errors,
            smooth_window=smooth_window,
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=TYPE_COLORS[type_name],
                       label=TYPE_NAMES[type_name])

        mean_val = np.mean(total)
        stats_text += f"  {TYPE_NAMES[type_name]}: {mean_val:.2f}\n"

    style_axes(ax, xlabel='训练回合', ylabel=ylabel,
               title=title, legend_kwargs={'title': '避墙策略'})
    add_stats_box(ax, stats_text, loc=loc)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ── 对外绘图接口 ────────────────────────────────────

def plot_player_score_comparison(stats, save_path=None, smooth_window=5):
    """各智能体总得分对比（4 子图）。"""
    return _plot_subplots(stats, 'score', '平均总得分',
                          '各智能体平均总得分 —— 避墙策略对比',
                          save_path, smooth_window)


def plot_total_score_comparison(stats, save_path=None, smooth_window=5):
    """所有智能体平均总得分对比（1 图）。"""
    return _plot_total(stats, 'score', '所有智能体平均总得分',
                       '总得分对比 —— 避墙策略', '回合均分',
                       save_path, smooth_window, loc='lower left')


def plot_player_wall_count_comparison(stats, save_path=None, smooth_window=5):
    """各智能体撞墙次数对比（4 子图）。"""
    return _plot_subplots(stats, 'wall_count', '平均撞墙次数',
                          '各智能体平均撞墙次数 —— 避墙策略对比',
                          save_path, smooth_window)


def plot_total_wall_count_comparison(stats, save_path=None, smooth_window=5):
    """所有智能体平均撞墙次数对比（1 图）。"""
    return _plot_total(stats, 'wall_count', '所有智能体平均撞墙次数',
                       '撞墙次数对比 —— 避墙策略', '回合均次',
                       save_path, smooth_window, loc='upper left')


# ── 辅助函数 ────────────────────────────────────────

def save_summary_csv(stats, output_dir):
    """保存汇总 CSV。"""
    summary_dir = Path(output_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)

    for type_name, data in stats.items():
        df = data['df']
        output_path = summary_dir / f"{type_name}_episode_stats.csv"
        df.to_csv(output_path, index=False)
        print(f"已保存: {output_path}")

    summary_data = []
    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if type_name not in stats:
            continue
        df = stats[type_name]['df']
        row = {
            'type': type_name,
            'type_name': TYPE_NAMES[type_name],
            'mean_total_score': df['total_score'].mean(),
            'mean_total_wall_count': df['total_wall_count'].mean(),
            'max_episodes': stats[type_name]['max_episodes'],
        }
        for player_id in stats[type_name]['players']:
            row[f'player_{player_id}_mean_score'] = \
                df[f'player_{player_id}_score'].mean()
            row[f'player_{player_id}_mean_wall_count'] = \
                df[f'player_{player_id}_wall_count'].mean()
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = summary_dir / "wall_shaping_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"已保存: {summary_path}")
    return summary_df


def print_summary(stats):
    """打印汇总统计。"""
    print("\n" + "=" * 70)
    print("汇总统计 —— 避墙策略对比")
    print("=" * 70)

    for type_name in ['no_shaping', 'linear', 'inverse', 'distance']:
        if type_name not in stats:
            continue
        data = stats[type_name]
        df = data['df']
        players = data['players']

        print(f"\n{TYPE_NAMES[type_name]}:")
        print(f"  回合数: {data['max_episodes']}")
        print(f"  平均总得分: {df['total_score'].mean():.2f}")
        print(f"  平均撞墙次数: {df['total_wall_count'].mean():.2f}")
        print("  各智能体数据:")
        for player_id in players:
            sc = df[f'player_{player_id}_score'].mean()
            wc = df[f'player_{player_id}_wall_count'].mean()
            print(f"    智能体 {player_id}: 得分={sc:.2f}, 撞墙={wc:.2f}")


# ── 主入口 ──────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("避墙策略对比分析")
    print("=" * 70)

    print("\n[1/6] 正在加载数据文件...")
    data_files = load_data_files(base_dir)
    for tn, files in data_files.items():
        print(f"  {TYPE_NAMES[tn]}: {len(files)} 个文件")

    print("\n[2/6] 正在聚合数据...")
    aggregated = aggregate_data_by_type(data_files)
    if aggregated is None or len(aggregated) == 0:
        print("\n[错误] 数据加载失败，请检查文件路径。")
        return

    print("\n[3/6] 正在计算回合统计...")
    stats = compute_episode_stats(aggregated)

    print("\n[4/6] 正在保存汇总 CSV...")
    save_summary_csv(stats, base_dir)

    print("\n[5/6] 正在生成图表...")
    plot_player_score_comparison(stats,
                                 save_path=summary_dir / "wall_shaping_player_score_comparison.png",
                                 smooth_window=5)
    plot_total_score_comparison(stats,
                                save_path=summary_dir / "wall_shaping_total_score_comparison.png",
                                smooth_window=5)
    plot_player_wall_count_comparison(stats,
                                      save_path=summary_dir / "wall_shaping_player_wall_count_comparison.png",
                                      smooth_window=5)
    plot_total_wall_count_comparison(stats,
                                     save_path=summary_dir / "wall_shaping_total_wall_count_comparison.png",
                                     smooth_window=5)

    print("\n[6/6] 打印汇总...")
    print_summary(stats)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n图表输出目录: {summary_dir}")
    print("  - wall_shaping_player_score_comparison.png")
    print("  - wall_shaping_total_score_comparison.png")
    print("  - wall_shaping_player_wall_count_comparison.png")
    print("  - wall_shaping_total_wall_count_comparison.png")
    print("\n所有图表均为 300 DPI，可直接用于论文。")


if __name__ == "__main__":
    main()
