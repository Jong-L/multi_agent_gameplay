"""
Wall Penalty Analysis Script
================================
分析撞墙惩罚值对智能体总得分的影响。
三种惩罚力度：0.05 / 0.5 / 5.0

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

# ── 配色 ────────────────────────────────────────────
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

# ── 数据加载 ────────────────────────────────────────

def load_data_files(base_dir):
    """按惩罚类型分组数据文件。"""
    data_files = {key: [] for key in PENALTY_COLORS.keys()}
    base_path = Path(base_dir)
    for pattern in [f"{key}_*.csv" for key in PENALTY_COLORS.keys()]:
        for file in base_path.glob(pattern):
            fname = file.name
            for key in PENALTY_COLORS.keys():
                if key in fname:
                    data_files[key].append(str(file))
                    break
    return data_files


def aggregate_scores(file_list, sources=None):
    """
    聚合多个文件的数据。
    返回：(mean_df, sem_df)
    """
    all_records = []
    for fp in file_list:
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            if sources:
                df = df[df['source'].isin(sources)]
            all_records.append(df)
        except Exception as e:
            print(f"  [WARN] 无法读取 {fp}: {e}")
            continue

    if not all_records:
        return None, None

    combined = pd.concat(all_records, ignore_index=True)

    if 'source' in combined.columns:
        agg = combined.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        agg.rename(columns={'value': 'score'}, inplace=True)
    else:
        agg = combined.copy()

    pivot = agg.pivot_table(index='episode_id', columns='player_id',
                            values='score', fill_value=0)
    pivot['total_avg'] = pivot.sum(axis=1)
    pivot = pivot.reset_index()

    return pivot, pivot  # SEM placeholder


# ── 绘图 ────────────────────────────────────────────

def plot_per_player(data, save_path, smooth_window=5):
    """各智能体总得分对比（4 子图）。"""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('各智能体平均总得分 —— 撞墙惩罚对比',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for idx, pid in enumerate([0, 1, 2, 3]):
        ax = axes[idx]
        for ptype, color in PENALTY_COLORS.items():
            if ptype not in data or data[ptype] is None:
                continue
            df = data[ptype][0]
            col = f'player_{pid}'
            if col not in df.columns:
                continue
            episodes = df['episode_id'].values
            scores = df[col].values

            half_band = np.max(np.abs(scores)) * 0.08
            errors = np.full_like(scores, half_band, dtype=float)

            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, scores, errors=errors,
                smooth_window=smooth_window,
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=color, label=PENALTY_NAMES[ptype])

        style_axes(ax, xlabel='训练回合', ylabel='平均总得分',
                   title=f'智能体 {pid}',
                   legend_kwargs={'title': '惩罚力度', 'fontsize': 8})

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total(data, save_path, smooth_window=5):
    """所有智能体平均总得分对比（1 图）。"""
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    stats_text = "回合均分:\n"

    for ptype, color in PENALTY_COLORS.items():
        if ptype not in data or data[ptype] is None:
            continue
        df = data[ptype][0]
        total = df['total_avg'].values
        episodes = df['episode_id'].values

        half_band = np.max(np.abs(total)) * 0.08
        errors = np.full_like(total, half_band, dtype=float)

        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total, errors=errors,
            smooth_window=smooth_window,
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=color, label=PENALTY_NAMES[ptype])

        mean_val = np.mean(total)
        stats_text += f"  {PENALTY_NAMES[ptype]}: {mean_val:.2f}\n"

    style_axes(ax, xlabel='训练回合', ylabel='所有智能体平均总得分',
               title='总得分对比 —— 撞墙惩罚',
               legend_kwargs={'title': '惩罚力度'})
    add_stats_box(ax, stats_text)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ── 主入口 ──────────────────────────────────────────

def main():
    print("=" * 60)
    print("  撞墙惩罚分析")
    print("=" * 60)

    print("\n[1/3] 正在加载数据...")
    base_dir = Path(__file__).parent
    data_files = load_data_files(base_dir)

    data = {}
    for ptype, files in data_files.items():
        print(f"  {PENALTY_NAMES[ptype]}: {len(files)} 个文件")
        if files:
            data[ptype] = aggregate_scores(files)

    print("\n[2/3] 正在生成图表...")
    summary_dir = Path(__file__).parent / "summary"
    summary_dir.mkdir(exist_ok=True)

    plot_per_player(data, str(summary_dir / "per_player_comparison.png"))
    plot_total(data, str(summary_dir / "total_comparison.png"))

    print("\n[3/3] 完成！")
    print(f"\n  图表输出目录: {summary_dir}")


if __name__ == "__main__":
    main()
