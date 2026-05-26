"""
Plot Template — Publication-Quality Academic Figure
=====================================================
复制此文件 → 修改 DATA SECTION → 运行 → 产出论文级矢量图。

依赖: publication_plot_utils.py (同目录)
需要: scipy (高斯滤波 + 样条插值), 无 scipy 时自动回退滑动平均

---------------------------------------------------------------------
使用方式:
  1. 在 DATA SECTION 填入你的数据路径/DataFrame
  2. 在 PLOT SECTION 调整图名、轴标签、配色
  3. 运行: python plot_template.py
---------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ── 加载共享工具 ─────────────────────────────────────────
from publication_plot_utils import (
    setup_style, save_figure,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
    COLORS, DEFAULT_PALETTE,
)


# ============================================================
# DATA SECTION — 修改这里加载你的数据
# ============================================================
def load_data():
    """
    在此函数中加载和预处理数据。
    返回一个字典: { '方案名': {'x': ..., 'y': ..., 'err': ...}, ... }
    
    x:  1D array — x 轴 (如 episode_id)
    y:  1D array — 均值 (如 average reward)
    err: 1D array — 误差带 (如 SEM, 标准差; 填 zeros_like(y) 则不显示误差带)
    """
    
    # === 示例: 从 CSV 加载 =================================
    # df = pd.read_csv("your_data.csv")
    # x  = df['episode_id'].values
    # y  = df['mean_reward'].values
    # err = df['sem'].values
    
    # === 示例: 生成演示数据 (请替换) =======================
    np.random.seed(42)
    n = 80
    x = np.arange(1, n + 1)
    
    data = {}
    for i, (name, color) in enumerate(zip(
        ['Method A', 'Method B', 'Method C'],
        [COLORS['blue'], COLORS['orange'], COLORS['green']]
    )):
        # 模拟强化学习训练曲线: 上升 + 噪声
        base = 50 + 150 * (1 - np.exp(-x / 20)) + i * 20
        noise = np.random.randn(n) * 15
        y = base + noise
        err = np.ones(n) * 8 + np.random.randn(n) * 2  # 模拟 SEM
        
        data[name] = {
            'x': x,
            'y': y,
            'err': err,
            'color': color,
        }
    
    return data


# ============================================================
# PLOT SECTION — 修改这里调整图表样式
# ============================================================
# ── 图表全局设置 ─────────────────────────────────────────
FIGURE_TITLE = "Training Curves Comparison"    # 图表标题
XLABEL = "Episode"                              # x 轴标签
YLABEL = "Average Reward"                       # y 轴标签
SAVE_PATH = "plot_output.png"                   # 保存路径
SMOOTH_WINDOW = 5                               # 平滑强度 (越大越平滑, 建议 3-10)
FIG_SIZE = (10, 6)                              # 图片尺寸 (宽, 高) 英寸

# ── 其他可选参数 ─────────────────────────────────────────
SHOW_STATS_BOX = True    # 是否显示统计信息框
SHOW_LEGEND = True       # 是否显示图例
YLIM_BOTTOM = None       # y 轴下限 (None=自动, 0=从零开始)


def build_plot(data):
    """
    核心画图函数。
    如果需要多子图或其他布局，在此函数中修改。
    """
    setup_style(context='paper', font_scale=1.3)
    
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=150)
    
    stats_lines = []
    
    for name, d in data.items():
        x_smooth, y_smooth, y_lower, y_upper = prepare_curve(
            d['x'], d['y'], d['err'],
            smooth_window=SMOOTH_WINDOW,
        )
        
        color = d.get('color', DEFAULT_PALETTE[0])
        
        plot_with_fill(ax, x_smooth, y_smooth, y_lower, y_upper,
                       color=color, label=name)
        
        if SHOW_STATS_BOX:
            mean_val = np.nanmean(d['y'])
            max_val  = np.nanmax(d['y'])
            stats_lines.append(f"{name}: μ={mean_val:.1f}, max={max_val:.1f}")
    
    # ── 样式设置 ───────────────────────────────────────
    ylim = None
    if YLIM_BOTTOM is not None:
        ylim = (YLIM_BOTTOM, None)
    
    style_axes(ax, xlabel=XLABEL, ylabel=YLABEL, title=FIGURE_TITLE,
               xlim=(0, None), ylim=ylim,
               legend_kwargs={'title': 'Method'} if SHOW_LEGEND else None)
    
    if SHOW_STATS_BOX and stats_lines:
        add_stats_box(ax, "\n".join(stats_lines))
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"  {FIGURE_TITLE}")
    print("=" * 60)
    
    print("\n[1/3] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} curves")
    
    print("\n[2/3] Building plot...")
    fig = build_plot(data)
    
    print("\n[3/3] Saving...")
    save_figure(fig, SAVE_PATH)
    
    print("\n" + "=" * 60)
    print(f"  Done! → {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
