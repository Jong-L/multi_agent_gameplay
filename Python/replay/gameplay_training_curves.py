"""
多智能体博弈训练曲线对比图
=============================
绘制 IPPO、对手池博弈、对手策略平均化三种方法的智能体 0 训练奖励曲线。
使用 publication_plot_utils 统一绘图管线。

Usage:
  conda activate gdrl
  python Python/replay/gameplay_training_curves.py
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 挂载项目根目录 ──
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "data_analyze"))

from publication_plot_utils import (
    setup_style, save_figure, COLORS, DEFAULT_PALETTE,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
)

# ── 中文字体 ──
import matplotlib
matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
})

# ── 文件路径 ──
DATA_DIR = _PROJECT_ROOT / "logs"
OUTPUT_DIR = _PROJECT_ROOT / "article" / "imgs"

FILES = {
    "IPPO":       DATA_DIR / "cleanrl_ippo_ippo_bootstrap_direct_direct.csv",
    "对手池博弈": DATA_DIR / "cleanrl_ippo_ippo_pool_agent0_extra.csv",
    "平均策略博弈": DATA_DIR / "cleanrl_ippo_ippo_average_opponent_agent0.csv",
}

# 颜色映射（与 pool_eval_plots.py 保持一致）
METHOD_COLORS = {
    "IPPO":           DEFAULT_PALETTE[0],  # blue
    "对手池博弈":     DEFAULT_PALETTE[1],  # orange
    "平均策略博弈": DEFAULT_PALETTE[2],  # green
}


def load_csv(csv_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """读取原始 CSV 数据，不做分箱聚合。"""
    df = pd.read_csv(csv_path)
    step = df["Step"].values.astype(float)
    value = df["Value"].values.astype(float)
    return step, value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载原始数据 ──
    all_data = {}
    for label, path in FILES.items():
        x, y = load_csv(path)
        print(f"[{label}] {len(x)} 个原始点, step [{x[0]:.0f}, {x[-1]:.0f}], "
              f"reward [{y.min():.1f}, {y.max():.1f}]")
        all_data[label] = (x, y)

    # ── IPPO 左移对齐：IPPO 实际从 ~1M 步开始，与其他方法起点对齐 ──
    ippo_offset = all_data["IPPO"][0][0]
    x_i, y_i = all_data["IPPO"]
    x_i = x_i - ippo_offset
    all_data["IPPO"] = (x_i, y_i)
    print(f"[对齐] IPPO 左移 {ippo_offset:.0f} 步, 新范围 [{x_i[0]:.0f}, {x_i[-1]:.0f}]")

    # 确定全局 y 上限（用于固定误差半带 12%）
    global_ymax = max(y.max() for _, y in all_data.values())
    half_band = global_ymax * 0.12

    # ── 画图 ──
    setup_style(font_scale=1.25)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    stats_lines = []
    for label in ("IPPO", "对手池博弈", "平均策略博弈"):
        x, y = all_data[label]
        e = np.full_like(y, half_band)

        x_s, y_s, y_l, y_u = prepare_curve(
            x, y, errors=e, smooth_window=5,
        )
        plot_with_fill(
            ax, x_s, y_s, y_l, y_u,
            color=METHOD_COLORS[label], label=label,
            linewidth=2.0, alpha=0.18,
        )

        # 统计信息（仅用 ≤9M 步的数据，与截断范围一致）
        stable_mask = (x >= x.max() * 0.6) & (x <= 9_000_000)
        if stable_mask.sum() > 1:
            stable_mean = y[stable_mask].mean()
            stable_std = y[stable_mask].std(ddof=1)
        else:
            stable_mean, stable_std = 0, 0
        stats_lines.append(f"{label}: 均值={stable_mean:.0f} ± {stable_std:.0f}")

    # ── 截断 x 轴至 9M ──
    ax.set_xlim(0, 9_000_000)

    # ── 轴标签与标题 ──
    ax.set_xlabel("训练步数 (百万步)", fontsize=12)
    ax.set_ylabel("智能体 0 回合奖励", fontsize=12)
    ax.set_title("多智能体博弈训练曲线：三种训练方法对比", fontsize=14, fontweight="bold")

    # 横轴刻度：转换为百万步
    x_ticks_raw = np.arange(0, 10_000_000, 1_000_000)
    ax.set_xticks(x_ticks_raw)
    ax.set_xticklabels([f"{t/1e6:.0f}" for t in x_ticks_raw])

    legend = ax.legend(fontsize=11, frameon=True, edgecolor="#cccccc",
                       loc="lower right")

    # 添加统计信息框
    stats_text = "\n".join(stats_lines)
    add_stats_box(ax, stats_text, loc="upper left", fontsize=9.5)

    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out_path = str(OUTPUT_DIR / "gameplay_training_curves.png")
    save_figure(fig, out_path)
    print(f"\n图片已保存至: {out_path}")


if __name__ == "__main__":
    main()
