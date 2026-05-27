"""
Pool Evaluation Visualization
==============================
从 pool_evaluate.py 生成的 stats JSON 读取数据，生成论文级对比图表。
所有标签、标题、图例均使用中文。

Usage:
  python Python/training/pool_eval_plots.py
  python Python/training/pool_eval_plots.py --stats logs/ippo_pool_eval_stats.json --output-dir article/imgs/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Project root for data_analyze imports
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "data_analyze"))

from publication_plot_utils import (
    setup_style,
    save_figure,
    DEFAULT_PALETTE,
)

# ── 中文字体配置 ──────────────────────────────────────────────────────
def _configure_fonts() -> None:
    """确保 matplotlib 使用中文字体。"""
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })
    # 清除字体缓存，避免旧缓存导致中文不显示
    try:
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

_configure_fonts()

# ── 模型中英文映射 ────────────────────────────────────────────────────
_MODEL_LABELS_MAP = {
    "Direct":   "IPPO",
    "Pool":     "对手池博弈",
    "Average":  "虚拟博弈",
}

# ── 命令行参数 ────────────────────────────────────────────────────────
_DEFAULT_STATS = "logs/ippo_pool_eval_stats.json"
_DEFAULT_OUTPUT = "article/imgs/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default=_DEFAULT_STATS, help="stats JSON 路径")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT, help="图片输出目录")
    parser.add_argument("--prefix", default="eval_compare", help="输出文件名前缀")
    parser.add_argument("--no-show", action="store_true", help="不弹出图片窗口")
    return parser.parse_args()


def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label(name: str) -> str:
    return _MODEL_LABELS_MAP.get(name, name)


# ═══════════════════════════════════════════════════════════════════════
#  图1：胜率对比（柱状图 + Bootstrap 95% CI）
# ═══════════════════════════════════════════════════════════════════════

def plot_win_rate_comparison(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style()
    models = stats["models"]
    labels = [_label(m["model_label"]) for m in models]
    win_rates = np.array([m["win_rate"]["value"] * 100 for m in models])
    ci_lower = np.array([m["win_rate"]["lower"] * 100 for m in models])
    ci_upper = np.array([m["win_rate"]["upper"] * 100 for m in models])
    errors_lower = win_rates - ci_lower
    errors_upper = ci_upper - win_rates

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    colors = DEFAULT_PALETTE[:len(labels)]
    bars = ax.bar(x, win_rates, yerr=[errors_lower, errors_upper],
                  capsize=6, color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "ecolor": "#444444"}, width=0.55)

    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{wr:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("胜率 (%)", fontsize=13)
    ax.set_title("智能体 0 胜率对比\n（95% Bootstrap 置信区间，20 组对手）", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(ci_upper) * 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 随机基线 (4 智能体 = 25%)
    ax.axhline(y=25, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(labels) - 0.5, 25 + 1, "随机基线 (25%)", fontsize=9,
            color="#D55E00", ha="right", va="bottom", alpha=0.8)

    pairwise = stats.get("pairwise_comparisons", [])
    _add_significance_brackets(ax, pairwise, win_rates, "win_rate")

    fig.tight_layout()
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_winrate.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图2：平均奖励对比（柱状图 + Bootstrap 95% CI）
# ═══════════════════════════════════════════════════════════════════════

def plot_reward_comparison(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style()
    models = stats["models"]
    labels = [_label(m["model_label"]) for m in models]
    rewards = np.array([m["mean_reward"]["value"] for m in models])
    ci_lower = np.array([m["mean_reward"]["lower"] for m in models])
    ci_upper = np.array([m["mean_reward"]["upper"] for m in models])
    errors_lower = rewards - ci_lower
    errors_upper = ci_upper - rewards

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    colors = DEFAULT_PALETTE[:len(labels)]
    bars = ax.bar(x, rewards, yerr=[errors_lower, errors_upper],
                  capsize=6, color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "ecolor": "#444444"}, width=0.55)

    for bar, rw in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors_upper) * 0.05,
                f"{rw:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("平均回合奖励", fontsize=13)
    ax.set_title("智能体 0 平均奖励对比\n（95% Bootstrap 置信区间，20 组对手）", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    pairwise = stats.get("pairwise_comparisons", [])
    _add_significance_brackets(ax, pairwise, rewards, "reward")

    fig.tight_layout()
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_reward.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图3：逐组对手胜率分解（分组柱状图）
# ═══════════════════════════════════════════════════════════════════════

def plot_per_group_winrate(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style(font_scale=1.1)
    models = stats["models"]
    labels = [_label(m["model_label"]) for m in models]
    n_groups = models[0]["n_groups"]
    n_models = len(models)

    data = np.zeros((n_models, n_groups))
    for i, m in enumerate(models):
        for g in range(n_groups):
            data[i, g] = m["per_group_win_rates"].get(str(g), 0) * 100

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.5), 4.5))
    x = np.arange(n_groups)
    width = 0.7 / n_models
    colors = DEFAULT_PALETTE[:n_models]

    for i in range(n_models):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, data[i], width, label=labels[i],
               color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("对手组编号", fontsize=12)
    ax.set_ylabel("胜率 (%)", fontsize=12)
    ax.set_title("各对手组胜率分解", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in range(n_groups)], fontsize=8)
    ax.axhline(y=25, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(fontsize=10, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_pergroup.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图4：配对效应量（Cohen's d 水平条形图）
# ═══════════════════════════════════════════════════════════════════════

def plot_effect_sizes(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style()
    pairwise = stats.get("pairwise_comparisons", [])
    if not pairwise:
        print("[Plot] 无配对比较数据，跳过效应量图。")
        return

    labels = [f"{p['models'][0]}\nvs\n{p['models'][1]}" for p in pairwise]
    d_values = np.array([abs(p["cohens_d_reward"]) for p in pairwise])
    d_wr_values = np.array([abs(p["cohens_d_win_rate"]) for p in pairwise])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(labels))
    height = 0.35
    colors = DEFAULT_PALETTE[:2]

    ax.barh(y - height / 2, d_values, height, label="Cohen's d（奖励）",
            color=colors[0], edgecolor="white")
    ax.barh(y + height / 2, d_wr_values, height, label="Cohen's d（胜率）",
            color=colors[1], edgecolor="white")

    # 显著性标记
    for i, p in enumerate(pairwise):
        sig = "**" if p["significant_01"] else ("*" if p["significant_05"] else "")
        if sig:
            max_d = max(d_values[i], d_wr_values[i])
            ax.text(max_d + 0.05, i, sig, fontsize=14, va="center", fontweight="bold",
                    color="#D55E00")

    ax.set_yticks(y)
    ax.set_yticklabels([_label(p["models"][0]) + " vs\n" + _label(p["models"][1]) for p in pairwise],
                       fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_title("配对效应量", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, edgecolor="#cccccc")

    # 效应量参考线
    ax.axvline(x=0.2, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.text(0.2, len(labels) - 0.2, "小", fontsize=8, color="gray", va="bottom")
    ax.text(0.5, len(labels) - 0.2, "中", fontsize=8, color="gray", va="bottom")
    ax.text(0.8, len(labels) - 0.2, "大", fontsize=8, color="gray", va="bottom")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_effectsize.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数：显著性括号
# ═══════════════════════════════════════════════════════════════════════

def _add_significance_brackets(
    ax: plt.Axes,
    pairwise: list[dict],
    values: np.ndarray,
    metric: str,
) -> None:
    if len(pairwise) == 0:
        return

    label_to_idx: dict[str, int] = {}
    for pw in pairwise:
        for lbl in pw["models"]:
            if lbl not in label_to_idx:
                label_to_idx[lbl] = len(label_to_idx)

    y_max = np.max(values)
    level = 0

    for pw in pairwise:
        i = label_to_idx.get(pw["models"][0], -1)
        j = label_to_idx.get(pw["models"][1], -1)
        if i < 0 or j < 0:
            continue

        sig = ""
        p_value_key = "ttest_win_rate" if metric == "win_rate" else "ttest_reward"
        p_val = pw[p_value_key]["p_value"]
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        y = y_max * 1.05 + level * y_max * 0.12
        ax.plot([i, i, j, j], [y, y + y_max * 0.02, y + y_max * 0.02, y],
                color="#444444", linewidth=1.0, clip_on=False)
        ax.text((i + j) / 2, y + y_max * 0.03, sig, ha="center", fontsize=14,
                fontweight="bold", color="#D55E00")
        level += 1


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    stats_path = pathlib.Path(args.stats)
    if not stats_path.exists():
        print(f"[Error] 未找到评估统计文件: {stats_path}")
        print(f"  请先运行 pool_evaluate.py 生成该文件。")
        sys.exit(1)

    stats = load_stats(str(stats_path))
    print(f"已加载: {stats_path} ({len(stats['models'])} 个模型)")

    output_dir = args.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    plot_win_rate_comparison(stats, output_dir, prefix)
    plot_reward_comparison(stats, output_dir, prefix)
    plot_per_group_winrate(stats, output_dir, prefix)
    plot_effect_sizes(stats, output_dir, prefix)

    print(f"\n全部图片已保存至 {output_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
