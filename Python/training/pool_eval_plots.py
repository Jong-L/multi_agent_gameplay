"""
Pool Evaluation Visualization (Reward-Only)
=============================================
从 pool_evaluate.py 生成的 stats JSON（仅含 agent_0 奖励数据）生成论文级图表。
共 3 张图：平均奖励对比、逐组奖励分布、配对效应量。

Usage:
  python Python/training/pool_eval_plots.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "data_analyze"))

from publication_plot_utils import setup_style, save_figure, DEFAULT_PALETTE

# ── 中文字体 ───────────────────────────────────────────────────────────
def _configure_fonts() -> None:
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })
    try:
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

_configure_fonts()

_MODEL_LABELS_MAP = {
    "Direct":   "IPPO",
    "Pool":     "对手池博弈",
    "Average":  "平均策略博弈",
    "Untrained": "未训练模型",
}

_DEFAULT_STATS = "logs/ippo_pool_eval_stats.json"
_DEFAULT_OUTPUT = "article/imgs/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default=_DEFAULT_STATS, help="stats JSON 路径")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT, help="图片输出目录")
    parser.add_argument("--prefix", default="eval_compare", help="文件名前缀")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label(name: str) -> str:
    return _MODEL_LABELS_MAP.get(name, name)


# ═══════════════════════════════════════════════════════════════════════
#  图1：平均奖励对比（柱状图 + Bootstrap 95% CI）
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

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    colors = DEFAULT_PALETTE[:len(labels)]
    bars = ax.bar(x, rewards, yerr=[errors_lower, errors_upper],
                  capsize=6, color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "ecolor": "#444444"}, width=0.55)

    for bar, rw in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors_upper) * 0.03,
                f"{rw:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("平均回合奖励", fontsize=13)
    ax.set_title("智能体 0 平均回合奖励对比\n（95% Bootstrap 置信区间，20 组对手，每组 5 局）",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 显著性标注
    pairwise = stats.get("pairwise_comparisons", [])
    _add_significance_brackets(ax, pairwise, rewards)

    fig.tight_layout()
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_reward.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图2：逐组对手奖励分布（分组柱状图）
# ═══════════════════════════════════════════════════════════════════════

def plot_per_group_reward(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style(font_scale=1.1)
    models = stats["models"]
    labels = [_label(m["model_label"]) for m in models]
    n_groups = models[0]["n_groups"]
    n_models = len(models)

    data = np.zeros((n_models, n_groups))
    for i, m in enumerate(models):
        for g in range(n_groups):
            data[i, g] = m["per_group_mean_rewards"].get(str(g), 0)

    # 增加宽度和高度，避免柱子挤在一起
    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.65), 5.5))
    x = np.arange(n_groups)
    width = 0.65 / n_models
    colors = DEFAULT_PALETTE[:n_models]

    for i in range(n_models):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, data[i], width, label=labels[i],
               color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("对手组编号", fontsize=12)
    ax.set_ylabel("平均回合奖励", fontsize=12)
    ax.set_title("各对手组平均奖励对比", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    # 每隔一组显示标签，避免重叠
    tick_labels = [str(g) if g % 2 == 0 else "" for g in range(n_groups)]
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.legend(fontsize=9.5, edgecolor="#cccccc", ncol=n_models)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.margins(x=0.015)

    fig.tight_layout(pad=1.2)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_pergroup.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图3：配对效应量（Cohen's d 水平条形图）
# ═══════════════════════════════════════════════════════════════════════

def plot_effect_sizes(stats: dict, output_dir: str, prefix: str) -> None:
    setup_style()
    pairwise = stats.get("pairwise_comparisons", [])
    if not pairwise:
        print("[Plot] 无训练方法间配对比较数据。")
        return

    labels = [_label(p["models"][0]) + " vs\n" + _label(p["models"][1]) for p in pairwise]
    d_values = np.array([p["cohens_d_reward"] for p in pairwise])
    colors = [DEFAULT_PALETTE[0] if d >= 0 else DEFAULT_PALETTE[2] for d in d_values]

    # 加宽画布，给多行中文标签和负向标注留足间距
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(range(len(labels)), d_values, color=colors, edgecolor="white", height=0.5)

    # 数值 + 显著性标记
    for i, (d, p) in enumerate(zip(d_values, pairwise)):
        sig = "**" if p["significant_01"] else ("*" if p["significant_05"] else "")
        text = f"d={d:.2f}{sig}"
        offset =0.01
        x_pos = d + offset if d >= 0 else d - offset
        ha = "left" if d >= 0 else "right"
        ax.text(x_pos, i, text, fontsize=9, va="center", ha=ha,
                fontweight="bold", color="#333333")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's d（奖励）", fontsize=12)
    ax.set_title("配对效应量", fontsize=13, fontweight="bold")

    # 效应量参考线
    limit = max(abs(d_values).max() * 1.3, 1.0)
    ax.set_xlim(-limit, limit)
    for x_val, name in [(0.2, "小"), (0.5, "中"), (0.8, "大")]:
        ax.axvline(x=x_val, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
        ax.axvline(x=-x_val, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    ax.axvline(x=0, color="#444444", linewidth=1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout(pad=1.5, rect=[0.22, 0.02, 1, 1])
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_effectsize.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  辅助：显著性括号
# ═══════════════════════════════════════════════════════════════════════

def _add_significance_brackets(
    ax: plt.Axes, pairwise: list[dict], values: np.ndarray,
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
        sig = "**" if pw["significant_01"] else ("*" if pw["significant_05"] else "")
        if not sig:
            continue
        y = y_max * 1.04 + level * y_max * 0.10
        ax.plot([i, i, j, j], [y, y + y_max * 0.015, y + y_max * 0.015, y],
                color="#444444", linewidth=1.0, clip_on=False)
        ax.text((i + j) / 2, y + y_max * 0.022, sig, ha="center", fontsize=14,
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
        sys.exit(1)

    stats = load_stats(str(stats_path))
    print(f"已加载: {stats_path} ({len(stats['models'])} 个模型)")

    output_dir = args.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    plot_reward_comparison(stats, output_dir, prefix)
    plot_per_group_reward(stats, output_dir, prefix)
    plot_effect_sizes(stats, output_dir, prefix)

    print(f"\n全部图片已保存至 {output_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
