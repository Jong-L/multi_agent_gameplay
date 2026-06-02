"""
Pool Evaluation Visualization (Reward + Behavior)
==================================================
从 pool_evaluate.py 生成的 stats JSON + CSV 生成论文级图表。
共 6 张图：平均奖励对比、逐组奖励分布、配对效应量、
          行为事件频率对比、战斗vs收集散点图、行为画像。

Usage:
  python Python/training/pool_eval_plots.py
  python Python/training/pool_eval_plots.py --csv logs/ippo_pool_eval.csv
"""
from __future__ import annotations

import argparse
import csv
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
    "Untrained": "初始策略",
}

_DEFAULT_STATS = "logs/ippo_pool_eval_stats.json"
_DEFAULT_CSV = "logs/ippo_pool_eval.csv"
_DEFAULT_OUTPUT = "article/imgs/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default=_DEFAULT_STATS, help="stats JSON 路径")
    parser.add_argument("--csv", default=_DEFAULT_CSV, help="评估 CSV 路径（行为分析用）")
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
#  数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> list[dict]:
    """Load evaluation CSV, returning rows as list of dicts with numeric fields."""
    rows: list[dict] = []
    numeric_keys = {
        "group_id", "episode", "agent_id", "reward",
        "attack_launched", "damage_dealt_to_player", "damage_dealt_to_enemy",
        "damage_taken", "kill_enemy", "kill_player",
        "ball_A_collected", "ball_B_collected", "died", "wall_collision", "game_score",
    }
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict = {}
            for k, v in row.items():
                if k in numeric_keys:
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _agent0_rows(csv_rows: list[dict], main_agent_id: int = 0) -> list[dict]:
    """Filter to main agent's rows only."""
    return [r for r in csv_rows if int(r.get("agent_id", -1)) == main_agent_id]


def _per_method_rows(
    csv_rows: list[dict], model_label: str, main_agent_id: int = 0,
) -> list[dict]:
    return [r for r in csv_rows
            if r["model_label"] == model_label and int(r["agent_id"]) == main_agent_id]


# ═══════════════════════════════════════════════════════════════════════
#  图4：行为事件频率对比（分组柱状图）
# ═══════════════════════════════════════════════════════════════════════

# 中文标签映射
_EVENT_LABELS = {
    "attack_launched":        "发动攻击",
    "damage_dealt_to_player": "对玩家造成伤害",
    "damage_dealt_to_enemy":  "对敌人造成伤害",
    "damage_taken":           "受到伤害",
    "kill_enemy":             "击杀敌人",
    "kill_player":            "击杀玩家",
    "ball_A_collected":       "拾取A球",
    "ball_B_collected":       "拾取B球",
    "died":                   "死亡次数",
    "wall_collision":         "撞墙次数",
}

# 行为字段归类
_EVENT_GROUPS = {
    "战斗": ["attack_launched", "damage_dealt_to_player", "damage_dealt_to_enemy",
             "damage_taken", "kill_enemy", "kill_player"],
    "生存": ["died", "wall_collision"],
    "资源": ["ball_A_collected", "ball_B_collected"],
}


def plot_behavior_events(
    csv_rows: list[dict], output_dir: str, prefix: str,
) -> None:
    """Plot 4: Per-episode mean behavior event counts (2-subplot grouped bar chart).

    Left: 战斗事件（攻击、造成伤害、击杀）
    Right: 生存与资源（拾球、死亡、撞墙、受击）
    """
    setup_style(font_scale=1.05)
    rows0 = _agent0_rows(csv_rows)
    model_labels_raw = sorted({r["model_label"] for r in rows0})
    model_labels = [_label(m) for m in model_labels_raw]
    n_methods = len(model_labels)
    colors = DEFAULT_PALETTE[:n_methods]

    # 子图1: 战斗行为
    combat_events = [
        "attack_launched", "damage_dealt_to_player", "damage_dealt_to_enemy",
        "kill_enemy", "kill_player",
    ]
    # 子图2: 生存与资源
    survival_events = [
        "ball_A_collected", "ball_B_collected",
        "died", "wall_collision", "damage_taken",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for sub_idx, (ax, events) in enumerate(
        zip(axes, [combat_events, survival_events])
    ):
        n_events = len(events)
        x = np.arange(n_events)
        width = 0.7 / n_methods

        for i, ml_raw in enumerate(model_labels_raw):
            method_rows = _per_method_rows(csv_rows, ml_raw)
            means = []
            stds = []
            for ev in events:
                vals = [r.get(ev, 0.0) for r in method_rows]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))
            offset = (i - (n_methods - 1) / 2) * width
            ax.bar(x + offset, means, width, label=model_labels[i],
                   color=colors[i], edgecolor="white", linewidth=0.5,
                   yerr=stds, capsize=3,
                   error_kw={"linewidth": 0.8, "ecolor": "#666666"})

        ax.set_xticks(x)
        ax.set_xticklabels([_EVENT_LABELS.get(ev, ev) for ev in events],
                           fontsize=9.5, rotation=20, ha="right")
        ax.set_ylabel("每局均值", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].set_title("战斗行为", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8.5, edgecolor="#cccccc", ncol=n_methods, loc="upper left")
    axes[1].set_title("生存与资源", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8.5, edgecolor="#cccccc", ncol=n_methods, loc="upper left")

    fig.suptitle("行为事件频率对比（智能体 0 每局事件计数均值 ± 1σ）",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.5)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_behavior_events.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图5：战斗 vs 收集散点图（每组一个点）
# ═══════════════════════════════════════════════════════════════════════

def plot_combat_vs_collection(
    csv_rows: list[dict], output_dir: str, prefix: str,
) -> None:
    """Plot 5: Combat index vs resource collection per group, colored by method."""
    setup_style(font_scale=1.05)
    rows0 = _agent0_rows(csv_rows)
    model_labels_raw = sorted({r["model_label"] for r in rows0})
    model_labels = [_label(m) for m in model_labels_raw]
    colors = DEFAULT_PALETTE[:len(model_labels_raw)]

    # 聚合到 group 级别
    groups = sorted({int(r["group_id"]) for r in rows0})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 子图1: combat_events vs resource_events
    ax = axes[0]
    for mi, ml_raw in enumerate(model_labels_raw):
        method_rows = _per_method_rows(csv_rows, ml_raw)
        combat_vals: list[float] = []
        resource_vals: list[float] = []
        for g in groups:
            g_rows = [r for r in method_rows if int(r["group_id"]) == g]
            combat = np.mean([r.get("attack_launched", 0) + r.get("damage_dealt_to_player", 0)
                              + r.get("damage_dealt_to_enemy", 0)
                              + r.get("kill_enemy", 0) + r.get("kill_player", 0)
                              for r in g_rows])
            resource = np.mean([r.get("ball_A_collected", 0) + r.get("ball_B_collected", 0)
                               for r in g_rows])
            combat_vals.append(combat)
            resource_vals.append(resource)
        ax.scatter(combat_vals, resource_vals, c=colors[mi], label=model_labels[mi],
                   alpha=0.75, edgecolors="white", linewidth=0.5, s=60, zorder=3)

    ax.set_xlabel("进攻事件（攻击+伤害+击杀）/局", fontsize=11)
    ax.set_ylabel("资源收集（拾球）/局", fontsize=11)
    ax.set_title("进攻性 vs 资源收集\n（每组对手一个点）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, edgecolor="#cccccc")
    ax.grid(alpha=0.3, linestyle="--")

    # 子图2: combat_events vs game_score
    ax = axes[1]
    for mi, ml_raw in enumerate(model_labels_raw):
        method_rows = _per_method_rows(csv_rows, ml_raw)
        combat_vals = []
        score_vals = []
        for g in groups:
            g_rows = [r for r in method_rows if int(r["group_id"]) == g]
            combat = np.mean([r.get("attack_launched", 0) + r.get("damage_dealt_to_player", 0)
                              + r.get("damage_dealt_to_enemy", 0)
                              + r.get("kill_enemy", 0) + r.get("kill_player", 0)
                              for r in g_rows])
            score = np.mean([r.get("game_score", 0) for r in g_rows])
            combat_vals.append(combat)
            score_vals.append(score)
        ax.scatter(combat_vals, score_vals, c=colors[mi], label=model_labels[mi],
                   alpha=0.75, edgecolors="white", linewidth=0.5, s=60, zorder=3)

    ax.set_xlabel("进攻事件（攻击+伤害+击杀）/局", fontsize=11)
    ax.set_ylabel("局内分数", fontsize=11)
    ax.set_title("进攻性 vs 局内得分\n（每组对手一个点）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, edgecolor="#cccccc")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout(pad=2.0)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_combat_collection.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图6：归一化行为画像（水平条形图替代雷达图）
# ═══════════════════════════════════════════════════════════════════════

# 行为画像维度定义
_PROFILE_DIMS = [
    ("进攻性",   ["attack_launched", "kill_player", "damage_dealt_to_player"]),
    ("刷野效率", ["damage_dealt_to_enemy", "kill_enemy"]),
    ("资源收集", ["ball_A_collected", "ball_B_collected"]),
    ("生存能力", ["died"]),       # 反向归一化
    ("稳健性",   ["wall_collision", "damage_taken"]),  # 反向归一化
]


def _normalize_profile(
    csv_rows: list[dict], model_labels_raw: list[str],
) -> tuple[dict, dict]:
    """Compute per-method mean values for each profile dimension, then min-max normalize.

    Returns (raw_values: {model: [d1, d2, ...]}, norm_values: {model: [d1_norm, ...]})
    """
    raw: dict[str, list[float]] = {ml: [] for ml in model_labels_raw}
    for ml in model_labels_raw:
        method_rows = _per_method_rows(csv_rows, ml)
        n_eps = max(len(method_rows), 1)
        for dim_name, fields in _PROFILE_DIMS:
            total = sum(
                np.sum([r.get(f, 0.0) for r in method_rows])
                for f in fields
            )
            raw[ml].append(total / n_eps)

    # 确定哪些维度需要反向（越小越好）
    reverse_dims = {"生存能力", "稳健性"}

    # min-max normalize across methods
    norm: dict[str, list[float]] = {}
    n_dims = len(_PROFILE_DIMS)
    for d in range(n_dims):
        vals = [raw[ml][d] for ml in model_labels_raw]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax - vmin > 1e-8 else 1.0
        dim_name = _PROFILE_DIMS[d][0]
        for ml in model_labels_raw:
            if ml not in norm:
                norm[ml] = []
            if dim_name in reverse_dims:
                norm[ml].append((vmax - raw[ml][d]) / rng)
            else:
                norm[ml].append((raw[ml][d] - vmin) / rng)

    return raw, norm


def plot_behavior_profile(
    csv_rows: list[dict], output_dir: str, prefix: str,
) -> None:
    """Plot 6: Normalized behavior profile (grouped horizontal bar chart)."""
    setup_style(font_scale=1.05)
    model_labels_raw = sorted({r["model_label"] for r in _agent0_rows(csv_rows)})
    model_labels = [_label(m) for m in model_labels_raw]
    colors = DEFAULT_PALETTE[:len(model_labels_raw)]

    _, norm_vals = _normalize_profile(csv_rows, model_labels_raw)

    dim_names = [d[0] for d in _PROFILE_DIMS]
    n_dims = len(dim_names)
    n_methods = len(model_labels)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(n_dims)
    height = 0.7 / n_methods

    for i, ml_raw in enumerate(model_labels_raw):
        vals = norm_vals[ml_raw]
        offset = (i - (n_methods - 1) / 2) * height
        ax.barh(y + offset, vals, height, label=model_labels[i],
                color=colors[i], edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(dim_names, fontsize=11)
    ax.set_xlabel("归一化得分（越高越优）", fontsize=11)
    ax.set_title("智能体 0 行为画像对比\n（各维度跨方法 Min-Max 归一化）",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=9.5, edgecolor="#cccccc", ncol=n_methods, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # 添加维度分组分隔线
    for yi in range(1, n_dims):
        ax.axhline(y=yi - 0.35, color="#cccccc", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.tight_layout(pad=1.5)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_behavior_profile.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图7：PvP 风险 vs PvE 收益散点图（验证"博弈收益风险对冲"）
# ═══════════════════════════════════════════════════════════════════════

def plot_pvp_risk_vs_pve_reward(
    csv_rows: list[dict], output_dir: str, prefix: str,
) -> None:
    """Plot 7: PvP engagement vs PvE efficiency per group, testing risk-reward trade-off.

    X-axis: PvP events (damage_dealt_to_player + kill_player)
    Y-axis: PvE events (damage_dealt_to_enemy + kill_enemy) or game_score
    """
    setup_style(font_scale=1.05)
    rows0 = _agent0_rows(csv_rows)
    model_labels_raw = sorted({r["model_label"] for r in rows0})
    model_labels = [_label(m) for m in model_labels_raw]
    colors = DEFAULT_PALETTE[:len(model_labels_raw)]
    groups = sorted({int(r["group_id"]) for r in rows0})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 子图1: PvP damage → PvE damage
    ax = axes[0]
    for mi, ml_raw in enumerate(model_labels_raw):
        method_rows = _per_method_rows(csv_rows, ml_raw)
        pvp_vals: list[float] = []
        pve_vals: list[float] = []
        for g in groups:
            g_rows = [r for r in method_rows if int(r["group_id"]) == g]
            pvp = np.mean([r.get("damage_dealt_to_player", 0) + r.get("kill_player", 0)
                          for r in g_rows])
            pve = np.mean([r.get("damage_dealt_to_enemy", 0) + r.get("kill_enemy", 0)
                          for r in g_rows])
            pvp_vals.append(pvp)
            pve_vals.append(pve)
        ax.scatter(pvp_vals, pve_vals, c=colors[mi], label=model_labels[mi],
                   alpha=0.8, edgecolors="white", linewidth=0.5, s=70, zorder=3)

    ax.set_xlabel("PvP 事件（对玩家伤害+击杀）/局", fontsize=11)
    ax.set_ylabel("PvE 事件（对敌人伤害+击杀）/局", fontsize=11)
    ax.set_title("PvP 参与度 vs PvE 效率\n（每组对手一个点）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, edgecolor="#cccccc")
    ax.grid(alpha=0.3, linestyle="--")

    # 子图2: PvP damage → game_score
    ax = axes[1]
    for mi, ml_raw in enumerate(model_labels_raw):
        method_rows = _per_method_rows(csv_rows, ml_raw)
        pvp_vals = []
        score_vals = []
        for g in groups:
            g_rows = [r for r in method_rows if int(r["group_id"]) == g]
            pvp = np.mean([r.get("damage_dealt_to_player", 0) + r.get("kill_player", 0)
                          for r in g_rows])
            score = np.mean([r.get("game_score", 0) for r in g_rows])
            pvp_vals.append(pvp)
            score_vals.append(score)
        ax.scatter(pvp_vals, score_vals, c=colors[mi], label=model_labels[mi],
                   alpha=0.8, edgecolors="white", linewidth=0.5, s=70, zorder=3)

    ax.set_xlabel("PvP 事件（对玩家伤害+击杀）/局", fontsize=11)
    ax.set_ylabel("局内分数", fontsize=11)
    ax.set_title("PvP 参与度 vs 局内得分\n（每组对手一个点）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, edgecolor="#cccccc")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout(pad=2.0)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_pvp_risk_pve_reward.png")
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  图8：逐组对手行为事件曲线（折线图，4 关键事件 × 4 方法）
# ═══════════════════════════════════════════════════════════════════════

def plot_per_group_behavior(
    csv_rows: list[dict], output_dir: str, prefix: str,
) -> None:
    """Plot 8: Per-group behavior event curves (4 key events, line plots)."""
    setup_style(font_scale=1.05)
    rows0 = _agent0_rows(csv_rows)
    model_labels_raw = sorted({r["model_label"] for r in rows0})
    model_labels = [_label(m) for m in model_labels_raw]
    colors = DEFAULT_PALETTE[:len(model_labels_raw)]
    groups = sorted({int(r["group_id"]) for r in rows0})

    # 选择 4 个关键行为指标
    panels = [
        ("damage_dealt_to_enemy",  "对敌人造成伤害"),
        ("damage_dealt_to_player", "对玩家造成伤害"),
        ("kill_enemy",             "击杀敌人"),
        ("ball_B_collected",       "拾取 B 球"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()

    for idx, (field, title) in enumerate(panels):
        ax = axes_flat[idx]
        for mi, ml_raw in enumerate(model_labels_raw):
            method_rows = _per_method_rows(csv_rows, ml_raw)
            y_vals = []
            for g in groups:
                g_rows = [r for r in method_rows if int(r["group_id"]) == g]
                y_vals.append(np.mean([r.get(field, 0.0) for r in g_rows]))
            ax.plot(groups, y_vals, 'o-', color=colors[mi], label=model_labels[mi],
                    markersize=5, linewidth=1.5, markerfacecolor='white',
                    markeredgewidth=1.2, alpha=0.85)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("对手组编号", fontsize=10)
        ax.set_ylabel("每局均值", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8.5, edgecolor="#cccccc", ncol=len(model_labels_raw))

    fig.suptitle("各对手组行为事件曲线（智能体 0，每组 5 局均值）",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.0)
    out_path = str(pathlib.Path(output_dir) / f"{prefix}_pergroup_behavior.png")
    save_figure(fig, out_path)


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

    # ── 奖励相关图（原有3张）──
    plot_reward_comparison(stats, output_dir, prefix)
    plot_per_group_reward(stats, output_dir, prefix)
    plot_effect_sizes(stats, output_dir, prefix)

    # ── 行为分析图（新增3张）──
    csv_path = pathlib.Path(args.csv)
    if csv_path.exists():
        print(f"已加载: {csv_path}（行为分析数据）")
        csv_rows = load_csv(str(csv_path))
        plot_behavior_events(csv_rows, output_dir, prefix)
        plot_combat_vs_collection(csv_rows, output_dir, prefix)
        plot_behavior_profile(csv_rows, output_dir, prefix)
        plot_pvp_risk_vs_pve_reward(csv_rows, output_dir, prefix)
        plot_per_group_behavior(csv_rows, output_dir, prefix)
    else:
        print(f"[Warning] CSV 文件不存在 ({csv_path})，跳过行为分析图。"
              f"请先运行 pool_evaluate.py 生成数据。")

    print(f"\n全部图片已保存至 {output_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
