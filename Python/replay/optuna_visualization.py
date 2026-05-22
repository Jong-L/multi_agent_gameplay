"""
Optuna 超参搜索结果可视化 (论文出版级)
========================================
读取 Optuna SQLite 数据库，生成可直接用于论文的图表。

图表包括:
  1. 优化历史 (目标值随 trial 变化, 含平滑曲线)
  2. 累积最优收敛 (best-so-far vs trial)
  3. 参数重要性排名 (水平条形图)
  4. 参数切片 (每个超参与目标值的关系, 分面散点 + 趋势)
  5. 平行坐标 (多参数联动视图, 颜色映射目标值)
  6. 经验分布函数 (EDF)
  7. Trial 时间线 (甘特图风格)
  8. 综合仪表盘 (2x2 论文整合图)

特性:
  - matplotlib + publication_plot_utils 管线
  - 高斯滤波 + B-spline 平滑
  - Wong 2011 色盲友好配色
  - 300 DPI PNG 输出 (支持 pdf/svg)
  - 自动处理 PRUNED/FAIL 状态 trial
"""

import optuna
from optuna.distributions import (
    FloatDistribution, IntDistribution, CategoricalDistribution
)
import optuna.trial
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端, 适合脚本
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pathlib
import argparse
import sys
import warnings

# ── 导入项目绘图工具 ──────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "data_analyze"))
_has_pub_utils = True
try:
    from publication_plot_utils import (
        setup_style, save_figure, COLORS, DEFAULT_PALETTE,
        gaussian_smooth, spline_smooth
    )
except ImportError:
    _has_pub_utils = False
    warnings.warn("data_analyze/publication_plot_utils.py 未找到, "
                  "使用默认 matplotlib 样式。")

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ╔══════════════════════════════════════════════════════════════╗
# ║                       默认配置                                ║
# ╚══════════════════════════════════════════════════════════════╝
DEFAULT_DB_PATH = str(
    _PROJECT_ROOT / "logs" / "optuna" / "custom_ppo.db"
)
DEFAULT_STUDY_NAME = "custom_ppo_optuna"
DEFAULT_OUTPUT_DIR = str(
    _PROJECT_ROOT / "logs" / "optuna" / "figures"
)

# ── 配色别名 ─────────────────────────────────────────────────
BLUE   = '#0173B2'
ORANGE = '#DE8F05'
GREEN  = '#029E73'
RED    = '#D55E00'
PURPLE = '#949DE3'
GRAY   = '#949494'
PINK   = '#CC78BC'


# ╔══════════════════════════════════════════════════════════════╗
# ║                     工具函数                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def load_study(db_path: str, study_name: str) -> optuna.Study:
    """从 SQLite 数据库加载 Optuna study。"""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"[OK] 已加载 study: {study.study_name}  "
          f"(trials={len(study.trials)}, direction={study.direction})")
    return study


def print_best_trial(study: optuna.Study) -> None:
    """打印最佳 trial 的参数汇总。"""
    trial = study.best_trial
    print("\n" + "=" * 60)
    print(f"  最佳 Trial #{trial.number}  |  value = {trial.value:.6f}")
    print("=" * 60)
    for key, val in trial.params.items():
        print(f"  {key:30s} = {val}")
    print("=" * 60 + "\n")


def _get_completed_trials(study: optuna.Study) -> list:
    """获取所有已完成 (非 PRUNED/FAIL) 的 trial。"""
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None]


def _is_minimize(study: optuna.Study) -> bool:
    """study 方向是否为最小化。"""
    return study.direction == optuna.study.StudyDirection.MINIMIZE


# ── 颜色工具 (放在前面, 被后续函数调用) ──
def _hex_to_rgb(h: str) -> tuple:
    """十六进制颜色 → (R, G, B)。"""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _blend_color(c1: str, c2: str, ratio: float) -> str:
    """按 ratio 混合两个十六进制颜色 (0.0=c1, 1.0=c2)。"""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = r1 + (r2 - r1) * ratio
    g = g1 + (g2 - g1) * ratio
    b = b1 + (b2 - b1) * ratio
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def _setup_style_safe():
    """安全调用 setup_style, 不可用时回退。"""
    if _has_pub_utils:
        setup_style()
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        matplotlib.rcParams.update({
            'figure.dpi': 150, 'savefig.dpi': 300,
            'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
            'savefig.pad_inches': 0.1,
        })
    # 强制 RGBA → RGB (避免部分渲染器兼容问题)
    matplotlib.rcParams['savefig.transparent'] = False


def _save(fig, out_dir, name, fmt='png'):
    """统一保存, 带格式回退。"""
    path = pathlib.Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / f"{name}.{fmt}"
    if _has_pub_utils:
        save_figure(fig, str(fpath), dpi=300)
    else:
        fig.savefig(str(fpath), dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {fpath}")
    return fpath


def _smooth_trend(x, y):
    """对 x,y 数据做高斯 + 样条平滑, 返回 (x_smooth, y_smooth)。"""
    n = len(x)
    if n < 4 or not _HAS_SCIPY:
        return np.array(x), np.array(y)
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return x, y
    x_c, y_c = x[mask], y[mask]
    # 按 x 排序
    order = np.argsort(x_c)
    x_c, y_c = x_c[order], y_c[order]
    # 高斯滤波
    sigma = max(1.0, n * 0.04)
    try:
        y_f = gaussian_filter1d(y_c, sigma=sigma, mode='nearest')
    except Exception:
        y_f = y_c
    # B-spline
    try:
        k = min(3, len(x_c) - 1)
        spl = make_interp_spline(x_c, y_f, k=k)
        x_dense = np.linspace(x_c[0], x_c[-1], max(200, n))
        y_smooth = spl(x_dense)
        return x_dense, y_smooth
    except Exception:
        return x_c, y_f


# ╔══════════════════════════════════════════════════════════════╗
# ║                   图表生成函数                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_optimization_history(study: optuna.Study, output_dir: str,
                               fmt: str = 'png') -> plt.Figure:
    """
    优化历史: 目标值随 trial number 变化。
    包含散点 (原始值) + 平滑趋势线 + 最优标注。
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) == 0:
        print("[skip] 优化历史: 无已完成 trial")
        return None

    numbers = np.array([t.number for t in trials])
    values  = np.array([t.value for t in trials])
    is_min  = _is_minimize(study)

    # 最佳 trial 索引
    best_idx = np.argmin(values) if is_min else np.argmax(values)
    best_val = values[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── 散点 (原始数据) ──
    ax.scatter(numbers, values, s=35, c=BLUE, alpha=0.45,
               edgecolors='none', zorder=2, label='Trial results')

    # ── 平滑趋势线 ──
    if len(trials) >= 5:
        xs, ys = _smooth_trend(numbers, values)
        ax.plot(xs, ys, color=BLUE, linewidth=2.2, zorder=4,
                label='Smoothed trend')

    # ── 最优标注 ──
    ax.scatter([numbers[best_idx]], [best_val], s=140,
               c=RED, marker='*', edgecolors='#666666',
               linewidth=0.6, zorder=5,
               label=f'Best (trial #{trials[best_idx].number}, '
                     f'{best_val:.4f})')
    ax.axhline(y=best_val, color=RED, linestyle='--', linewidth=1.0,
               alpha=0.5, zorder=1)

    # ── 样式 ──
    ax.set_xlabel('Trial Number', fontsize=13)
    direction_label = 'Minimize' if is_min else 'Maximize'
    ax.set_ylabel(f'Objective Value ({direction_label})', fontsize=13)
    ax.set_title('Optimization History', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              edgecolor='#cccccc')
    ax.set_xlim(left=-0.5)
    fig.tight_layout()

    _save(fig, output_dir, 'optimization_history', fmt)
    return fig


def plot_best_so_far(study: optuna.Study, output_dir: str,
                      fmt: str = 'png') -> plt.Figure:
    """
    累积最优收敛: 每步到目前为止的最优目标值。
    展示搜索过程中优化器的逐步改进。
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) == 0:
        print("[skip] 累积最优: 无已完成 trial")
        return None

    is_min = _is_minimize(study)
    numbers = np.array([t.number for t in trials])
    values  = np.array([t.value for t in trials])

    # 计算 best-so-far
    if is_min:
        best_sofar = np.minimum.accumulate(values)
    else:
        best_sofar = np.maximum.accumulate(values)

    # 最终值
    final_best = best_sofar[-1]

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── 阶梯线 ──
    ax.step(numbers, best_sofar, where='post', color=GREEN,
            linewidth=2.2, zorder=3, label='Best so far')

    # ── 填充区域 ──
    y_min = min(values)
    y_max = max(values)
    ax.fill_between(numbers, best_sofar, y_min - 0.05 * (y_max - y_min),
                    step='post', color=GREEN, alpha=0.12,
                    edgecolor='none', zorder=1)

    # ── 终点标注 ──
    ax.scatter([numbers[-1]], [final_best], s=100, c=GREEN,
               edgecolors='#555555', linewidth=0.8,
               zorder=5)
    ax.annotate(f'  {final_best:.4f}',
                xy=(numbers[-1], final_best),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold', color=GREEN,
                va='bottom')

    # ── 初始值水平线 ──
    ax.axhline(y=best_sofar[0], color=GRAY, linestyle=':',
               linewidth=0.8, alpha=0.6, zorder=0)

    # ── 样式 ──
    ax.set_xlabel('Trial Number', fontsize=13)
    direction_label = 'Minimize' if is_min else 'Maximize'
    ax.set_ylabel(f'Best Objective ({direction_label})', fontsize=13)
    ax.set_title('Best-So-Far Convergence', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, frameon=True,
              edgecolor='#cccccc')
    ax.set_xlim(left=-0.5)
    fig.tight_layout()

    _save(fig, output_dir, 'best_so_far', fmt)
    return fig


def plot_param_importances(study: optuna.Study, output_dir: str,
                            fmt: str = 'png') -> plt.Figure:
    """
    参数重要性排名: 水平条形图, 按重要性降序排列。
    """
    _setup_style_safe()
    try:
        importances = optuna.importance.get_param_importances(study)
    except (ValueError, ImportError, RuntimeError) as e:
        print(f"[skip] 参数重要性: {e}")
        return None

    if not importances:
        print("[skip] 参数重要性: 无有效数据")
        return None

    # 按值降序排列
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names, vals = zip(*items)
    names = list(names)
    vals = np.array(vals)

    # 动态高度
    n = len(names)
    fig_h = max(3.5, 0.45 * n + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    # 颜色渐变 (重要 → 不重要)
    colors = [BLUE] * n
    for i in range(n):
        ratio = 1.0 - (i / max(n - 1, 1)) * 0.55
        colors[i] = _blend_color(BLUE, '#dddddd', ratio)

    bars = ax.barh(range(n), vals, color=colors, edgecolor='white',
                   linewidth=0.6, height=0.65)

    # 数值标签
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9.5, color='#333333')

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=13)
    ax.set_title('Hyperparameter Importance', fontsize=15, fontweight='bold')
    ax.set_xlim(right=max(vals) * 1.18)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    _save(fig, output_dir, 'param_importances', fmt)
    return fig


def plot_param_slice(study: optuna.Study, output_dir: str,
                      fmt: str = 'png') -> plt.Figure:
    """
    参数切片: 每个超参与目标值的分面散点图 + 平滑趋势线。
    自动检测参数类型 (连续/离散/类别), 选择合适的可视化方式。
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) < 3:
        print("[skip] 参数切片: trial 数量不足 (需要 ≥3)")
        return None

    # 获取参数名 (取第一个 COMPLETE trial 的分布)
    try:
        distributions = trials[0].distributions
    except AttributeError:
        print("[skip] 参数切片: 无法获取参数分布")
        return None

    param_names = list(distributions.keys())
    n_params = len(param_names)
    if n_params == 0:
        print("[skip] 参数切片: 无参数")
        return None

    is_min = _is_minimize(study)
    direction_label = '↓ (minimize)' if is_min else '↑ (maximize)'

    # 收集所有 (param_value, objective_value) 对
    param_data = {}
    for name in param_names:
        x_vals = []
        y_vals = []
        for t in trials:
            if name in t.params and t.value is not None:
                x_vals.append(t.params[name])
                y_vals.append(t.value)
        param_data[name] = (np.array(x_vals), np.array(y_vals))

    # 网格布局
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 4.0 * n_rows))
    if n_params == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(param_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        x_all, y_all = param_data[name]
        dist = distributions[name]

        # 判断参数类型
        unique_x = np.unique(x_all)
        is_cat = isinstance(dist, CategoricalDistribution)
        is_small_int = (isinstance(dist, IntDistribution)
                        and len(unique_x) <= 15)
        is_continuous = (isinstance(dist, FloatDistribution)
                         or (isinstance(dist, IntDistribution)
                             and len(unique_x) > 15))

        if is_cat or is_small_int:
            # ── 类别 / 少量离散: 箱线图 + 散点 ──
            groups = {}
            for xv, yv in zip(x_all, y_all):
                groups.setdefault(str(xv), []).append(yv)
            cat_labels = sorted(groups.keys(),
                                key=lambda k: np.median(groups[k]))
            bp_data = [groups[k] for k in cat_labels]
            bp = ax.boxplot(bp_data, patch_artist=True,
                            widths=0.55, showfliers=False,
                            medianprops=dict(color='#333333', linewidth=1.2))
            for patch in bp['boxes']:
                patch.set_facecolor(BLUE)
                patch.set_alpha(0.25)
            # 散点 overlay
            for i, lbl in enumerate(cat_labels):
                y_g = groups[lbl]
                jitter = np.random.uniform(-0.15, 0.15, len(y_g))
                ax.scatter(np.full(len(y_g), i + 1) + jitter, y_g,
                           s=12, c=BLUE, alpha=0.35, edgecolors='none')
            ax.set_xticklabels(cat_labels, fontsize=8, rotation=25, ha='right')

        else:
            # ── 连续型: 散点 + 平滑趋势 ──
            ax.scatter(x_all, y_all, s=18, c=BLUE, alpha=0.35,
                       edgecolors='none', zorder=2)
            if len(x_all) >= 5:
                xs, ys = _smooth_trend(x_all, y_all)
                ax.plot(xs, ys, color=BLUE, linewidth=2.0, zorder=3)

        # ── 最优参数位置竖线 ──
        best_trial = study.best_trial
        if name in best_trial.params:
            best_p = best_trial.params[name]
            ax.axvline(x=best_p, color=RED, linestyle='--',
                       linewidth=1.0, alpha=0.5, zorder=1)

        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel(f'Objective {direction_label}', fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.4)

    # 隐藏空子图
    for idx in range(n_params, axes.size):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Parameter Slice: Hyperparameter vs Objective',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()

    _save(fig, output_dir, 'param_slice', fmt)
    return fig


# ── 平行坐标归一化工具 ──
def _normalize_param_values(values, dist, all_vals=None):
    """
    将一组参数值归一化到 [0, 1]。

    参数:
        values:   当前 trial 子集的参数值 (1D array)
        dist:     Optuna 分布对象
        all_vals: 全量 trial 的参数值 (用于确定全局 min/max)

    返回:
        (normalized_values, tick_info)
        tick_info = {'type': 'continuous'/'categorical', 'ticks': [(frac, label), ...]}
    """
    values = np.asarray(values, dtype=object)
    numeric = np.array([float(v) if not isinstance(v, (str, bool)) else np.nan
                        for v in values], dtype=float)

    if isinstance(dist, CategoricalDistribution):
        choices = list(dist.choices)
        n = len(choices)
        mapping = {c: i / max(1, n - 1) for i, c in enumerate(choices)}
        normed = np.array([mapping.get(v, 0.5) for v in values])
        ticks = [(i / max(1, n - 1), str(c)[:18]) for i, c in enumerate(choices)]
        return normed, {'type': 'categorical', 'ticks': ticks}

    elif isinstance(dist, (FloatDistribution, IntDistribution)):
        # 用全量数据确定范围 (保证轴范围一致)
        if all_vals is not None:
            all_num = np.array([float(v) if not isinstance(v, (str, bool)) else np.nan
                                for v in all_vals], dtype=float)
            vmin = np.nanmin(all_num)
            vmax = np.nanmax(all_num)
        else:
            vmin = np.nanmin(numeric)
            vmax = np.nanmax(numeric)

        if vmax - vmin < 1e-10:
            return np.full(len(values), 0.5), {
                'type': 'continuous', 'ticks': [(0.5, f'{vmin:.3g}')]}

        normed = np.clip((numeric - vmin) / (vmax - vmin), 0, 1)
        # 生成 5 个均匀刻度
        tick_fracs = [0, 0.25, 0.5, 0.75, 1.0]
        ticks = []
        for f in tick_fracs:
            actual = vmin + f * (vmax - vmin)
            if isinstance(dist, FloatDistribution):
                label = f'{actual:.3g}'
            else:
                label = f'{int(round(actual))}'
            ticks.append((f, label))
        return normed, {'type': 'continuous', 'ticks': ticks}

    else:
        return np.full(len(values), 0.5), {
            'type': 'continuous', 'ticks': [(0.5, 'N/A')]}


def plot_parallel_coordinate(study: optuna.Study, output_dir: str,
                              fmt: str = 'png', n_display: int = 50) -> plt.Figure:
    """
    平行坐标图: 每个 trial 是一条折线, 横跨所有参数维度。
    颜色按目标值着色 → 一眼看出哪些参数组合通向优解。

    控制参数:
        n_display: 最多显示前 N 个最优 trial (避免线条过密)
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) < 2:
        print("[skip] 平行坐标: 需 ≥2 已完成 trial")
        return None

    distributions = trials[0].distributions
    param_names = list(distributions.keys())
    n_params = len(param_names)
    if n_params < 2:
        print("[skip] 平行坐标: 需 ≥2 参数")
        return None

    is_min = _is_minimize(study)

    # ── 收集全量数据 ──
    all_rows = []
    for t in trials:
        row = [t.params.get(n, np.nan) for n in param_names]
        row.append(t.value)
        all_rows.append(row)

    # 按目标值排序 + 取 top N
    all_rows.sort(key=lambda r: r[-1], reverse=not is_min)
    top_rows = all_rows[:min(n_display, len(all_rows))]

    obj_vals = np.array([r[-1] for r in top_rows])
    n_shown = len(top_rows)

    # ── 归一化 (使用全量数据确定范围, 保证轴刻度稳定) ──
    all_param_vals = {name: np.array([r[i] for r in all_rows])
                      for i, name in enumerate(param_names)}

    normed = np.zeros((n_shown, n_params))
    tick_infos = []
    for j, name in enumerate(param_names):
        vals = np.array([r[j] for r in top_rows])
        normed[:, j], tinfo = _normalize_param_values(
            vals, distributions[name],
            all_vals=all_param_vals[name]
        )
        tick_infos.append(tinfo)

    # ── 颜色映射 ──
    cmap = plt.cm.viridis_r if is_min else plt.cm.viridis
    norm = Normalize(vmin=obj_vals.min(), vmax=obj_vals.max())

    # ── 画布 ──
    fig_w = max(11, n_params * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    x_pos = np.arange(n_params)

    # ── 参数轴 (细竖线) ──
    for x in x_pos:
        ax.axvline(x=x, color=GRAY, linewidth=0.7, alpha=0.35, zorder=1)

    # ── 背景网格 ──
    for y_frac in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y=y_frac, color=GRAY, linewidth=0.3, alpha=0.15,
                   linestyle='--', zorder=0)

    # ── 绘制折线 ──
    # Top-3 用粗线突出, 其余渐变透明
    for i in range(n_shown):
        if i < 3:
            alpha, lw = 0.9, 2.5
            zorder = 5
        elif i < max(5, n_shown * 0.1):
            alpha, lw = 0.55, 1.5
            zorder = 3
        else:
            alpha, lw = 0.25, 0.8
            zorder = 2
        ax.plot(x_pos, normed[i], color=cmap(norm(obj_vals[i])),
                alpha=alpha, linewidth=lw, zorder=zorder)

    # ── 顶部标记 Best trial ──
    best_obj = obj_vals[0]
    best_line = normed[0]
    ax.plot(x_pos, best_line, color=cmap(norm(best_obj)),
            alpha=0.95, linewidth=3.0, zorder=6,
            marker='o', markersize=5, markerfacecolor='white',
            markeredgecolor=cmap(norm(best_obj)),
            markeredgewidth=1.5)

    # ── 底部标注 trial 数量 ──
    ax.text(0.01, -0.12,
            f'Showing top {n_shown} / {len(trials)} trials',
            transform=ax.transAxes, fontsize=8, color=GRAY,
            style='italic')

    # ── X 轴: 参数名 ──
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, fontsize=9.5, rotation=25, ha='right')
    ax.set_xlim(-0.3, n_params - 0.7)
    ax.set_ylim(-0.08, 1.08)

    # 隐藏默认 Y 轴标签, 改为在各参数轴线右侧标注实际值
    ax.set_yticks([])
    ax.set_ylabel('')

    # ── 每个参数轴右侧标注实际刻度值 ──
    for j, name in enumerate(param_names):
        tinfo = tick_infos[j]
        x = j + 0.30  # 标注在轴右侧
        for frac, label in tinfo['ticks']:
            if 0 <= frac <= 1:
                ax.text(x, frac, label,
                        fontsize=6.5, color='#555555',
                        va='center', ha='left', alpha=0.85)

    # ── Colorbar ──
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.06)
    direction_label = '↓ lower=better' if is_min else '↑ higher=better'
    cbar.set_label(f'Objective  {direction_label}', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ── 标题 ──
    ax.set_title('Parallel Coordinate: Hyperparameter Interaction',
                 fontsize=15, fontweight='bold', pad=20)

    fig.tight_layout()
    _save(fig, output_dir, 'parallel_coordinate', fmt)
    return fig


def plot_edf(study: optuna.Study, output_dir: str,
              fmt: str = 'png') -> plt.Figure:
    """
    经验分布函数 (EDF): 目标值的累积分布。
    展示搜索结果的整体分布特征。
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) == 0:
        print("[skip] EDF: 无已完成 trial")
        return None

    values = np.sort([t.value for t in trials])
    n = len(values)
    ecdf_y = np.arange(1, n + 1) / n

    # 统计量
    median_val = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── EDF 阶梯线 ──
    ax.step(values, ecdf_y, where='post', color=BLUE,
            linewidth=2.2, zorder=3)

    # ── 参考线 ──
    for val, label, ls in [
        (median_val, 'Median', (0, (4, 2, 1, 2))),
        (q25, 'Q25', (0, (2, 3))),
        (q75, 'Q75', (0, (2, 3))),
    ]:
        ax.axvline(x=val, color=GRAY, linestyle=ls,
                   linewidth=1.0, alpha=0.6, zorder=1)
        ax.axhline(y=np.searchsorted(values, val) / n,
                   color=GRAY, linestyle=ls,
                   linewidth=0.6, alpha=0.4, zorder=1)

    # ── 统计信息框 ──
    stats_text = (
        f"N = {n}\n"
        f"Median = {median_val:.4f}\n"
        f"Q25    = {q25:.4f}\n"
        f"Q75    = {q75:.4f}"
    )
    ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.92, edgecolor='#cccccc'))

    # ── 样式 ──
    ax.set_xlabel('Objective Value', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('Empirical Distribution Function (EDF)',
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    _save(fig, output_dir, 'edf', fmt)
    return fig


def plot_timeline(study: optuna.Study, output_dir: str,
                   fmt: str = 'png') -> plt.Figure:
    """
    Trial 时间线: 甘特图风格, 按开始时间排序。
    颜色区分 COMPLETE / PRUNED / FAIL 三种状态。
    """
    _setup_style_safe()

    # 获取所有 trial 的时间信息
    timeline_data = []
    for t in study.trials:
        if t.datetime_start is None:
            continue
        start = t.datetime_start
        if t.datetime_complete is not None:
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        else:
            duration = 0
        timeline_data.append({
            'number': t.number,
            'start': start,
            'duration': max(duration, 0.5),  # 最小 0.5s 可见
            'state': t.state.name,
            'value': t.value,
        })

    if not timeline_data:
        print("[skip] 时间线: 无时间数据")
        return None

    # 按开始时间排序
    timeline_data.sort(key=lambda d: d['start'])
    t0 = timeline_data[0]['start']

    numbers = [d['number'] for d in timeline_data]
    starts  = [(d['start'] - t0).total_seconds() / 60 for d in timeline_data]
    durations = [d['duration'] / 60 for d in timeline_data]
    states  = [d['state'] for d in timeline_data]
    values  = [d['value'] for d in timeline_data]

    # 颜色映射
    state_colors = {
        'COMPLETE': GREEN,
        'PRUNED':   ORANGE,
        'FAIL':     RED,
    }

    n_trials = len(timeline_data)
    # 限制高度: 每 trial 0.22 inches, 最大 24 inches
    fig_h = min(24, max(4, 0.22 * n_trials + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # ── 水平条形 ──
    bar_colors = [state_colors.get(s, GRAY) for s in states]
    bars = ax.barh(range(len(timeline_data)), durations,
                   left=starts, color=bar_colors,
                   edgecolor='white', linewidth=0.4, height=0.7)

    # ── 最优值标记 ──
    is_min = _is_minimize(study)
    valid_vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if valid_vals and len(valid_vals) >= 1:
        if is_min:
            best_idx = min(valid_vals, key=lambda x: x[1])[0]
        else:
            best_idx = max(valid_vals, key=lambda x: x[1])[0]
        ax.annotate(f'Best: {values[best_idx]:.4f}',
                    xy=(starts[best_idx] + durations[best_idx],
                        best_idx),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=RED, va='center',
                    fontweight='bold')

    # ── 图例 ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN,  label='COMPLETE'),
        Patch(facecolor=ORANGE, label='PRUNED'),
        Patch(facecolor=RED,    label='FAIL'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=9, frameon=True, edgecolor='#cccccc')

    # ── 样式 ──
    ax.set_yticks(range(len(timeline_data)))
    ax.set_yticklabels([f'Trial #{n}' for n in numbers], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Elapsed Time (minutes)', fontsize=13)
    ax.set_title('Trial Timeline', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.25, linewidth=0.4)
    fig.tight_layout()

    _save(fig, output_dir, 'timeline', fmt)
    return fig


def plot_dashboard(study: optuna.Study, output_dir: str,
                    fmt: str = 'png') -> plt.Figure:
    """
    综合仪表盘: 2x2 布局, 一张图展示核心搜索全景。
    左上: 优化历史 | 右上: 累积最优
    左下: 参数重要性 | 右下: EDF
    适合作为论文中的超参搜索概览图。
    """
    _setup_style_safe()
    trials = _get_completed_trials(study)
    if len(trials) == 0:
        print("[skip] 仪表盘: 无已完成 trial")
        return None

    is_min = _is_minimize(study)
    numbers = np.array([t.number for t in trials])
    values  = np.array([t.value for t in trials])
    best_idx = np.argmin(values) if is_min else np.argmax(values)
    best_val = values[best_idx]
    if is_min:
        best_sofar = np.minimum.accumulate(values)
    else:
        best_sofar = np.maximum.accumulate(values)
    sorted_vals = np.sort(values)

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28)

    # ── (0,0) 优化历史 ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(numbers, values, s=28, c=BLUE, alpha=0.4,
                edgecolors='none', zorder=2)
    if len(trials) >= 5:
        xs, ys = _smooth_trend(numbers, values)
        ax1.plot(xs, ys, color=BLUE, linewidth=2.0, zorder=4)
    ax1.scatter([numbers[best_idx]], [best_val], s=110,
                c=RED, marker='*', edgecolors='#555555',
                linewidth=0.5, zorder=5)
    ax1.axhline(y=best_val, color=RED, linestyle='--',
                linewidth=0.8, alpha=0.45, zorder=1)
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Objective Value', fontsize=11)
    ax1.set_title('(a) Optimization History', fontsize=13,
                  fontweight='bold', loc='left')
    ax1.grid(alpha=0.25, linewidth=0.4)

    # ── (0,1) 累积最优 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.step(numbers, best_sofar, where='post', color=GREEN,
             linewidth=2.0, zorder=3)
    ax2.fill_between(numbers, best_sofar,
                     np.min(values) - 0.02 * np.ptp(values),
                     step='post', color=GREEN, alpha=0.10,
                     edgecolor='none', zorder=1)
    ax2.scatter([numbers[-1]], [best_sofar[-1]], s=90,
                c=GREEN, edgecolors='#555555',
                linewidth=0.5, zorder=5)
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Best Objective', fontsize=11)
    ax2.set_title('(b) Best-So-Far Convergence', fontsize=13,
                  fontweight='bold', loc='left')
    ax2.grid(alpha=0.25, linewidth=0.4)

    # ── (1,0) 参数重要性 ──
    ax3 = fig.add_subplot(gs[1, 0])
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            items = sorted(importances.items(), key=lambda x: x[1])
            pnames, pvals = zip(*items)
            pnames = list(pnames)
            pvals = np.array(pvals)
            n_p = len(pnames)
            pcolors = [_blend_color('#dddddd', BLUE, (i / max(n_p - 1, 1)))
                       for i in range(n_p)]
            ax3.barh(range(n_p), pvals, color=pcolors,
                     edgecolor='white', linewidth=0.5, height=0.6)
            ax3.set_yticks(range(n_p))
            ax3.set_yticklabels(pnames, fontsize=9)
            ax3.invert_yaxis()
    except Exception:
        ax3.text(0.5, 0.5, 'Importance not available',
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=11, color=GRAY)
    ax3.set_xlabel('Importance', fontsize=11)
    ax3.set_title('(c) Parameter Importance', fontsize=13,
                  fontweight='bold', loc='left')
    ax3.grid(axis='x', alpha=0.25, linewidth=0.4)

    # ── (1,1) EDF ──
    ax4 = fig.add_subplot(gs[1, 1])
    n = len(sorted_vals)
    ax4.step(sorted_vals, np.arange(1, n + 1) / n,
             where='post', color=BLUE, linewidth=2.0)
    med = np.median(sorted_vals)
    ax4.axvline(x=med, color=GRAY, linestyle='--', linewidth=0.8, alpha=0.6)
    ax4.set_xlabel('Objective Value', fontsize=11)
    ax4.set_ylabel('Cumulative Probability', fontsize=11)
    ax4.set_title(f'(d) EDF (N={n})', fontsize=13,
                  fontweight='bold', loc='left')
    ax4.set_ylim(0, 1.02)
    ax4.grid(alpha=0.25, linewidth=0.4)

    fig.suptitle('Hyperparameter Search: Comprehensive Overview',
                 fontsize=16, fontweight='bold', y=1.01)

    _save(fig, output_dir, 'dashboard', fmt)
    return fig


# ╔══════════════════════════════════════════════════════════════╗
# ║                     主流程                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_all(study: optuna.Study, output_dir: str,
                  fmt: str = 'png',
                  charts: list = None) -> None:
    """生成所有 (或指定) 可视化图表。

    参数:
        charts: 图表名列表, None=全部。可选: history, bestsofar,
                importances, slice, edf, timeline, dashboard
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_charts = {
        'history':      ('优化历史',       plot_optimization_history),
        'bestsofar':    ('累积最优收敛',   plot_best_so_far),
        'importances':  ('参数重要性',     plot_param_importances),
        'slice':        ('参数切片',       plot_param_slice),
        'parallel':     ('平行坐标',       plot_parallel_coordinate),
        'edf':          ('经验分布函数',   plot_edf),
        'timeline':     ('Trial 时间线',   plot_timeline),
        'dashboard':    ('综合仪表盘',     plot_dashboard),
    }

    # 确定要生成哪些图
    if charts is None:
        selected = list(all_charts.items())
    else:
        chart_set = set(charts)
        if 'all' in chart_set:
            selected = list(all_charts.items())
        else:
            selected = [(k, v) for k, v in all_charts.items()
                        if k in chart_set]
            unknown = chart_set - set(all_charts.keys()) - {'all'}
            if unknown:
                print(f"[WARN] 未知图表: {unknown}, 已忽略")

    print(f"\n{'='*55}")
    print(f"  生成论文级超参搜索可视化  (format={fmt})")
    print(f"{'='*55}\n")

    for key, (name, fn) in selected:
        print(f"  [{name}] ...")
        try:
            fn(study, output_dir, fmt)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] 图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna 超参搜索结果可视化 (论文出版级)"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH,
        help=f"Optuna SQLite 数据库路径 (默认: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--study-name", type=str, default=DEFAULT_STUDY_NAME,
        help=f"Study 名称 (默认: {DEFAULT_STUDY_NAME})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"图表输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format", "-f", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="输出格式: png (300 DPI 位图) | pdf (矢量) | svg (矢量)",
    )
    parser.add_argument(
        "--charts", "-c", type=str, nargs="+", default=None,
        help="指定生成哪些图表 (默认全部)。可选: history bestsofar "
             "importances slice parallel edf timeline dashboard all",
    )
    args = parser.parse_args()

    study = load_study(args.db, args.study_name)
    print_best_trial(study)
    generate_all(study, args.output_dir, args.format, args.charts)


if __name__ == "__main__":
    main()
