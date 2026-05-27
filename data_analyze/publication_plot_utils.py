"""
Publication-Quality Plot Utilities
====================================
两级平滑管线 + 统一 Seaborn 风格，产出可直接用于论文的矢量图。

平滑管线:
  原始数据 → 高斯滤波 (去噪) → B-spline 样条插值 (光滑化) → 画图

使用示例:
    from publication_plot_utils import setup_style, prepare_curve, plot_with_fill, save_figure
    setup_style()
    x_smooth, y_smooth, y_lower, y_upper = prepare_curve(x, y, err, smooth_window=5)
    plot_with_fill(ax, x_smooth, y_smooth, y_lower, y_upper, color='#0173B2', label='Method A')
    save_figure(fig, 'output.png')
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.interpolate import make_interp_spline
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ============================================================
# 配色方案 (Colorblind-friendly, Wong 2011 Nature Methods)
# ============================================================
COLORS = {
    'blue':     '#0173B2',
    'orange':   '#DE8F05',
    'green':    '#029E73',
    'pink':     '#CC78BC',
    'brown':    '#CA9161',
    'purple':   '#949DE3',
    'yellow':   '#FECE4B',
    'gray':     '#949494',
    'red':      '#D55E00',
    'cyan':     '#56B4E9',
}

# 默认方案颜色序列 (足够 5-8 条曲线)
DEFAULT_PALETTE = [
    '#0173B2',  # blue
    '#DE8F05',  # orange
    '#029E73',  # green
    '#CC78BC',  # pink
    '#CA9161',  # brown
    '#949DE3',  # purple
    '#FECE4B',  # yellow
    '#949494',  # gray
]


# ============================================================
# 风格设置
# ============================================================
def setup_style(context='paper', font_scale=1.3, style='whitegrid'):
    """
    统一 Seaborn + Matplotlib 论文级风格。
    
    参数:
        context: 'paper' | 'talk' | 'poster' | 'notebook'
        font_scale: 字体缩放 (默认 1.3)
        style: 'whitegrid' | 'white' | 'darkgrid' | 'ticks'
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    
    # 微调 rcParams
    matplotlib.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'mathtext.fontset': 'stix',
    })
    
    # 尝试设置中文字体（Windows系统）
    import platform
    if platform.system() == 'Windows':
        # Windows常用中文字体列表
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'].insert(0, font)
                break
            except:
                continue
    
    plt.rcParams['axes.unicode_minus'] = False


def save_figure(fig, path, dpi=300, **kwargs):
    """统一保存接口。"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ============================================================
# 平滑管线
# ============================================================
def _convolve_smooth(y, window):
    """滑动平均 (scipy 不可用时的回退方案)"""
    window = max(1, int(window))
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def gaussian_smooth(y, sigma=2.0):
    """高斯滤波 — 比滑动平均更自然，无箱形伪影。"""
    if not _HAS_SCIPY:
        return y
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    y_filled = y.copy()
    if not mask.all():
        y_filled[~mask] = np.interp(
            np.flatnonzero(~mask), np.flatnonzero(mask), y[mask]
        )
    return gaussian_filter1d(y_filled, sigma=sigma, mode='nearest')


def spline_smooth(x, y, smoothing_factor=None, n_points=None):
    """
    B-spline 样条插值平滑。
    
    参数:
        x, y: 原始数据 (1D)
        smoothing_factor: 平滑因子 (None=自动; 越大越平滑, 0=完全插值不降噪)
        n_points: 输出点数量 (默认 max(200, len(x)))
    
    返回:
        x_dense, y_smooth
    """
    if not _HAS_SCIPY:
        return x, y
    
    n = len(x)
    if n < 4:
        return x, y
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return x, y
    x_clean, y_clean = x[mask], y[mask]
    
    if smoothing_factor is None:
        y_range = np.ptp(y_clean)
        smoothing_factor = max(0.001, y_range * 0.05)
    
    try:
        k = min(3, len(x_clean) - 1)
        spl = make_interp_spline(x_clean, y_clean, k=k)
        spl.set_smoothing_factor(smoothing_factor)
        n_pts = n_points if n_points else max(200, len(x))
        x_dense = np.linspace(x_clean[0], x_clean[-1], n_pts)
        y_smooth = spl(x_dense)
        return x_dense, y_smooth
    except Exception:
        return x, y


def prepare_curve(episodes, values, errors=None, smooth_window=5,
                  error_band=True):
    """
    统一平滑管线入口。
    
    原始数据 → 高斯滤波 → 样条插值 → 输出 (x, y, y_lower, y_upper)
    
    参数:
        episodes:  x 轴数据
        values:    y 轴均值
        errors:    y 轴误差 (SEM 或 STD), None 则不画误差带
        smooth_window: 高斯滤波的窗口比例 (越大越平滑)
        error_band: 是否对 error 也做平滑/样条 (默认 True)
    
    返回:
        (x_smooth, y_smooth, y_lower, y_upper)
        如果 errors=None，y_lower/y_upper 均为 y_smooth
    """
    n = len(values)
    if errors is None:
        errors = np.zeros_like(values)
        error_band = False
    
    if n < 3:
        return (episodes, values,
                values - errors, values + errors)
    
    # Step 1: 高斯滤波去噪
    if _HAS_SCIPY:
        sigma = max(1.0, smooth_window * 0.35)
        y_filt = gaussian_smooth(values, sigma=sigma)
        e_filt = gaussian_smooth(errors, sigma=sigma) if error_band else errors
        x_filt = episodes
    else:
        # fallback: 滑动平均
        if n >= smooth_window:
            y_filt = _convolve_smooth(values, smooth_window)
            e_filt = _convolve_smooth(errors, smooth_window) if error_band else errors
            x_filt = episodes[smooth_window - 1:]
        else:
            y_filt, e_filt, x_filt = values, errors, episodes
    
    # Step 2: 样条插值光滑化
    if _HAS_SCIPY:
        x_smooth, y_smooth = spline_smooth(x_filt, y_filt)
        if error_band and np.any(errors > 0):
            _, y_lower = spline_smooth(x_filt, y_filt - e_filt)
            _, y_upper = spline_smooth(x_filt, y_filt + e_filt)
        else:
            y_lower = y_upper = y_smooth
    else:
        x_smooth, y_smooth = x_filt, y_filt
        y_lower = y_filt - e_filt
        y_upper = y_filt + e_filt
    
    return x_smooth, y_smooth, y_lower, y_upper


# ============================================================
# 高级画图封装
# ============================================================
def plot_with_fill(ax, x, y, y_lower, y_upper, color, label,
                   linewidth=2.0, alpha=0.18, zorder_line=3,
                   **line_kwargs):
    """
    在指定 Axes 上画一条曲线 + 淡色误差填充。
    
    填充在前 (zorder=1)，曲线在后 (zorder=3)，保证曲线不被填充遮盖。
    """
    ax.fill_between(x, y_lower, y_upper,
                    alpha=alpha, color=color,
                    edgecolor='none', linewidth=0,
                    zorder=1)
    sns.lineplot(x=x, y=y, label=label, color=color,
                 linewidth=linewidth, ax=ax, zorder=zorder_line,
                 **line_kwargs)


def add_stats_box(ax, text, loc='lower right',
                  fontsize=9, family=None):
    """在图上添加半透明统计信息框。"""
    # 如果没有指定字体，使用支持中文的字体
    if family is None:
        import platform
        if platform.system() == 'Windows':
            family = 'Microsoft YaHei'  # Windows 使用微软雅黑
        else:
            family = 'sans-serif'  # 其他系统使用无衬线字体
    
    loc_map = {
        'lower right': (0.98, 0.02, 'bottom', 'right'),
        'upper right': (0.98, 0.98, 'top', 'right'),
        'lower left':  (0.02, 0.02, 'bottom', 'left'),
        'upper left':  (0.02, 0.98, 'top', 'left'),
    }
    x, y, va, ha = loc_map.get(loc, loc_map['lower right'])
    ax.text(x, y, text, transform=ax.transAxes,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.9, edgecolor='#cccccc'),
            fontsize=fontsize, family=family)


def style_axes(ax, xlabel='', ylabel='', title='',
               xlim=None, ylim=None, legend_kwargs=None):
    """统一轴标签、标题、图例样式。"""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    default_legend = dict(loc='upper left', fontsize=10,
                          frameon=True, edgecolor='#cccccc')
    if legend_kwargs:
        default_legend.update(legend_kwargs)
    ax.legend(**default_legend)
