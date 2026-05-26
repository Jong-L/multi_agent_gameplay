"""
Ball Reward Shaping Scheme Comparison Analysis
=============================================
Analyzes 5 potential shaping schemes across 10 independent runs each:
  Scheme 1 (LINEAR):          prefix=lrrs_
  Scheme 2 (EXPONENTIAL):     prefix=exprs_
  Scheme 3 (INVERSE):         prefix=invprs_
  Scheme 4 (DISTANCE_REWARD): prefix=distance_rs_
  Scheme 5 (SPARSE):          prefix=sparse_

Outputs:
  - summary/*.csv  : aggregated data
  - summary/*.png  : seaborn plots (publication quality)

Data format: episode_id, player_id, source, value, game_time
Ball sources: collect_ball_A, collect_ball_B
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

try:
    from scipy.interpolate import make_interp_spline
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[WARN] scipy not available, fallback to np.convolve smoothing")

# ============================================================
# Configuration
# ============================================================
# 数据目录（脚本同目录）
BASE_DIR = Path(__file__).parent.resolve()
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")

# SPARSE 方案需截断到其他方案的最大 episode 数（保证对比公平）
SPARSE_TRUNCATE_EPISODE = 80

SCHEMES = {
    'LINEAR':           {'prefix': 'lrrs_',           'label': 'Linear',           'color': '#0173B2'},
    'EXPONENTIAL':      {'prefix': 'exprs_',          'label': 'Exponential',      'color': '#DE8F05'},
    'INVERSE':          {'prefix': 'invprs_',         'label': 'Inverse',          'color': '#029E73'},
    'DISTANCE_REWARD':  {'prefix': 'distance_rs_',    'label': 'Distance Reward',   'color': '#CC78BC'},
    'SPARSE':           {'prefix': 'sparse_',         'label': 'Sparse',            'color': '#F14CC1'},
}
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


# ============================================================
# Data Loading
# ============================================================
def find_scheme_files(base_dir, prefix):
    """Find all CSV files matching a scheme prefix."""
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(prefix) and f.endswith('.csv'):
            files.append(os.path.join(base_dir, f))
    files.sort()
    return files


def load_and_aggregate_file(filepath):
    """
    Load a single CSV file and aggregate ball rewards per (episode_id, player_id).
    Returns: DataFrame with columns [episode_id, player_id, ball_reward]
             or None if file is empty/malformed.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(filepath)}: {e}")
        return None

    if df.empty or 'source' not in df.columns:
        return None

    # Filter ball sources
    ball_df = df[df['source'].isin(BALL_SOURCES)].copy()
    if ball_df.empty:
        # File has no ball records — return empty aggregated with correct columns
        return pd.DataFrame(columns=['episode_id', 'player_id', 'ball_reward'])

    # Aggregate: sum value per (episode_id, player_id)
    agg = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    agg.rename(columns={'value': 'ball_reward'}, inplace=True)
    return agg


def check_episode_consistency(file_list, scheme_name):
    """
    Check if all files in a scheme have the same max episode_id.
    Returns: max_episode (int) if consistent, None if inconsistent.
    Note: SPARSE scheme skips this check (multi-batch runs have different lengths).
    """
    # SPARSE 方案来自多个批次，各文件长度不同，跳过一致性检查
    if scheme_name == 'Sparse':
        print(f"  [SKIP-CONSISTENCY] {scheme_name}: multi-batch runs, "
              f"consistency check bypassed (will truncate later).")
        return None  # None signals "skip check", not "fail"

    episode_counts = {}
    for fp in file_list:
        agg = load_and_aggregate_file(fp)
        if agg is None or agg.empty:
            max_ep = 0
        else:
            max_ep = agg['episode_id'].max()
        episode_counts[os.path.basename(fp)] = max_ep

    unique_counts = set(episode_counts.values())
    if len(unique_counts) > 1:
        print(f"\n  [ERROR] {scheme_name}: Episode counts are INCONSISTENT!")
        for fname, cnt in episode_counts.items():
            print(f"    {fname}: {cnt} episodes")
        return None

    max_ep = list(unique_counts)[0]
    print(f"  [OK] {scheme_name}: All {len(file_list)} files have {max_ep} episodes")
    return max_ep


def compute_scheme_average(file_list, scheme_name):
    """
    Compute per-episode per-player mean and SEM across all files in a scheme.
    Missing players in a file's episode are treated as 0.
    Returns: (mean_df, sem_df, players)
    """
    # Load all files
    all_records = []
    for fp in file_list:
        agg = load_and_aggregate_file(fp)
        if agg is None:
            agg = pd.DataFrame(columns=['episode_id', 'player_id', 'ball_reward'])
        all_records.append(agg)

    # Get all unique episodes and players
    all_episodes = set()
    all_players = set()
    for rec in all_records:
        if not rec.empty:
            all_episodes.update(rec['episode_id'].unique())
            all_players.update(rec['player_id'].unique())
    all_episodes = sorted(list(all_episodes))
    all_players = sorted(list(all_players))

    if len(all_episodes) == 0:
        print(f"  [WARN] {scheme_name}: No data found!")
        return None

    num_files = len(file_list)
    mean_rows = []
    sem_rows = []

    for ep in all_episodes:
        mean_row = {'episode_id': ep}
        sem_row = {'episode_id': ep}

        # Build per-player value lists and per-file totals
        player_values = {pid: [] for pid in all_players}
        file_totals = []

        for rec in all_records:
            file_total = 0.0
            for pid in all_players:
                mask = (rec['episode_id'] == ep) & (rec['player_id'] == pid)
                if mask.any():
                    val = rec.loc[mask, 'ball_reward'].values[0]
                else:
                    val = 0.0
                player_values[pid].append(val)
                file_total += val
            file_totals.append(file_total)

        # Per-player mean and SEM
        for pid in all_players:
            values = np.array(player_values[pid])
            mean_val = values.mean()
            sem_val = values.std(ddof=1) / np.sqrt(num_files) if num_files > 1 else 0.0
            mean_row[f'player_{pid}'] = mean_val
            sem_row[f'player_{pid}'] = sem_val

        # Total mean and SEM
        total_mean = np.mean(file_totals)
        total_sem = np.std(file_totals, ddof=1) / np.sqrt(num_files) if num_files > 1 else 0.0
        mean_row['total_avg'] = total_mean
        sem_row['total_avg'] = total_sem

        mean_rows.append(mean_row)
        sem_rows.append(sem_row)

    mean_df = pd.DataFrame(mean_rows)
    sem_df = pd.DataFrame(sem_rows)
    return mean_df, sem_df, all_players


# ============================================================
# Plotting
# ============================================================
def setup_seaborn_style():
    """统一 seaborn 绘图风格（与 data_analyze 保持一致）"""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)


def smooth_curve(y, window):
    """滑动平均平滑；窗口小于数据长度时才生效"""
    window = max(1, int(window))
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def smooth_with_spline(x, y, smoothing_factor=None):
    """
    使用 B-spline 插值对数据做平滑重采样，返回 (x_dense, y_smooth)。
    
    产生的曲线没有尖锐拐角，适合科研论文画图。
    
    参数:
        x: 原始 x 坐标 (1D)
        y: 原始 y 值 (1D)
        smoothing_factor: spline 平滑因子 (None=自动, 正数越大越平滑)
    
    返回:
        x_dense: 密集 x 坐标 (200 点)
        y_smooth: 平滑后 y 值
    """
    if not _HAS_SCIPY:
        return x, y  # fallback
    
    n = len(x)
    if n < 4:
        return x, y
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # 去除 NaN/Inf
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return x, y
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 自动选择平滑因子：数据越 noisy 用越大的 s
    if smoothing_factor is None:
        # 以数据 y-range 的 5% 作为默认平滑容差
        y_range = np.ptp(y_clean)
        smoothing_factor = max(0.001, y_range * 0.05)
    
    # k=3 三次样条，边界用 natural 条件
    try:
        k = min(3, len(x_clean) - 1)
        spl = make_interp_spline(x_clean, y_clean, k=k)
        spl.set_smoothing_factor(smoothing_factor)
        
        x_dense = np.linspace(x_clean[0], x_clean[-1], max(200, len(x)))
        y_smooth = spl(x_dense)
        return x_dense, y_smooth
    except Exception:
        return x, y  # 数值问题则回退


def gaussian_smooth(y, sigma=2.0):
    """
    高斯滤波平滑 — 比滑动平均更自然，无箱形伪影。
    
    参数:
        y: 1D 数据
        sigma: 高斯核标准差 (默认 2.0，越大越平滑)
    """
    if not _HAS_SCIPY:
        return y
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    y_filled = y.copy()
    if not mask.all():
        # NaN 填充为前后均值
        y_filled[~mask] = np.interp(
            np.flatnonzero(~mask), np.flatnonzero(mask), y[mask]
        )
    return gaussian_filter1d(y_filled, sigma=sigma, mode='nearest')


def plot_per_player(scheme_data, save_path, smooth_window):
    """
    Plot per-player average ball reward vs episode for all schemes.
    4 subplots (one per player), each with one line per scheme.
    使用 高斯滤波 → 样条插值 两级平滑管线。
    """
    setup_seaborn_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Per-Player Average Ball Reward by Shaping Scheme',
                 fontsize=15, fontweight='bold', y=1.02)
    axes = axes.flatten()

    # Collect all player IDs from all schemes
    all_players = set()
    for scheme_key, (df, sem_df, players) in scheme_data.items():
        all_players.update(players)
    all_players = sorted(list(all_players))

    for idx, pid in enumerate(all_players):
        ax = axes[idx]
        col = f'player_{pid}'

        for scheme_key, (df, sem_df, players) in scheme_data.items():
            if col not in df.columns:
                continue
            episodes = df['episode_id'].values
            rewards = df[col].values
            sem_vals = sem_df[col].values

            x_smooth, y_smooth, y_lower, y_upper = _prepare_smooth_curve(
                episodes, rewards, sem_vals, smooth_window, scheme_key
            )

            color = SCHEMES[scheme_key]['color']

            # 填充区域
            ax.fill_between(x_smooth, y_lower, y_upper,
                            alpha=0.18,
                            color=color,
                            edgecolor='none',
                            linewidth=0,
                            zorder=1)

            # 均值曲线
            sns.lineplot(x=x_smooth, y=y_smooth,
                         label=SCHEMES[scheme_key]['label'],
                         color=color,
                         linewidth=2, ax=ax,
                         zorder=3)

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Reward', fontsize=11)
        ax.set_title(f'Player {pid}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#cccccc')
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_total(scheme_data, save_path, smooth_window):
    """
    Plot total average ball reward vs episode for all schemes.
    1 plot with one line per scheme.
    使用 高斯滤波 → 样条插值 两级平滑管线。
    """
    setup_seaborn_style()
    fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=150)

    stats_text = "Mean Reward per Episode:\n"
    for scheme_key, (df, sem_df, players) in scheme_data.items():
        episodes = df['episode_id'].values
        total = df['total_avg'].values
        total_sem = sem_df['total_avg'].values

        x_smooth, y_smooth, y_lower, y_upper = _prepare_smooth_curve(
            episodes, total, total_sem, smooth_window, scheme_key
        )

        color = SCHEMES[scheme_key]['color']

        # 填充区域
        ax.fill_between(x_smooth, y_lower, y_upper,
                        alpha=0.18,
                        color=color,
                        edgecolor='none',
                        linewidth=0,
                        zorder=1)

        # 均值曲线
        sns.lineplot(x=x_smooth, y=y_smooth,
                      label=SCHEMES[scheme_key]['label'],
                      color=color,
                      linewidth=2.0, ax=ax,
                      zorder=3)

        mean_val = np.mean(total)
        stats_text += f"  {SCHEMES[scheme_key]['label']}: {mean_val:.2f}\n"

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward (all players)', fontsize=12)
    ax.set_title('Total Ball Collection Reward by Shaping Scheme',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Scheme',
              edgecolor='#cccccc')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    # Stats annotation
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                       edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def _prepare_smooth_curve(episodes, rewards, sem_vals, smooth_window, scheme_key):
    """
    统一平滑管线：
    1. 先用高斯滤波去除高频噪声
    2. 再用样条插值生成光滑曲线
    返回 (x_smooth, y_smooth, y_lower, y_upper) 用于画图。
    """
    n = len(rewards)

    # Step 0: 数据太少不处理
    if n < 3:
        return episodes, rewards, rewards - sem_vals, rewards + sem_vals

    # Step 1: 高斯平滑（sigma 与数据长度成比例，噪点多时更平滑）
    # 窗口期越长 → sigma 越大
    sigma = max(1.0, smooth_window * 0.35)
    if _HAS_SCIPY:
        sm_mean = gaussian_smooth(rewards, sigma=sigma)
        sm_sem = gaussian_smooth(sem_vals, sigma=sigma)
        sm_x = episodes
    else:
        # fallback: 滑动平均
        if n >= smooth_window:
            sm_mean = smooth_curve(rewards, smooth_window)
            sm_sem = smooth_curve(sem_vals, smooth_window)
            sm_x = episodes[smooth_window - 1:]
        else:
            sm_mean = rewards
            sm_sem = sem_vals
            sm_x = episodes

    # Step 2: 样条插值生成密集光滑曲线（消除 fill_between 尖角）
    if _HAS_SCIPY:
        x_dense, y_dense = smooth_with_spline(sm_x, sm_mean)
        _, y_lower = smooth_with_spline(sm_x, sm_mean - sm_sem)
        _, y_upper = smooth_with_spline(sm_x, sm_mean + sm_sem)
        return x_dense, y_dense, y_lower, y_upper
    else:
        return sm_x, sm_mean, sm_mean - sm_sem, sm_mean + sm_sem


def plot_individual_player_comparison(scheme_data, player_id, save_path, smooth_window):
    """
    Standalone plot for a single player: all schemes compared.
    Useful for detailed per-player analysis.
    
    使用 高斯滤波 → 样条插值 两级平滑管线，
    fill_between 用更低 alpha + edgecolor='none' 消除硬边。
    """
    setup_seaborn_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    col = f'player_{player_id}'
    for scheme_key, (df, sem_df, players) in scheme_data.items():
        if col not in df.columns:
            continue
        episodes = df['episode_id'].values
        rewards = df[col].values
        sem_vals = sem_df[col].values

        x_smooth, y_smooth, y_lower, y_upper = _prepare_smooth_curve(
            episodes, rewards, sem_vals, smooth_window, scheme_key
        )

        color = SCHEMES[scheme_key]['color']

        # 填充区域：更低透明度 + 无硬边
        ax.fill_between(x_smooth, y_lower, y_upper,
                        alpha=0.18,
                        color=color,
                        edgecolor='none',
                        linewidth=0,
                        zorder=1)

        # 均值曲线画在填充之上
        sns.lineplot(x=x_smooth, y=y_smooth,
                     label=SCHEMES[scheme_key]['label'],
                     color=color,
                     linewidth=2.0, ax=ax,
                     zorder=3)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'Player {player_id} Average Ball Reward', fontsize=12)
    ax.set_title(f'Player {player_id} Ball Reward by Shaping Scheme',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#cccccc')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Ball Reward Shaping Scheme Comparison Analysis")
    print("=" * 60)

    # Ensure summary directory exists
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    print(f"\nOutput directory: {SUMMARY_DIR}")

    scheme_data = {}  # key: scheme_key, value: (mean_df, sem_df, players)
    scheme_max_episodes = {}  # key: scheme_key → max episode before truncation

    # --------------------------------------------------------
    # Step 1: Load and process each scheme
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 1/4] Loading and processing scheme data...")
    print("-" * 60)

    for scheme_key, scheme_info in SCHEMES.items():
        prefix = scheme_info['prefix']
        label = scheme_info['label']

        print(f"\n  Processing: {label} (prefix={prefix})")
        files = find_scheme_files(BASE_DIR, prefix)
        print(f"    Found {len(files)} files.")

        if len(files) == 0:
            print(f"  [WARN] No files found for {label}, skipping.")
            continue

        # Check episode consistency
        # SPARSE 方案来自多批次（不同 pid 运行数不同），跳过一致性检查
        skip_consistency = (scheme_key == 'SPARSE')
        max_ep = check_episode_consistency(files, label)
        if max_ep is None and not skip_consistency:
            print(f"  [SKIP] {label} due to inconsistent episode counts.")
            continue

        # Compute average across files
        result = compute_scheme_average(files, label)
        if result is None:
            continue
        df, sem_df, players = result

        # SPARSE 方案截断到 SPARSE_TRUNCATE_EPISODE，保证各方案 episode 数一致
        if scheme_key == 'SPARSE' and len(df) > SPARSE_TRUNCATE_EPISODE:
            df = df[df['episode_id'] <= SPARSE_TRUNCATE_EPISODE].copy()
            sem_df = sem_df[sem_df['episode_id'] <= SPARSE_TRUNCATE_EPISODE].copy()
            df.reset_index(drop=True, inplace=True)
            sem_df.reset_index(drop=True, inplace=True)
            print(f"    [TRUNCATE] SPARSE truncated to episode <= {SPARSE_TRUNCATE_EPISODE} "
                  f"({len(df)} episodes)")

        scheme_data[scheme_key] = (df, sem_df, players)
        scheme_max_episodes[scheme_key] = len(df)
        print(f"    Computed average: {len(df)} episodes, players={players}")

        # Save per-scheme CSV (mean + sem)
        csv_path = os.path.join(SUMMARY_DIR, f"{scheme_key.lower()}_average.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved CSV: {os.path.basename(csv_path)}")

        sem_csv_path = os.path.join(SUMMARY_DIR, f"{scheme_key.lower()}_sem.csv")
        sem_df.to_csv(sem_csv_path, index=False)
        print(f"    Saved SEM CSV: {os.path.basename(sem_csv_path)}")

    if len(scheme_data) == 0:
        print("\n[ERROR] No valid scheme data loaded. Exiting.")
        sys.exit(1)

    # 动态平滑窗口：取所有方案最小 episode 数的 5%，至少为 1
    min_episodes = min(scheme_max_episodes.values())
    smooth_window = max(1, int(min_episodes * 0.05))
    print(f"\n  [INFO] Dynamic smooth window = {min_episodes} × 5% = {smooth_window}")

    # --------------------------------------------------------
    # Step 2: Print summary statistics
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 2/4] Summary Statistics")
    print("-" * 60)

    for scheme_key, (df, sem_df, players) in scheme_data.items():
        label = SCHEMES[scheme_key]['label']
        print(f"\n  {label}:")
        print(f"    Episodes: {len(df)}")
        for pid in players:
            col = f'player_{pid}'
            if col in df.columns:
                avg = df[col].mean()
                print(f"    Player {pid} avg: {avg:.2f}")
        total_avg = df['total_avg'].mean()
        print(f"    Total avg per episode: {total_avg:.2f}")

    # --------------------------------------------------------
    # Step 3: Generate plots
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 3/4] Generating plots...")
    print("-" * 60)

    # Plot 1: Per-player (4 subplots)
    print("\n  Plot 1: Per-player comparison (4 subplots)...")
    plot_per_player(scheme_data,
                    os.path.join(SUMMARY_DIR, "per_player_comparison.png"),
                    smooth_window)

    # Plot 2: Total comparison
    print("\n  Plot 2: Total comparison...")
    plot_total(scheme_data,
               os.path.join(SUMMARY_DIR, "total_comparison.png"),
               smooth_window)

    # Plot 3-N: Individual player plots
    all_players = set()
    for scheme_key, (df, sem_df, players) in scheme_data.items():
        all_players.update(players)
    for pid in sorted(list(all_players)):
        print(f"\n  Plot: Player {pid} standalone...")
        plot_individual_player_comparison(
            scheme_data, pid,
            os.path.join(SUMMARY_DIR, f"player_{pid}_comparison.png"),
            smooth_window)

    # --------------------------------------------------------
    # Step 4: Save combined CSV for further analysis
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 4/4] Saving combined data...")
    print("-" * 60)

    # Build a combined DataFrame with all schemes
    combined = None
    for scheme_key, (df, sem_df, players) in scheme_data.items():
        label = SCHEMES[scheme_key]['label']
        df_copy = df.copy()
        df_copy['scheme'] = label
        if combined is None:
            combined = df_copy
        else:
            combined = pd.concat([combined, df_copy], ignore_index=True)

    if combined is not None:
        combined_path = os.path.join(SUMMARY_DIR, "all_schemes_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  [SAVED] {combined_path}")

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print(f"\n  Output files in: {SUMMARY_DIR}")
    print("  - per_player_comparison.png  : 4-subplot per-player comparison")
    print("  - total_comparison.png      : total average comparison")
    print("  - player_X_comparison.png  : individual player plots")
    print("  - *_average.csv             : per-scheme aggregated data")
    print("  - all_schemes_combined.csv  : combined data for further analysis")


if __name__ == "__main__":
    main()
