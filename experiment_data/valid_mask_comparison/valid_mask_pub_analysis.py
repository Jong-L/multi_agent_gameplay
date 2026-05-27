"""
Valid Mask Comparison — Publication-Style Analysis
===================================================
Compares 2 observation mask strategies:
  1. no_valid_mask  — raw missing data (no validity signal)
  2. use_valid_mask — explicit validity mask for missing slot entries

Metrics (preserved from valid_mask_analysis.py):
  - Total score = sum of ALL values
  - Ball score = collect_ball_A + collect_ball_B only

Data processing: per-file → mean ± SEM across 5 runs (real error bands)
Drawing: publication_plot_utils pipeline (Gaussian filter → B-spline → fill_between)
Colors: Wong 2011 colorblind-friendly DEFAULT_PALETTE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt


def _set_chinese_font():
    """
    动态检测并设置中文字体。
    在 sns.set_* / setup_style() 之后调用，防止被覆盖。
    """
    import matplotlib.font_manager as fm
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'PingFang SC',
        'KaiTi', 'FangSong', 'STKaiti', 'SimSun', 'YouYuan',
    ]
    available = set(f.name for f in fm.fontManager.ttflist)
    for c in candidates:
        if c in available:
            matplotlib.rcParams['font.sans-serif'] = [c, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    matplotlib.rcParams['axes.unicode_minus'] = False


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data_analyze"))
from publication_plot_utils import (
    setup_style, prepare_curve, plot_with_fill,
    style_axes, save_figure, add_stats_box, DEFAULT_PALETTE
)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "summary"
OUTPUT_DIR.mkdir(exist_ok=True)

MASK_NAMES = {
    'no_valid_mask': '无有效掩码',
    'use_valid_mask': '使用有效掩码',
}

MASK_COLORS = {
    'no_valid_mask': DEFAULT_PALETTE[1],  # #DE8F05 orange
    'use_valid_mask': DEFAULT_PALETTE[0],  # #0173B2 blue
}

# ==============================================================================
# File discovery
# ==============================================================================

def discover_files():
    """Discover all CSV files grouped by mask type."""
    patterns = {
        'no_valid_mask': 'no_valid_mask_*.csv',
        'use_valid_mask': 'use_valid_mask_*.csv',
    }
    result = {}
    for key, pattern in patterns.items():
        files = sorted(DATA_DIR.glob(pattern))
        if files:
            result[key] = [str(f) for f in files]
    return result


# ==============================================================================
# Data loading & aggregation
# ==============================================================================

def load_and_aggregate(files, mask_type):
    """
    Load all files for a mask type, aggregate per-episode per-player.
    Returns: (mean_df, sem_df, players)
    """
    all_records = []
    for fp in files:
        df = pd.read_csv(fp)
        if df.empty:
            continue
        # Aggregate per (episode_id, player_id)
        if 'source' in df.columns:
            ball_df = df[df['source'].isin(['collect_ball_A', 'collect_ball_B'])].copy()
        else:
            ball_df = pd.DataFrame()
        
        # Total = sum of ALL sources
        total_df = df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        total_df.rename(columns={'value': 'total_score'}, inplace=True)
        
        # Ball = sum of ball sources only
        if not ball_df.empty:
            ball_df = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
            ball_df.rename(columns={'value': 'ball_score'}, inplace=True)
        else:
            ball_df = pd.DataFrame(columns=['episode_id', 'player_id', 'ball_score'])
        
        merged = pd.merge(total_df, ball_df, on=['episode_id', 'player_id'], how='left')
        merged['ball_score'] = merged['ball_score'].fillna(0.0)
        all_records.append(merged)
    
    if not all_records:
        return None, None, []
    
    # Compute mean and SEM across files
    all_episodes = sorted(set().union(*[df['episode_id'].unique() for df in all_records]))
    all_players = sorted(set().union(*[df['player_id'].unique() for df in all_records]))
    
    mean_rows = []
    sem_rows = []
    for ep in all_episodes:
        mean_row = {'episode_id': ep}
        sem_row = {'episode_id': ep}
        for pid in all_players:
            values_total = []
            values_ball = []
            for df in all_records:
                row = df[(df['episode_id'] == ep) & (df['player_id'] == pid)]
                if not row.empty:
                    values_total.append(row['total_score'].values[0])
                    values_ball.append(row['ball_score'].values[0])
            mean_row[f'player_{pid}'] = np.mean(values_total) if values_total else 0.0
            sem_row[f'player_{pid}'] = np.std(values_total, ddof=1) / np.sqrt(len(values_total)) if len(values_total) > 1 else 0.0
        mean_row['total_avg'] = np.mean([mean_row[f'player_{pid}'] for pid in all_players])
        sem_row['total_avg'] = np.std([mean_row[f'player_{pid}'] for pid in all_players], ddof=1) / np.sqrt(len(all_players)) if len(all_players) > 1 else 0.0
        mean_rows.append(mean_row)
        sem_rows.append(sem_row)
    
    return pd.DataFrame(mean_rows), pd.DataFrame(sem_rows), all_players


# ==============================================================================
# Plotting
# ==============================================================================

def plot_per_player(mask_data, save_path, smooth_sigma=2.0):
    """
    Plot per-player average total score vs episode for all mask types.
    4 subplots (one per player), each with one line per mask type.
    """
    setup_style()
    _set_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('各智能体平均总得分——有效掩码对比',
                 fontsize=15, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    all_players = set()
    for mask_type, (df, sem_df, players) in mask_data.items():
        all_players.update(players)
    all_players = sorted(list(all_players))
    
    for idx, pid in enumerate(all_players):
        ax = axes[idx]
        col = f'player_{pid}'
        for mask_type, (df, sem_df, players) in mask_data.items():
            if col not in df.columns:
                continue
            episodes = df['episode_id'].values
            rewards = df[col].values
            sem_vals = sem_df[col].values
            
            # Smoothing
            if len(rewards), 2.0))
            else:
                sm_x, sm_mean, sm_sem = episodes, rewards, sem_vals
            
            color = MASK_COLORS[mask_type]
            ax.fill_between(sm_x, sm_mean - sm_sem, sm_mean + sm_sem,
                            alpha=0.18, color=color, edgecolor='none', linewidth=0, zorder=1)
            sns.lineplot(x=sm_x, y=sm_mean, label=MASK_NAMES[mask_type],
                         color=color, linewidth=2.0, ax=ax, zorder=3)
        
        ax.set_xlabel('回合', fontsize=11)
        ax.set_ylabel('平均总得分', fontsize=11)
        ax.set_title(f'智能体 {pid}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#cccccc')
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_total(mask_data, save_path, smooth_sigma=2.0):
    """
    Plot total average score vs episode for all mask types.
    1 plot with one line per mask type.
    """
    setup_style()
    _set_chinese_font()
    fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=150)
    
    stats_text = "每回合平均奖励：\n"
    for mask_type, (df, sem_df, players) in mask_data.items():
        episodes = df['episode_id'].values
        total = df['total_avg'].values
        total_sem = sem_df['total_avg'].values
        
        # Smoothing
        if len(total), 2.0))
        else:
            sm_x, sm_mean, sm_sem = episodes, total, total_sem
        
        color = MASK_COLORS[mask_type]
        ax.fill_between(sm_x, sm_mean - sm_sem, sm_mean + sm_sem,
                        alpha=0.18, color=color, edgecolor='none', linewidth=0, zorder=1)
        sns.lineplot(x=sm_x, y=sm_mean, label=MASK_NAMES[mask_type],
                     color=color, linewidth=2.0, ax=ax, zorder=3)
        
        mean_val = np.mean(total)
        stats_text += f"  {MASK_NAMES[mask_type]}: {mean_val:.2f}\n"
    
    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel('所有智能体平均总得分', fontsize=12)
    ax.set_title('总得分对比——有效掩码',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='有效掩码',
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
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_per_player_ball(mask_data, save_path, smooth_sigma=2.0):
    """
    Plot per-player average BALL score vs episode for all mask types.
    4 subplots (one per player), each with one line per mask type.
    """
    setup_style()
    _set_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('各智能体平均采球得分——有效掩码对比',
                 fontsize=15, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    all_players = set()
    for mask_type, (df, sem_df, players) in mask_data.items():
        all_players.update(players)
    all_players = sorted(list(all_players))
    
    for idx, pid in enumerate(all_players):
        ax = axes[idx]
        col = f'player_{pid}'
        for mask_type, (df, sem_df, players) in mask_data.items():
            if col not in df.columns:
                continue
            episodes = df['episode_id'].values
            rewards = df[col].values
            sem_vals = sem_df[col].values
            
            # Smoothing
            if len(rewards), 2.0))
            else:
                sm_x, sm_mean, sm_sem = episodes, rewards, sem_vals
            
            color = MASK_COLORS[mask_type]
            ax.fill_between(sm_x, sm_mean - sm_sem, sm_mean + sm_sem,
                            alpha=0.18, color=color, edgecolor='none', linewidth=0, zorder=1)
            sns.lineplot(x=sm_x, y=sm_mean, label=MASK_NAMES[mask_type],
                         color=color, linewidth=2.0, ax=ax, zorder=3)
        
        ax.set_xlabel('回合', fontsize=11)
        ax.set_ylabel('平均采球得分', fontsize=11)
        ax.set_title(f'智能体 {pid}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_total_ball(mask_data, save_path, smooth_sigma=2.0):
    """
    Plot total average BALL score vs episode for all mask types.
    1 plot with one line per mask type.
    """
    setup_style()
    _set_chinese_font()
    fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=150)
    
    stats_text = "每回合平均采球得分：\n"
    for mask_type, (df, sem_df, players) in mask_data.items():
        if 'ball_avg' not in df.columns:
            continue
        episodes = df['episode_id'].values
        total = df['ball_avg'].values
        total_sem = sem_df['ball_avg'].values
        
        # Smoothing
        if len(total), 2.0))
        else:
            sm_x, sm_mean, sm_sem = episodes, total, total_sem
        
        color = MASK_COLORS[mask_type]
        ax.fill_between(sm_x, sm_mean - sm_sem, sm_mean + sm_sem,
                        alpha=0.18, color=color, edgecolor='none', linewidth=0, zorder=1)
        sns.lineplot(x=sm_x, y=sm_mean, label=MASK_NAMES[mask_type],
                     color=color, linewidth=2.0, ax=ax, zorder=3)
        
        mean_val = np.mean(total)
        stats_text += f"  {MASK_NAMES[mask_type]}: {mean_val:.2f}\n"
    
    ax.set_xlabel('回合', fontsize=12)
    ax.set_ylabel('所有智能体平均采球得分', fontsize=12)
    ax.set_title('采球总得分对比——有效掩码',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='有效掩码',
              edgecolor='#cccccc')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                       edgecolor='gray'),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("  Valid Mask Comparison — Publication-Style Analysis")
    print("=" * 70)
    
    setup_style()
    _set_chinese_font()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Step 1: Discover files
    print("\n" + "-" * 70)
    print("[Step 1/4] Discovering files...")
    print("-" * 70)
    
    mask_data = discover_files()
    for mask_type, files in mask_data.items():
        print(f"  {MASK_NAMES[mask_type]}: {len(files)} files")
    
    if not mask_data:
        print("\n[ERROR] No valid files found. Exiting.")
        sys.exit(1)
    
    # Step 2: Load and aggregate
    print("\n" + "-" * 70)
    print("[Step 2/4] Loading and aggregating data...")
    print("-" * 70)
    
    aggregated = {}
    for mask_type, files in mask_data.items():
        print(f"\n  Processing: {MASK_NAMES[mask_type]} ({len(files)} files)...")
        mean_df, sem_df, players = load_and_aggregate(files, mask_type)
        if mean_df is not None:
            aggregated[mask_type] = (mean_df, sem_df, players)
            print(f"    Aggregated: {len(mean_df)} episodes, players={players}")
            
            # Save CSV
            csv_path = OUTPUT_DIR / f"{mask_type}_average.csv"
            mean_df.to_csv(csv_path, index=False)
            print(f"    Saved CSV: {csv_path.name}")
    
    if not aggregated:
        print("\n[ERROR] No valid data aggregated. Exiting.")
        sys.exit(1)
    
    # Step 3: Generate plots
    print("\n" + "-" * 70)
    print("[Step 3/4] Generating plots...")
    print("-" * 70)
    
    # Plot 1: Per-player total score
    print("\n  Plot 1: Per-player total score comparison...")
    plot_per_player(aggregated, str(OUTPUT_DIR / "per_player_total_comparison.png"))
    
    # Plot 2: Total average
    print("\n  Plot 2: Total average comparison...")
    plot_total(aggregated, str(OUTPUT_DIR / "total_total_comparison.png"))
    
    # Plot 3: Per-player ball score
    print("\n  Plot 3: Per-player ball score comparison...")
    plot_per_player_ball(aggregated, str(OUTPUT_DIR / "per_player_ball_comparison.png"))
    
    # Plot 4: Total ball average
    print("\n  Plot 4: Total ball average comparison...")
    plot_total_ball(aggregated, str(OUTPUT_DIR / "total_ball_comparison.png"))
    
    # Step 4: Save combined data
    print("\n" + "-" * 70)
    print("[Step 4/4] Saving combined data...")
    print("-" * 70)
    
    combined = None
    for mask_type, (df, sem_df, players) in aggregated.items():
        df_copy = df.copy()
        df_copy['mask_type'] = MASK_NAMES[mask_type]
        if combined is None:
            combined = df_copy
        else:
            combined = pd.concat([combined, df_copy], ignore_index=True)
    
    if combined is not None:
        combined_path = OUTPUT_DIR / "all_masks_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"  [SAVED] {combined_path}")
    
    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)
    print(f"\n  Output files in: {OUTPUT_DIR}")
    print("  - per_player_total_comparison.png")
    print("  - total_total_comparison.png")
    print("  - per_player_ball_comparison.png")
    print("  - total_ball_comparison.png")
    print("  - *_average.csv")


if __name__ == "__main__":
    main()
