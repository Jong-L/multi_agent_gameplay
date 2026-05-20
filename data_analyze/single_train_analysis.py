"""
Single Training Analysis
Analyze individual training runs and plot total ball reward with mean reference line.
使用两级平滑管线 (高斯滤波 → 样条插值)，产出论文级图表。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from publication_plot_utils import (
    setup_style, save_figure,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
    COLORS,
)

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_and_aggregate_ball_data(file_path):
    """Load CSV and aggregate ball scores by episode"""
    df = pd.read_csv(file_path)
    ball_df = df[df['source'].isin(BALL_SOURCES)]
    
    aggregated = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    aggregated.columns = ['episode_id', 'player_id', 'total_ball_reward']
    
    episode_totals = aggregated.groupby('episode_id')['total_ball_reward'].sum().reset_index()
    episode_totals.columns = ['episode_id', 'total_ball_reward']
    
    print(f"  Loaded: {Path(file_path).name}")
    print(f"    Raw records: {len(df)}")
    print(f"    Ball records: {len(ball_df)}")
    print(f"    Episodes: {len(episode_totals)}")
    print(f"    Mean total reward: {episode_totals['total_ball_reward'].mean():.2f}")
    
    return episode_totals


def plot_single_training(episode_data, title, save_path, smooth_window=5):
    """Plot total ball reward with mean reference line"""
    setup_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    episodes = episode_data['episode_id'].values
    rewards = episode_data['total_ball_reward'].values
    mean_reward = episode_data['total_ball_reward'].mean()
    
    # ── 两级平滑管线 ──────────────────────────────────
    x_smooth, y_smooth, y_lower, y_upper = prepare_curve(
        episodes, rewards,
        errors=np.full_like(rewards, 4.0),  # 固定 4 单位半带宽
        smooth_window=smooth_window,
    )
    
    plot_with_fill(ax, x_smooth, y_smooth, y_lower, y_upper,
                   color=COLORS['blue'], label='Total Ball Reward')
    
    # ── 均值参考线 ────────────────────────────────────
    ax.axhline(y=mean_reward, color=COLORS['orange'],
               linestyle='--', linewidth=2,
               label=f'Mean: {mean_reward:.2f}')
    
    style_axes(ax, xlabel='Episode', ylabel='Total Ball Reward', title=title,
               xlim=(0, None), ylim=(0, None))
    
    # ── 统计信息 ──────────────────────────────────────
    stats_text = f"Episodes: {len(episode_data)}\n"
    stats_text += f"Mean: {mean_reward:.2f}\n"
    stats_text += f"Max: {episode_data['total_ball_reward'].max():.2f}\n"
    stats_text += f"Min: {episode_data['total_ball_reward'].min():.2f}"
    add_stats_box(ax, stats_text, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, save_path)
    return fig


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    files = {
        'Linear Potential': {
            'path': f"{base_dir}\\train_rewards_ball_lrrs_2026-04-24_18-24-19_pid29760.csv",
            'output': f"{base_dir}\\single_train_linear.png"
        },
        'Exponential Potential': {
            'path': f"{base_dir}\\train_rewards_ball_exprs_2026-04-24_20-57-21_pid29628.csv",
            'output': f"{base_dir}\\single_train_exponential.png"
        }
    }
    
    print("="*60)
    print("Single Training Analysis")
    print("="*60)
    
    for name, info in files.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print('='*60)
        
        if not Path(info['path']).exists():
            print(f"[ERROR] File not found: {info['path']}")
            continue
        
        episode_data = load_and_aggregate_ball_data(info['path'])
        plot_single_training(episode_data,
                             f'Single Training: {name}',
                             info['output'],
                             smooth_window=5)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    for name, info in files.items():
        print(f"  - {info['output']}")


if __name__ == "__main__":
    main()
