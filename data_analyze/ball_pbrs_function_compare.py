"""
Ball Reward Potential Function Comparison
===========================================
比较不同势能函数 (Linear, Inverse Prop, Exponential) 的奖励曲线。
使用两级平滑管线，产出论文级图表。

Window 1: 4 subplots (one per player), each with 3 curves
Window 2: 1 subplot with total average, 3 curves
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from publication_plot_utils import (
    setup_style, save_figure,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
    COLORS,
)

# ── 配置 ────────────────────────────────────────────
FUNC_COLORS = {
    'lr':   COLORS['blue'],     # #0173B2
    'invp': COLORS['orange'],   # #DE8F05
    'exp':  COLORS['green'],    # #029E73
}
FUNC_NAMES = {
    'lr':   'Linear',
    'invp': 'Inverse Prop',
    'exp':  'Exponential',
}


def load_data(base_dir):
    """Load all three function type data"""
    data = {}
    for func in ['lr', 'invp', 'exp']:
        file_path = f"{base_dir}\\{func}_average_ball_reward.csv"
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            data[func] = {
                'df': df,
                'name': FUNC_NAMES[func],
            }
            print(f"Loaded {func}: {len(df)} episodes")
        else:
            print(f"Warning: {file_path} not found")
    return data


def get_player_columns(df):
    """Get player column names"""
    return sorted([c for c in df.columns
                   if c.startswith('player_') and c.endswith('_ball_avg')])


def plot_player_comparison(data, save_path=None, smooth_window=5):
    """4 subplots (one per player), each with 3 curves"""
    first_func = list(data.keys())[0]
    player_cols = get_player_columns(data[first_func]['df'])
    
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle('Ball Collection Reward: Linear vs Inverse vs Exponential\n'
                 '(Per Player Comparison)',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        for func_key, color in FUNC_COLORS.items():
            if func_key not in data:
                continue
            df = data[func_key]['df']
            episodes = df['episode_id'].values
            rewards = df[player_col].values
            
            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, rewards, errors=np.zeros_like(rewards),
                smooth_window=smooth_window,
            )
            
            # 此处没有 SEM 数据，用常量半带 (±2)
            y_l = y_s - 2
            y_u = y_s + 2
            
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=color, label=FUNC_NAMES[func_key],
                           linewidth=1.5)
        
        style_axes(ax, xlabel='Episode', ylabel='Average Ball Reward',
                   title=f'Player {player_id}')
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total_comparison(data, save_path=None, smooth_window=5):
    """1 subplot with total average, 3 curves"""
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    stats_text = "Mean Reward per Episode:\n"
    
    for func_key, color in FUNC_COLORS.items():
        if func_key not in data:
            continue
        df = data[func_key]['df']
        episodes = df['episode_id'].values
        total = df['total_ball_avg_reward'].values
        
        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total, errors=np.zeros_like(total),
            smooth_window=smooth_window,
        )
        y_l = y_s - 3
        y_u = y_s + 3
        
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=color, label=FUNC_NAMES[func_key],
                       linewidth=1.5)
        
        avg_reward = total.mean()
        stats_text += f"{FUNC_NAMES[func_key]}: {avg_reward:.1f}\n"
    
    style_axes(ax, xlabel='Episode', ylabel='Total Average Ball Reward',
               title='Total Ball Collection Reward by Potential Function',
               legend_kwargs={'title': 'Potential Function'})
    
    add_stats_box(ax, stats_text)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def print_summary(data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for func_key in ['lr', 'invp', 'exp']:
        if func_key not in data:
            continue
        df = data[func_key]['df']
        print(f"\n{FUNC_NAMES[func_key]}:")
        print(f"  Episodes: {len(df)}")
        for col in get_player_columns(df):
            pid = col.split('_')[1]
            print(f"  Player {pid} avg: {df[col].mean():.2f}")
        print(f"  Total avg per episode: {df['total_ball_avg_reward'].mean():.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    print("="*60)
    print("Ball Reward Comparison (Publication Style)")
    print("="*60)
    
    print("\nLoading data...")
    data = load_data(base_dir)
    
    if len(data) == 0:
        print("No data files found!")
        return
    
    print_summary(data)
    
    print("\nCreating plots...")
    plot_player_comparison(data,
                           save_path=f"{base_dir}\\player_ball_reward_comparison.png",
                           smooth_window=5)
    plot_total_comparison(data,
                          save_path=f"{base_dir}\\total_ball_reward_comparison.png",
                          smooth_window=5)
    
    print("\n" + "="*60)
    print("Plots generated at 300 DPI!")
    print("="*60)


if __name__ == "__main__":
    main()
