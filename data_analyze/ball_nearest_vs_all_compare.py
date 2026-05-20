"""
Nearest vs All Potential Calculation Comparison
=================================================
比较 nearest-ball 和 all-balls 势能方案的球收集奖励。
使用两级平滑管线，产出论文级图表。

Window 1: 4 subplots (one per player), each with 2 curves
Window 2: 1 subplot with total average, 2 curves
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
SCHEME_COLORS = {
    'nearest': COLORS['blue'],    # #0173B2
    'all':     COLORS['orange'],  # #DE8F05
}
SCHEME_NAMES = {
    'nearest': 'Nearest Ball Only',
    'all':     'All Balls',
}


def load_data(base_dir):
    """Load both scheme data"""
    data = {}
    for scheme in ['nearest', 'all']:
        file_path = f"{base_dir}\\{scheme}_average_ball_reward.csv"
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            data[scheme] = {'df': df, 'name': SCHEME_NAMES[scheme]}
            print(f"Loaded {scheme}: {len(df)} episodes")
        else:
            print(f"Warning: {file_path} not found")
    return data


def get_player_columns(df):
    """Get player column names"""
    return sorted([c for c in df.columns
                   if c.startswith('player_') and c.endswith('_ball_avg')])


def plot_player_comparison(data, save_path=None, smooth_window=5):
    """4 subplots (one per player), each with 2 curves"""
    first_scheme = list(data.keys())[0]
    player_cols = get_player_columns(data[first_scheme]['df'])
    
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle('Ball Collection Reward: Nearest vs All Balls\n'
                 '(Per Player Comparison)',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        for scheme_key, color in SCHEME_COLORS.items():
            if scheme_key not in data:
                continue
            df = data[scheme_key]['df']
            episodes = df['episode_id'].values
            rewards = df[player_col].values
            
            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, rewards, errors=np.zeros_like(rewards),
                smooth_window=smooth_window,
            )
            y_l = y_s - 2
            y_u = y_s + 2
            
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=color, label=SCHEME_NAMES[scheme_key],
                           linewidth=1.5)
        
        style_axes(ax, xlabel='Episode', ylabel='Average Ball Reward',
                   title=f'Player {player_id}')
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total_comparison(data, save_path=None, smooth_window=5):
    """1 subplot with total average, 2 curves"""
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    stats_text = "Mean Reward per Episode:\n"
    
    for scheme_key, color in SCHEME_COLORS.items():
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        episodes = df['episode_id'].values
        total = df['total_ball_avg_reward'].values
        
        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total, errors=np.zeros_like(total),
            smooth_window=smooth_window,
        )
        y_l = y_s - 3
        y_u = y_s + 3
        
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=color, label=SCHEME_NAMES[scheme_key],
                       linewidth=1.5)
        
        avg_reward = total.mean()
        stats_text += f"{SCHEME_NAMES[scheme_key]}: {avg_reward:.1f}\n"
    
    # 计算改善幅度
    if 'nearest' in data and 'all' in data:
        nearest_avg = data['nearest']['df']['total_ball_avg_reward'].mean()
        all_avg = data['all']['df']['total_ball_avg_reward'].mean()
        improvement = ((nearest_avg - all_avg) / all_avg) * 100
        stats_text += f"\nImprovement: +{improvement:.1f}%"
    
    style_axes(ax, xlabel='Episode', ylabel='Total Average Ball Reward',
               title='Total Ball Collection Reward Comparison',
               legend_kwargs={'title': 'Potential Scheme'})
    
    add_stats_box(ax, stats_text)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def print_summary(data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY: Nearest vs All Balls Potential Calculation")
    print("="*60)
    for scheme_key in ['nearest', 'all']:
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        print(f"\n{SCHEME_NAMES[scheme_key]}:")
        print(f"  Episodes: {len(df)}")
        for col in get_player_columns(df):
            pid = col.split('_')[1]
            print(f"  Player {pid} avg: {df[col].mean():.2f}")
        print(f"  Total avg per episode: {df['total_ball_avg_reward'].mean():.2f}")
    
    if 'nearest' in data and 'all' in data:
        nearest_avg = data['nearest']['df']['total_ball_avg_reward'].mean()
        all_avg = data['all']['df']['total_ball_avg_reward'].mean()
        improvement = ((nearest_avg - all_avg) / all_avg) * 100
        print(f"\n{'='*60}")
        print(f"KEY FINDING:")
        print(f"  Nearest-only scheme improves ball collection by {improvement:.1f}%")
        print(f"  ({nearest_avg:.1f} vs {all_avg:.1f} average reward)")
        print("="*60)


def main():
    base_dir = r"experiment_data\game_reward_log"
    
    print("="*60)
    print("Nearest vs All Balls: Potential Calculation Comparison")
    print("="*60)
    
    print("\nLoading data...")
    data = load_data(base_dir)
    
    if len(data) == 0:
        print("No data files found!")
        return
    
    print_summary(data)
    
    print("\nCreating plots...")
    plot_player_comparison(data,
                           save_path=f"{base_dir}\\nearest_vs_all_player_comparison.png",
                           smooth_window=5)
    plot_total_comparison(data,
                          save_path=f"{base_dir}\\nearest_vs_all_total_comparison.png",
                          smooth_window=5)
    
    print("\n" + "="*60)
    print("Plots generated at 300 DPI!")
    print("="*60)


if __name__ == "__main__":
    main()
