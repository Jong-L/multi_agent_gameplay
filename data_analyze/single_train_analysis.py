"""
Single Training Analysis
Analyze individual training runs and plot total ball reward with mean reference line
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_and_aggregate_ball_data(file_path):
    """Load CSV and aggregate ball scores by episode"""
    df = pd.read_csv(file_path)
    ball_df = df[df['source'].isin(BALL_SOURCES)]
    
    # Aggregate by episode and player
    aggregated = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    aggregated.columns = ['episode_id', 'player_id', 'total_ball_reward']
    
    # Sum across all players per episode
    episode_totals = aggregated.groupby('episode_id')['total_ball_reward'].sum().reset_index()
    episode_totals.columns = ['episode_id', 'total_ball_reward']
    
    print(f"  Loaded: {Path(file_path).name}")
    print(f"    Raw records: {len(df)}")
    print(f"    Ball records: {len(ball_df)}")
    print(f"    Episodes: {len(episode_totals)}")
    print(f"    Mean total reward: {episode_totals['total_ball_reward'].mean():.2f}")
    
    return episode_totals


def smooth_curve(y, window=10):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_single_training(episode_data, title, save_path, smooth_window=10):
    """Plot total ball reward with mean reference line"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    episodes = episode_data['episode_id'].values
    rewards = episode_data['total_ball_reward'].values
    mean_reward = episode_data['total_ball_reward'].mean()
    
    # Smoothing
    if smooth_window > 1 and len(rewards) > smooth_window:
        smoothed_rewards = smooth_curve(rewards, smooth_window)
        smoothed_episodes = episodes[smooth_window-1:]
    else:
        smoothed_rewards = rewards
        smoothed_episodes = episodes
    
    # Plot reward curve
    sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                color='#0173B2',
                linewidth=2,
                label='Total Ball Reward',
                ax=ax)
    
    # Plot mean reference line
    ax.axhline(y=mean_reward, 
               color='#DE8F05', 
               linestyle='--', 
               linewidth=2,
               label=f'Mean: {mean_reward:.2f}')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Ball Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics text box
    stats_text = f"Episodes: {len(episode_data)}\n"
    stats_text += f"Mean: {mean_reward:.2f}\n"
    stats_text += f"Max: {episode_data['total_ball_reward'].max():.2f}\n"
    stats_text += f"Min: {episode_data['total_ball_reward'].min():.2f}"
    
    ax.text(0.98, 0.98, stats_text, 
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=10,
           family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    # Define files to analyze
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
        
        # Load data
        episode_data = load_and_aggregate_ball_data(info['path'])
        
        # Plot
        fig = plot_single_training(episode_data, 
                                   f'Single Training: {name}', 
                                   info['output'],
                                   smooth_window=10)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    for name, info in files.items():
        print(f"  - {info['output']}")
    
    print("\nDisplaying plots... (close windows to exit)")
    plt.show()


if __name__ == "__main__":
    main()
