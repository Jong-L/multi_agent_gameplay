"""
Ball Reward Potential Function Comparison (Seaborn Style)
Plot ball reward curves for different potential functions (LR, INVP, EXP)
Window 1: 4 subplots (one per player), each with 3 curves (LR, INVP, EXP)
Window 2: 1 subplot with total average, 3 curves (LR, INVP, EXP)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set seaborn style for publication
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

# Color palette for functions (colorblind-friendly)
FUNC_COLORS = {
    'lr': '#0173B2',      # Blue
    'invp': '#DE8F05',    # Orange
    'exp': '#029E73'      # Green
}

FUNC_NAMES = {
    'lr': 'Linear',
    'invp': 'Inverse Prop',
    'exp': 'Exponential'
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
                'episodes': df['episode_id'].values
            }
            print(f"Loaded {func}: {len(df)} episodes")
        else:
            print(f"Warning: {file_path} not found")
    
    return data


def get_player_columns(df):
    """Get player column names from dataframe"""
    player_cols = [col for col in df.columns if col.startswith('player_') and col.endswith('_ball_avg')]
    return sorted(player_cols)


def smooth_curve(y, window=5):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_player_comparison(data, save_path=None, smooth_window=5):
    """
    Window 1: 4 subplots (one per player)
    Each subplot shows 3 curves for LR, INVP, EXP
    """
    first_func = list(data.keys())[0]
    player_cols = get_player_columns(data[first_func]['df'])
    num_players = len(player_cols)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle('Ball Collection Reward Comparison by Player', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        # Plot each function type
        for func_key in ['lr', 'invp', 'exp']:
            if func_key not in data:
                continue
            df = data[func_key]['df']
            episodes = df['episode_id'].values
            rewards = df[player_col].values
            
            # Apply smoothing
            if smooth_window > 1 and len(rewards) > smooth_window:
                smoothed_rewards = smooth_curve(rewards, smooth_window)
                smoothed_episodes = episodes[smooth_window-1:]
            else:
                smoothed_rewards = rewards
                smoothed_episodes = episodes
            
            sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                        label=FUNC_NAMES[func_key],
                        color=FUNC_COLORS[func_key],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Reward', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_comparison(data, save_path=None, smooth_window=5):
    """
    Window 2: 1 subplot with total average
    Shows 3 curves for LR, INVP, EXP
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each function type
    for func_key in ['lr', 'invp', 'exp']:
        if func_key not in data:
            continue
        df = data[func_key]['df']
        episodes = df['episode_id'].values
        total_rewards = df['total_ball_avg_reward'].values
        
        # Apply smoothing
        if smooth_window > 1 and len(total_rewards) > smooth_window:
            smoothed_rewards = smooth_curve(total_rewards, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_rewards = total_rewards
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                    label=FUNC_NAMES[func_key],
                    color=FUNC_COLORS[func_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward', fontsize=12)
    ax.set_title('Total Ball Collection Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Potential Function')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics annotation
    stats_text = "Mean Reward per Episode:\n"
    for func_key in ['lr', 'invp', 'exp']:
        if func_key in data:
            avg_reward = data[func_key]['df']['total_ball_avg_reward'].mean()
            stats_text += f"{FUNC_NAMES[func_key]}: {avg_reward:.1f}\n"
    
    ax.text(0.98, 0.02, stats_text, 
           transform=ax.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=10,
           family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
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
        
        player_cols = get_player_columns(df)
        for col in player_cols:
            player_id = col.split('_')[1]
            avg = df[col].mean()
            print(f"  Player {player_id} avg: {avg:.2f}")
        
        total_avg = df['total_ball_avg_reward'].mean()
        print(f"  Total avg per episode: {total_avg:.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    print("="*60)
    print("Ball Reward Comparison (Seaborn Style)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = load_data(base_dir)
    
    if len(data) == 0:
        print("No data files found!")
        return
    
    # Print summary
    print_summary(data)
    
    # Create plots
    print("\nCreating plots with Seaborn...")
    
    # Window 1: Player comparison
    fig1 = plot_player_comparison(data, 
                                  save_path=f"{base_dir}\\player_ball_reward_comparison.png",
                                  smooth_window=10)
    
    # Window 2: Total comparison
    fig2 = plot_total_comparison(data, 
                                 save_path=f"{base_dir}\\total_ball_reward_comparison.png",
                                 smooth_window=10)
    
    print("\n" + "="*60)
    print("Plots generated!")
    print("="*60)
    print(f"\n1. Player comparison: player_ball_reward_comparison.png")
    print(f"2. Total comparison: total_ball_reward_comparison.png")
    print("\nBoth saved at 300 DPI for publication quality")
    
    # Show plots
    # print("\nDisplaying plots... (close windows to exit)")
    # plt.show()


if __name__ == "__main__":
    main()
