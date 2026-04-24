"""
Nearest vs All Potential Calculation Comparison
Compare ball collection rewards between nearest-ball and all-balls potential schemes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set seaborn style for publication
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Color palette (colorblind-friendly)
SCHEME_COLORS = {
    'nearest': '#0173B2',  # Blue
    'all': '#DE8F05'       # Orange
}

SCHEME_NAMES = {
    'nearest': 'Nearest Ball Only',
    'all': 'All Balls'
}


def load_data(base_dir):
    """Load both scheme data"""
    data = {}
    
    for scheme in ['nearest', 'all']:
        file_path = f"{base_dir}\\{scheme}_average_ball_reward.csv"
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            data[scheme] = {
                'df': df,
                'name': SCHEME_NAMES[scheme],
                'episodes': df['episode_id'].values
            }
            print(f"Loaded {scheme}: {len(df)} episodes")
        else:
            print(f"Warning: {file_path} not found")
    
    return data


def get_player_columns(df):
    """Get player column names"""
    player_cols = [col for col in df.columns if col.startswith('player_') and col.endswith('_ball_avg')]
    return sorted(player_cols)


def smooth_curve(y, window=10):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_player_comparison(data, save_path=None, smooth_window=5):
    """Window 1: 4 subplots (one per player), each with 2 curves"""
    first_scheme = list(data.keys())[0]
    player_cols = get_player_columns(data[first_scheme]['df'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle('Ball Collection Reward: Nearest vs All Balls\n(Per Player Comparison)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        for scheme_key in ['nearest', 'all']:
            if scheme_key not in data:
                continue
            df = data[scheme_key]['df']
            episodes = df['episode_id'].values
            rewards = df[player_col].values
            
            # Smoothing
            if smooth_window > 1 and len(rewards) > smooth_window:
                smoothed_rewards = smooth_curve(rewards, smooth_window)
                smoothed_episodes = episodes[smooth_window-1:]
            else:
                smoothed_rewards = rewards
                smoothed_episodes = episodes
            
            sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                        label=SCHEME_NAMES[scheme_key],
                        color=SCHEME_COLORS[scheme_key],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Reward', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_comparison(data, save_path=None, smooth_window=5):
    """Window 2: 1 subplot with total average, 2 curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    for scheme_key in ['nearest', 'all']:
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        episodes = df['episode_id'].values
        total_rewards = df['total_ball_avg_reward'].values
        
        # Smoothing
        if smooth_window > 1 and len(total_rewards) > smooth_window:
            smoothed_rewards = smooth_curve(total_rewards, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_rewards = total_rewards
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                    label=SCHEME_NAMES[scheme_key],
                    color=SCHEME_COLORS[scheme_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward', fontsize=12)
    ax.set_title('Total Ball Collection Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Potential Scheme')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Statistics annotation
    stats_text = "Mean Reward per Episode:\n"
    for scheme_key in ['nearest', 'all']:
        if scheme_key in data:
            avg_reward = data[scheme_key]['df']['total_ball_avg_reward'].mean()
            stats_text += f"{SCHEME_NAMES[scheme_key]}: {avg_reward:.1f}\n"
    
    improvement = 0
    if 'nearest' in data and 'all' in data:
        nearest_avg = data['nearest']['df']['total_ball_avg_reward'].mean()
        all_avg = data['all']['df']['total_ball_avg_reward'].mean()
        improvement = ((nearest_avg - all_avg) / all_avg) * 100
        stats_text += f"\nImprovement: +{improvement:.1f}%"
    
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
    print("SUMMARY: Nearest vs All Balls Potential Calculation")
    print("="*60)
    
    for scheme_key in ['nearest', 'all']:
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        print(f"\n{SCHEME_NAMES[scheme_key]}:")
        print(f"  Episodes: {len(df)}")
        
        player_cols = get_player_columns(df)
        for col in player_cols:
            player_id = col.split('_')[1]
            avg = df[col].mean()
            print(f"  Player {player_id} avg: {avg:.2f}")
        
        total_avg = df['total_ball_avg_reward'].mean()
        print(f"  Total avg per episode: {total_avg:.2f}")
    
    # Calculate improvement
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
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    print("="*60)
    print("Nearest vs All Balls: Potential Calculation Comparison")
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
    print("\nCreating plots...")
    
    fig1 = plot_player_comparison(data, 
                                  save_path=f"{base_dir}\\nearest_vs_all_player_comparison.png",
                                  smooth_window=10)
    
    fig2 = plot_total_comparison(data, 
                                 save_path=f"{base_dir}\\nearest_vs_all_total_comparison.png",
                                 smooth_window=10)
    
    print("\n" + "="*60)
    print("Plots generated!")
    print("="*60)
    print(f"\n1. Player comparison: nearest_vs_all_player_comparison.png")
    print(f"2. Total comparison: nearest_vs_all_total_comparison.png")
    
    # print("\nDisplaying plots... (close windows to exit)")
    # plt.show()


if __name__ == "__main__":
    main()
