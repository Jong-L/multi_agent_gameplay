"""
Terminal State Zero Comparison Analysis
Compares ball collection rewards between terminal_zero and terminal_not_zero
potential function designs.

Each type has 5 data files:
- terminal_zero: lrrs_terminal_zero_*.csv
- terminal_not_zero: lrrs_terminal_not_zero_*.csv

Generates:
1. Player average ball reward vs episode (4 subplots)
2. Total average ball reward vs episode (1 plot)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set seaborn style for publication
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Color palette (colorblind-friendly)
TYPE_COLORS = {
    'terminal_zero': '#0173B2',      # Blue
    'terminal_not_zero': '#DE8F05'   # Orange
}

TYPE_NAMES = {
    'terminal_zero': 'Terminal = 0',
    'terminal_not_zero': 'Terminal ≠ 0'
}

# Ball collection sources
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_data_files(base_dir):
    """
    Load all data files organized by type
    Returns: dict {type: [list of file paths]}
    """
    data_files = {
        'terminal_zero': [],
        'terminal_not_zero': []
    }
    
    base_path = Path(base_dir)
    
    # Find files by pattern
    for file in base_path.glob("*.csv"):
        fname = file.name
        if "terminal_zero_" in fname and "terminal_not_zero" not in fname:
            data_files['terminal_zero'].append(str(file))
        elif "terminal_not_zero" in fname:
            data_files['terminal_not_zero'].append(str(file))
    
    # Sort files for consistency
    for key in data_files:
        data_files[key].sort()
    
    return data_files


def load_and_process_file(file_path):
    """Load a single CSV file and return processed dataframe"""
    df = pd.read_csv(file_path)
    # Filter only ball collection sources
    ball_df = df[df['source'].isin(BALL_SOURCES)]
    return ball_df


def aggregate_data_by_type(data_files):
    """
    Aggregate ball reward data for each type across all files
    Returns: dict {type: aggregated_data}
    """
    aggregated = {}
    
    for type_name, file_list in data_files.items():
        print(f"\nProcessing {TYPE_NAMES[type_name]} ({len(file_list)} files)...")
        
        episode_counts = {}
        all_data = []
        
        for fp in file_list:
            df = load_and_process_file(fp)
            max_ep = df['episode_id'].max() if len(df) > 0 else 0
            episode_counts[Path(fp).name] = max_ep
            all_data.append(df)
        
        # Check episode consistency
        unique_counts = set(episode_counts.values())
        if len(unique_counts) > 1:
            print(f"  [!] Warning: Inconsistent episode counts: {episode_counts}")
            print("  [!] Analysis aborted - all files must have the same number of episodes")
            return None
        
        max_episodes = list(unique_counts)[0]
        print(f"  [OK] Consistent: {max_episodes} episodes")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        aggregated[type_name] = {
            'df': combined_df,
            'max_episodes': max_episodes,
            'num_files': len(file_list)
        }
    
    return aggregated


def compute_episode_stats(aggregated_data):
    """
    Compute per-episode ball reward statistics for each type
    Returns: dict with player ball rewards, total ball rewards per type
    """
    stats = {}
    
    for type_name, data in aggregated_data.items():
        df = data['df']
        max_episodes = data['max_episodes']
        num_files = data['num_files']
        
        # Get all unique player IDs
        all_players = sorted(df['player_id'].unique())
        
        episode_stats = []
        
        for episode in range(1, max_episodes + 1):
            ep_data = {'episode_id': episode}
            
            # Filter data for this episode
            ep_df = df[df['episode_id'] == episode]
            
            # Compute ball reward per player
            player_rewards = {}
            
            for player_id in all_players:
                player_df = ep_df[ep_df['player_id'] == player_id]
                
                # Sum of ball collection rewards
                total_reward = player_df['value'].sum()
                player_rewards[player_id] = total_reward / num_files  # Average across files
                
                ep_data[f'player_{player_id}_ball_reward'] = player_rewards[player_id]
            
            # Total across all players
            total_reward = sum(player_rewards.values())
            ep_data['total_ball_reward'] = total_reward
            
            episode_stats.append(ep_data)
        
        stats_df = pd.DataFrame(episode_stats)
        stats[type_name] = {
            'df': stats_df,
            'players': all_players,
            'max_episodes': max_episodes
        }
    
    return stats


def smooth_curve(y, window=5):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_player_ball_reward_comparison(stats, save_path=None, smooth_window=10):
    """
    Plot 1: Each player's average ball reward vs episode
    4 subplots (one per player), each with 2 curves (terminal_zero vs terminal_not_zero)
    """
    # Get player list from first type
    first_type = list(stats.keys())[0]
    players = stats[first_type]['players']
    num_players = len(players)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Player Average Ball Reward vs Episode (Terminal State Comparison)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_ball_reward'
        
        # Plot each type
        for type_name in ['terminal_zero', 'terminal_not_zero']:
            if type_name not in stats:
                continue
            
            df = stats[type_name]['df']
            episodes = df['episode_id'].values
            rewards = df[col_name].values
            
            # Apply smoothing
            if smooth_window > 1 and len(rewards) > smooth_window:
                smoothed_rewards = smooth_curve(rewards, smooth_window)
                smoothed_episodes = episodes[smooth_window-1:]
            else:
                smoothed_rewards = rewards
                smoothed_episodes = episodes
            
            sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                        label=TYPE_NAMES[type_name],
                        color=TYPE_COLORS[type_name],
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


def plot_total_ball_reward_comparison(stats, save_path=None, smooth_window=10):
    """
    Plot 2: All players' average ball reward vs episode
    1 plot with 2 curves (terminal_zero vs terminal_not_zero)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each type
    for type_name in ['terminal_zero', 'terminal_not_zero']:
        if type_name not in stats:
            continue
        
        df = stats[type_name]['df']
        episodes = df['episode_id'].values
        total_rewards = df['total_ball_reward'].values
        
        # Apply smoothing
        if smooth_window > 1 and len(total_rewards) > smooth_window:
            smoothed_rewards = smooth_curve(total_rewards, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_rewards = total_rewards
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_rewards,
                    label=TYPE_NAMES[type_name],
                    color=TYPE_COLORS[type_name],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward', fontsize=12)
    ax.set_title('Total Ball Reward vs Episode (Terminal State Comparison)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Potential Function')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics annotation (positioned to avoid legend overlap)
    stats_text = "Mean Ball Reward:\n"
    for type_name in ['terminal_zero', 'terminal_not_zero']:
        if type_name in stats:
            avg_reward = stats[type_name]['df']['total_ball_reward'].mean()
            stats_text += f"{TYPE_NAMES[type_name]}: {avg_reward:.1f}\n"
    
    ax.text(0.02, 0.02, stats_text, 
           transform=ax.transAxes,
           verticalalignment='bottom',
           horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=10,
           family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def save_summary_csv(stats, output_dir):
    """Save summary statistics to CSV files"""
    summary_dir = Path(output_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    for type_name, data in stats.items():
        df = data['df']
        output_path = summary_dir / f"{type_name}_episode_stats.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    # Create overall summary
    summary_data = []
    for type_name in ['terminal_zero', 'terminal_not_zero']:
        if type_name not in stats:
            continue
        
        df = stats[type_name]['df']
        row = {
            'type': type_name,
            'type_name': TYPE_NAMES[type_name],
            'mean_total_ball_reward': df['total_ball_reward'].mean(),
            'max_episodes': stats[type_name]['max_episodes']
        }
        
        # Add per-player stats
        players = stats[type_name]['players']
        for player_id in players:
            row[f'player_{player_id}_mean_reward'] = df[f'player_{player_id}_ball_reward'].mean()
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = summary_dir / "overall_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    return summary_df


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS - Ball Collection Reward")
    print("="*70)
    
    for type_name in ['terminal_zero', 'terminal_not_zero']:
        if type_name not in stats:
            continue
        
        data = stats[type_name]
        df = data['df']
        players = data['players']
        
        print(f"\n{TYPE_NAMES[type_name]}:")
        print(f"  Episodes: {data['max_episodes']}")
        print(f"  Mean Total Ball Reward: {df['total_ball_reward'].mean():.2f}")
        
        print("  Per-player stats:")
        for player_id in players:
            col = f'player_{player_id}_ball_reward'
            avg = df[col].mean()
            print(f"    Player {player_id}: {avg:.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\potential_func_terminal_zero_comparison"
    summary_dir = Path(base_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Terminal State Zero Comparison - Ball Reward Analysis")
    print("="*70)
    
    # Step 1: Load data file paths
    print("\n[1/5] Loading data file paths...")
    data_files = load_data_files(base_dir)
    for tn, files in data_files.items():
        print(f"  {TYPE_NAMES[tn]}: {len(files)} files")
    
    # Step 2: Aggregate data
    print("\n[2/5] Aggregating data...")
    aggregated = aggregate_data_by_type(data_files)
    
    if aggregated is None:
        print("\n[ERROR] Episode count inconsistency detected. Analysis aborted.")
        return
    
    # Step 3: Compute episode statistics
    print("\n[3/5] Computing episode statistics...")
    stats = compute_episode_stats(aggregated)
    
    # Step 4: Save summary CSV
    print("\n[4/5] Saving summary CSV files...")
    summary_df = save_summary_csv(stats, base_dir)
    
    # Step 5: Generate plots
    print("\n[5/5] Generating plots with Seaborn...")
    
    # Plot 1: Player ball reward comparison
    print("  Creating player ball reward comparison...")
    fig1 = plot_player_ball_reward_comparison(stats, 
                                              save_path=summary_dir / "player_ball_reward_comparison.png",
                                              smooth_window=10)
    
    # Plot 2: Total ball reward comparison
    print("  Creating total ball reward comparison...")
    fig2 = plot_total_ball_reward_comparison(stats, 
                                             save_path=summary_dir / "total_ball_reward_comparison.png",
                                             smooth_window=10)
    
    # Print summary
    print_summary(stats)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nGenerated files in {summary_dir}:")
    print("  - player_ball_reward_comparison.png")
    print("  - total_ball_reward_comparison.png")
    print("  - *_episode_stats.csv (per-type episode data)")
    print("  - overall_summary.csv")
    print("\nAll plots saved at 300 DPI for publication quality")


if __name__ == "__main__":
    main()
