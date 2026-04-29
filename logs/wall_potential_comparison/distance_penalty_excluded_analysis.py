"""
Distance Penalty Excluded Analysis
Analyzes the Distance Penalty scheme with wall_distance excluded from total score.
This reveals the true performance without the continuous distance penalty component.

Only processes distance_penalty files (5 files).
Note: wall_distance records are excluded from score calculation.

Generates:
1. Player average score (excluding wall_distance) vs episode
2. Total average score (excluding wall_distance) vs episode
3. Comparison with original scores (including wall_distance)
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

# Color palette
SCHEME_NAME = 'Distance Penalty - Excluded'
SCHEME_COLOR = '#D55E00'  # Red


def load_data_files(base_dir):
    """Load only distance_penalty files"""
    base_path = Path(base_dir)
    file_list = []
    
    for file in base_path.glob("*.csv"):
        fname = file.name
        if "distance_penalty_" in fname:
            file_list.append(str(file))
    
    file_list.sort()
    return file_list


def load_and_process_file(file_path):
    """Load CSV file"""
    df = pd.read_csv(file_path)
    return df


def aggregate_data(file_list):
    """Aggregate data across all files"""
    print(f"\nProcessing {SCHEME_NAME} ({len(file_list)} files)...")
    
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
    
    return {
        'df': combined_df,
        'max_episodes': max_episodes,
        'num_files': len(file_list)
    }


def compute_episode_stats(data):
    """
    Compute per-episode statistics
    - excluded_score: total score excluding wall_distance
    - original_score: total score including wall_distance
    """
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
        
        player_excluded_scores = {}
        player_original_scores = {}
        player_wall_counts = {}
        
        for player_id in all_players:
            player_df = ep_df[ep_df['player_id'] == player_id]
            
            # Original score (all values including wall_distance)
            original_score = player_df['value'].sum()
            player_original_scores[player_id] = original_score / num_files
            
            # Excluded score (excluding wall_distance)
            excluded_df = player_df[player_df['source'] != 'wall_distance']
            excluded_score = excluded_df['value'].sum()
            player_excluded_scores[player_id] = excluded_score / num_files
            
            # Wall collision count (only wall_collision source)
            wall_df = player_df[player_df['source'] == 'wall_collision']
            wall_count = len(wall_df)
            player_wall_counts[player_id] = wall_count / num_files
            
            ep_data[f'player_{player_id}_original_score'] = player_original_scores[player_id]
            ep_data[f'player_{player_id}_excluded_score'] = player_excluded_scores[player_id]
            ep_data[f'player_{player_id}_wall_count'] = player_wall_counts[player_id]
        
        # Totals
        total_original = sum(player_original_scores.values())
        total_excluded = sum(player_excluded_scores[player_id] for player_id in all_players)
        total_wall_count = sum(player_wall_counts.values())
        
        ep_data['total_original_score'] = total_original
        ep_data['total_excluded_score'] = total_excluded
        ep_data['total_wall_count'] = total_wall_count
        
        episode_stats.append(ep_data)
    
    stats_df = pd.DataFrame(episode_stats)
    return {
        'df': stats_df,
        'players': all_players,
        'max_episodes': max_episodes
    }


def smooth_curve(y, window=5):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_player_comparison(stats, save_path=None, smooth_window=10):
    """
    Plot: Each player's average score vs episode
    4 subplots, each with 2 curves (original vs excluded)
    """
    players = stats['players']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(f'Player Average Score vs Episode ({SCHEME_NAME})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        
        df = stats['df']
        episodes = df['episode_id'].values
        
        # Original score (with wall_distance penalty)
        original_scores = df[f'player_{player_id}_original_score'].values
        if smooth_window > 1 and len(original_scores) > smooth_window:
            smoothed = smooth_curve(original_scores, smooth_window)
            smoothed_ep = episodes[smooth_window-1:]
        else:
            smoothed = original_scores
            smoothed_ep = episodes
        
        sns.lineplot(x=smoothed_ep, y=smoothed,
                    label='With Distance Penalty',
                    color='#8B4513',  # Brown (darker red)
                    linewidth=2,
                    linestyle='--',
                    ax=ax)
        
        # Excluded score (without wall_distance)
        excluded_scores = df[f'player_{player_id}_excluded_score'].values
        if smooth_window > 1 and len(excluded_scores) > smooth_window:
            smoothed = smooth_curve(excluded_scores, smooth_window)
            smoothed_ep = episodes[smooth_window-1:]
        else:
            smoothed = excluded_scores
            smoothed_ep = episodes
        
        sns.lineplot(x=smoothed_ep, y=smoothed,
                    label='Excluded Distance Penalty',
                    color=SCHEME_COLOR,
                    linewidth=2.5,
                    ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Score', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_comparison(stats, save_path=None, smooth_window=10):
    """
    Plot: Total average score vs episode
    1 plot with 2 curves (original vs excluded)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    df = stats['df']
    episodes = df['episode_id'].values
    
    # Original score (with wall_distance penalty)
    original_scores = df['total_original_score'].values
    if smooth_window > 1 and len(original_scores) > smooth_window:
        smoothed = smooth_curve(original_scores, smooth_window)
        smoothed_ep = episodes[smooth_window-1:]
    else:
        smoothed = original_scores
        smoothed_ep = episodes
    
    sns.lineplot(x=smoothed_ep, y=smoothed,
                label='With Distance Penalty',
                color='#8B4513',
                linewidth=2.5,
                linestyle='--',
                ax=ax)
    
    # Excluded score (without wall_distance)
    excluded_scores = df['total_excluded_score'].values
    if smooth_window > 1 and len(excluded_scores) > smooth_window:
        smoothed = smooth_curve(excluded_scores, smooth_window)
        smoothed_ep = episodes[smooth_window-1:]
    else:
        smoothed = excluded_scores
        smoothed_ep = episodes
    
    sns.lineplot(x=smoothed_ep, y=smoothed,
                label='Excluded Distance Penalty',
                color=SCHEME_COLOR,
                linewidth=2.5,
                ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Score', fontsize=12)
    ax.set_title(f'Total Average Score vs Episode ({SCHEME_NAME})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.set_xlim(0, None)
    
    # Add statistics annotation
    orig_mean = df['total_original_score'].mean()
    excl_mean = df['total_excluded_score'].mean()
    diff = orig_mean - excl_mean
    
    stats_text = "Mean Total Score:\n"
    stats_text += f"With Distance Penalty: {orig_mean:.1f}\n"
    stats_text += f"Excluded: {excl_mean:.1f}\n"
    stats_text += f"Difference: {diff:.1f}\n"
    stats_text += f"Improvement: {(-diff/excl_mean*100):.1f}%"
    
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
    
    df = stats['df']
    
    # Save episode stats
    output_path = summary_dir / "distance_penalty_excluded_episode_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Create summary
    players = stats['players']
    orig_mean = df['total_original_score'].mean()
    excl_mean = df['total_excluded_score'].mean()
    
    summary_data = {
        'scheme': SCHEME_NAME,
        'max_episodes': stats['max_episodes'],
        'mean_total_original_score': orig_mean,
        'mean_total_excluded_score': excl_mean,
        'mean_total_wall_count': df['total_wall_count'].mean(),
        'score_difference': orig_mean - excl_mean,
        'improvement_percentage': (-(orig_mean - excl_mean) / excl_mean * 100) if excl_mean != 0 else 0
    }
    
    for player_id in players:
        summary_data[f'player_{player_id}_original'] = df[f'player_{player_id}_original_score'].mean()
        summary_data[f'player_{player_id}_excluded'] = df[f'player_{player_id}_excluded_score'].mean()
        summary_data[f'player_{player_id}_walls'] = df[f'player_{player_id}_wall_count'].mean()
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = summary_dir / "distance_penalty_excluded_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    return summary_df


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print(f"SUMMARY STATISTICS - {SCHEME_NAME}")
    print("="*70)
    
    df = stats['df']
    players = stats['players']
    
    orig_mean = df['total_original_score'].mean()
    excl_mean = df['total_excluded_score'].mean()
    diff = orig_mean - excl_mean
    
    print(f"\nEpisodes: {stats['max_episodes']}")
    print(f"Mean Total Score (With Distance Penalty): {orig_mean:.2f}")
    print(f"Mean Total Score (Excluded Distance Penalty): {excl_mean:.2f}")
    print(f"Score Difference: {diff:.2f}")
    print(f"Improvement: {(-diff/excl_mean*100):.1f}%")
    print(f"Mean Total Wall Collisions: {df['total_wall_count'].mean():.2f}")
    
    print("\nPer-player stats:")
    for player_id in players:
        orig = df[f'player_{player_id}_original_score'].mean()
        excl = df[f'player_{player_id}_excluded_score'].mean()
        walls = df[f'player_{player_id}_wall_count'].mean()
        print(f"  Player {player_id}: Original={orig:.2f}, Excluded={excl:.2f}, Walls={walls:.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\wall_potential_comparison"
    summary_dir = Path(base_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print(f"{SCHEME_NAME} Analysis")
    print("="*70)
    
    # Step 1: Load data file paths
    print("\n[1/4] Loading data file paths...")
    file_list = load_data_files(base_dir)
    print(f"  Found {len(file_list)} files")
    
    if len(file_list) == 0:
        print("[ERROR] No files found!")
        return
    
    # Step 2: Aggregate data
    print("\n[2/4] Aggregating data...")
    data = aggregate_data(file_list)
    
    if data is None:
        print("\n[ERROR] Episode count inconsistency detected.")
        return
    
    # Step 3: Compute episode statistics
    print("\n[3/4] Computing episode statistics...")
    stats = compute_episode_stats(data)
    
    # Step 4: Save summary CSV
    print("\n[4/4] Saving summary CSV files...")
    summary_df = save_summary_csv(stats, base_dir)
    
    # Generate plots
    print("\nGenerating plots with Seaborn...")
    
    print("  Creating player comparison...")
    fig1 = plot_player_comparison(stats, 
                                  save_path=summary_dir / "distance_penalty_excluded_player_comparison.png",
                                  smooth_window=10)
    
    print("  Creating total comparison...")
    fig2 = plot_total_comparison(stats, 
                                 save_path=summary_dir / "distance_penalty_excluded_total_comparison.png",
                                 smooth_window=10)
    
    # Print summary
    print_summary(stats)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nGenerated files in {summary_dir}:")
    print("  - distance_penalty_excluded_player_comparison.png")
    print("  - distance_penalty_excluded_total_comparison.png")
    print("  - distance_penalty_excluded_episode_stats.csv")
    print("  - distance_penalty_excluded_summary.csv")
    print("\nAll plots saved at 300 DPI for publication quality")


if __name__ == "__main__":
    main()