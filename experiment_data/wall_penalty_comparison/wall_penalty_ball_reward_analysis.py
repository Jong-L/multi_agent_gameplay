"""
Wall Penalty vs Ball Reward Analysis Script
Analyzes the relationship between wall collision penalty values and BALL COLLECTION rewards
Three penalty types: 00p5 (0.05), 0p5 (0.5), 5 (5.0)
Each type has 5 data files

Generates:
1. Player average BALL score vs episode and penalty (4 subplots)
2. Total average BALL score vs episode and penalty (1 plot)
3. Player average wall collision count vs episode and penalty (4 subplots)
4. Total average wall collision count vs penalty (1 plot)

Key difference from wall_penalty_analysis.py:
- This script ONLY counts collect_ball_A and collect_ball_B as score
- Wall collision penalties are tracked separately as counts, not subtracted from score
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

# Color palette for penalty values (colorblind-friendly)
PENALTY_COLORS = {
    '00p5': '#0173B2',    # Blue - 0.05
    '0p5': '#DE8F05',     # Orange - 0.5
    '5': '#029E73'        # Green - 5.0
}

PENALTY_NAMES = {
    '00p5': 'Penalty=0.05',
    '0p5': 'Penalty=0.5',
    '5': 'Penalty=5.0'
}

PENALTY_VALUES = {
    '00p5': 0.05,
    '0p5': 0.5,
    '5': 5.0
}

# Ball collection sources only
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_data_files(base_dir):
    """
    Load all data files organized by penalty type
    Returns: dict {penalty_type: [list of file paths]}
    """
    data_files = {
        '00p5': [],
        '0p5': [],
        '5': []
    }
    
    base_path = Path(base_dir)
    
    # Find files by pattern
    for file in base_path.glob("*.csv"):
        fname = file.name
        if "wall_penalty_analysis.py" in fname:
            continue
        if "00p5penalty" in fname:
            data_files['00p5'].append(str(file))
        elif "0p5penalty" in fname:
            data_files['0p5'].append(str(file))
        elif "5penalty" in fname and "00p5" not in fname and "0p5" not in fname:
            data_files['5'].append(str(file))
    
    # Sort files for consistency
    for key in data_files:
        data_files[key].sort()
    
    return data_files


def load_and_process_file(file_path):
    """
    Load a single CSV file and return processed dataframe
    Columns: episode_id, player_id, source, value, game_time
    """
    df = pd.read_csv(file_path)
    return df


def aggregate_data_by_penalty(data_files):
    """
    Aggregate data for each penalty type across all files
    Returns: dict {penalty_type: aggregated_data}
    """
    aggregated = {}
    
    for penalty_type, file_list in data_files.items():
        print(f"\nProcessing {PENALTY_NAMES[penalty_type]} ({len(file_list)} files)...")
        
        if len(file_list) == 0:
            print(f"  [!] Warning: No files found for {penalty_type}")
            continue
        
        all_episodes = []
        episode_counts = {}
        
        for fp in file_list:
            df = load_and_process_file(fp)
            max_ep = df['episode_id'].max() if len(df) > 0 else 0
            episode_counts[Path(fp).name] = max_ep
            all_episodes.append(df)
        
        # Check episode consistency
        unique_counts = set(episode_counts.values())
        if len(unique_counts) > 1:
            print(f"  [!] Warning: Inconsistent episode counts: {episode_counts}")
            print("  [!] Analysis aborted - all files must have the same number of episodes")
            return None
        
        max_episodes = list(unique_counts)[0]
        print(f"  [OK] Consistent: {max_episodes} episodes")
        
        # Combine all data
        combined_df = pd.concat(all_episodes, ignore_index=True)
        aggregated[penalty_type] = {
            'df': combined_df,
            'max_episodes': max_episodes,
            'num_files': len(file_list)
        }
    
    return aggregated


def compute_episode_stats(aggregated_data):
    """
    Compute per-episode statistics for each penalty type
    Returns: dict with player ball scores, total ball scores, wall counts per penalty type
    """
    stats = {}
    
    for penalty_type, data in aggregated_data.items():
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
            
            # Compute ball score per player (only collect_ball_A and collect_ball_B)
            player_ball_scores = {}
            player_wall_counts = {}
            
            for player_id in all_players:
                player_df = ep_df[ep_df['player_id'] == player_id]
                
                # Ball collection score (only BALL_SOURCES)
                ball_df = player_df[player_df['source'].isin(BALL_SOURCES)]
                ball_score = ball_df['value'].sum()
                player_ball_scores[player_id] = ball_score / num_files  # Average across files
                
                # Wall collision count
                wall_df = player_df[player_df['source'] == 'wall_collision']
                wall_count = len(wall_df)
                player_wall_counts[player_id] = wall_count / num_files  # Average across files
                
                ep_data[f'player_{player_id}_ball_score'] = player_ball_scores[player_id]
                ep_data[f'player_{player_id}_wall_count'] = player_wall_counts[player_id]
            
            # Total across all players
            total_ball_score = sum(player_ball_scores.values())
            total_wall_count = sum(player_wall_counts.values())
            
            ep_data['total_ball_score'] = total_ball_score
            ep_data['total_wall_count'] = total_wall_count
            
            episode_stats.append(ep_data)
        
        stats_df = pd.DataFrame(episode_stats)
        stats[penalty_type] = {
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


def plot_player_ball_score_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 1: Each player's average BALL score vs episode and penalty
    4 subplots (one per player), each with 3 curves (3 penalty values)
    """
    # Get player list from first penalty type
    first_penalty = list(stats.keys())[0]
    players = stats[first_penalty]['players']
    num_players = len(players)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Player Average Ball Collection Score vs Episode and Wall Penalty', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_ball_score'
        
        # Plot each penalty type
        for penalty_type in ['00p5', '0p5', '5']:
            if penalty_type not in stats:
                continue
            
            df = stats[penalty_type]['df']
            episodes = df['episode_id'].values
            scores = df[col_name].values
            
            # Apply smoothing
            if smooth_window > 1 and len(scores) > smooth_window:
                smoothed_scores = smooth_curve(scores, smooth_window)
                smoothed_episodes = episodes[smooth_window-1:]
            else:
                smoothed_scores = scores
                smoothed_episodes = episodes
            
            sns.lineplot(x=smoothed_episodes, y=smoothed_scores,
                        label=PENALTY_NAMES[penalty_type],
                        color=PENALTY_COLORS[penalty_type],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Collection Score', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_ball_score_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 2: All players' average BALL score vs episode and penalty
    1 plot with 3 curves (3 penalty values)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each penalty type
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        
        df = stats[penalty_type]['df']
        episodes = df['episode_id'].values
        total_scores = df['total_ball_score'].values
        
        # Apply smoothing
        if smooth_window > 1 and len(total_scores) > smooth_window:
            smoothed_scores = smooth_curve(total_scores, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_scores = total_scores
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_scores,
                    label=PENALTY_NAMES[penalty_type],
                    color=PENALTY_COLORS[penalty_type],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Collection Score', fontsize=12)
    ax.set_title('Total Ball Collection Score vs Episode and Wall Penalty', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Wall Penalty')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics annotation (positioned to avoid legend overlap)
    stats_text = "Mean Ball Collection Score:\n"
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type in stats:
            avg_score = stats[penalty_type]['df']['total_ball_score'].mean()
            stats_text += f"{PENALTY_NAMES[penalty_type]}: {avg_score:.1f}\n"
    
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


def plot_player_wall_count_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 3: Each player's average wall collision count vs episode and penalty
    4 subplots (one per player), each with 3 curves (3 penalty values)
    """
    # Get player list from first penalty type
    first_penalty = list(stats.keys())[0]
    players = stats[first_penalty]['players']
    num_players = len(players)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Player Average Wall Collision Count vs Episode and Wall Penalty', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_wall_count'
        
        # Plot each penalty type
        for penalty_type in ['00p5', '0p5', '5']:
            if penalty_type not in stats:
                continue
            
            df = stats[penalty_type]['df']
            episodes = df['episode_id'].values
            wall_counts = df[col_name].values
            
            # Apply smoothing
            if smooth_window > 1 and len(wall_counts) > smooth_window:
                smoothed_counts = smooth_curve(wall_counts, smooth_window)
                smoothed_episodes = episodes[smooth_window-1:]
            else:
                smoothed_counts = wall_counts
                smoothed_episodes = episodes
            
            sns.lineplot(x=smoothed_episodes, y=smoothed_counts,
                        label=PENALTY_NAMES[penalty_type],
                        color=PENALTY_COLORS[penalty_type],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Wall Collision Count', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_wall_count_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 4: All players' average wall collision count vs episode and penalty
    1 plot with 3 curves (3 penalty values)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each penalty type
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        
        df = stats[penalty_type]['df']
        episodes = df['episode_id'].values
        total_wall_counts = df['total_wall_count'].values
        
        # Apply smoothing
        if smooth_window > 1 and len(total_wall_counts) > smooth_window:
            smoothed_counts = smooth_curve(total_wall_counts, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_counts = total_wall_counts
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_counts,
                    label=PENALTY_NAMES[penalty_type],
                    color=PENALTY_COLORS[penalty_type],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Wall Collision Count', fontsize=12)
    ax.set_title('Total Wall Collision Count vs Episode and Penalty', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, frameon=True, title='Wall Penalty')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics annotation (positioned to avoid legend overlap)
    stats_text = "Mean Wall Collisions:\n"
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type in stats:
            avg_count = stats[penalty_type]['df']['total_wall_count'].mean()
            stats_text += f"{PENALTY_NAMES[penalty_type]}: {avg_count:.1f}\n"
    
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes,
           verticalalignment='top',
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
    
    for penalty_type, data in stats.items():
        df = data['df']
        output_path = summary_dir / f"{penalty_type}_ball_episode_stats.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    # Create overall summary
    summary_data = []
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        
        df = stats[penalty_type]['df']
        row = {
            'penalty_type': penalty_type,
            'penalty_value': PENALTY_VALUES[penalty_type],
            'mean_total_ball_score': df['total_ball_score'].mean(),
            'mean_total_wall_count': df['total_wall_count'].mean(),
            'max_episodes': stats[penalty_type]['max_episodes']
        }
        
        # Add per-player stats
        players = stats[penalty_type]['players']
        for player_id in players:
            row[f'player_{player_id}_mean_ball_score'] = df[f'player_{player_id}_ball_score'].mean()
            row[f'player_{player_id}_mean_wall_count'] = df[f'player_{player_id}_wall_count'].mean()
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = summary_dir / "ball_overall_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    return summary_df


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("BALL COLLECTION SCORE SUMMARY")
    print("="*70)
    
    for penalty_type in ['00p5', '0p5', '5']:
        if penalty_type not in stats:
            continue
        
        data = stats[penalty_type]
        df = data['df']
        players = data['players']
        
        print(f"\n{PENALTY_NAMES[penalty_type]}:")
        print(f"  Episodes: {data['max_episodes']}")
        print(f"  Mean Total Ball Score: {df['total_ball_score'].mean():.2f}")
        print(f"  Mean Total Wall Collisions: {df['total_wall_count'].mean():.2f}")
        
        print("  Per-player stats:")
        for player_id in players:
            score_col = f'player_{player_id}_ball_score'
            wall_col = f'player_{player_id}_wall_count'
            print(f"    Player {player_id}: Ball Score={df[score_col].mean():.2f}, Walls={df[wall_col].mean():.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\wall_penalty_comparison"
    summary_dir = Path(base_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Wall Penalty vs Ball Collection Score Analysis (Seaborn Style)")
    print("="*70)
    print(f"Ball sources counted: {BALL_SOURCES}")
    
    # Step 1: Load data file paths
    print("\n[1/6] Loading data file paths...")
    data_files = load_data_files(base_dir)
    for pt, files in data_files.items():
        print(f"  {PENALTY_NAMES[pt]}: {len(files)} files")
    
    # Step 2: Aggregate data
    print("\n[2/6] Aggregating data...")
    aggregated = aggregate_data_by_penalty(data_files)
    
    if aggregated is None:
        print("\n[ERROR] Episode count inconsistency detected. Analysis aborted.")
        return
    
    if len(aggregated) == 0:
        print("\n[ERROR] No data loaded. Please check file paths.")
        return
    
    # Step 3: Compute episode statistics
    print("\n[3/6] Computing episode statistics...")
    stats = compute_episode_stats(aggregated)
    
    # Step 4: Save summary CSV
    print("\n[4/6] Saving summary CSV files...")
    summary_df = save_summary_csv(stats, base_dir)
    
    # Step 5: Generate plots
    print("\n[5/6] Generating plots with Seaborn...")
    
    # Plot 1: Player ball score comparison
    print("  Creating player ball score comparison...")
    fig1 = plot_player_ball_score_comparison(stats, 
                                        save_path=summary_dir / "player_ball_score_comparison.png",
                                        smooth_window=5)
    
    # Plot 2: Total ball score comparison
    print("  Creating total ball score comparison...")
    fig2 = plot_total_ball_score_comparison(stats, 
                                       save_path=summary_dir / "total_ball_score_comparison.png",
                                       smooth_window=5)
    
    # Plot 3: Player wall count comparison
    print("  Creating player wall count comparison...")
    fig3 = plot_player_wall_count_comparison(stats, 
                                             save_path=summary_dir / "player_wall_count_comparison.png",
                                             smooth_window=5)
    
    # Plot 4: Total wall count comparison
    print("  Creating total wall count comparison...")
    fig4 = plot_total_wall_count_comparison(stats, 
                                            save_path=summary_dir / "total_wall_count_comparison.png",
                                            smooth_window=5)
    
    # Step 6: Print summary
    print("\n[6/6] Printing summary...")
    print_summary(stats)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nGenerated files in {summary_dir}:")
    print("  - player_ball_score_comparison.png")
    print("  - total_ball_score_comparison.png")
    print("  - player_wall_count_comparison.png")
    print("  - total_wall_count_comparison.png")
    print("  - *_ball_episode_stats.csv (per-penalty episode data)")
    print("  - ball_overall_summary.csv")
    print("\nAll plots saved at 300 DPI for publication quality")


if __name__ == "__main__":
    main()
