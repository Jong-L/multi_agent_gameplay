"""
Valid Mask Analysis Script
Analyzes the relationship between using valid mask and player scores
Two types: no_valid_mask, use_valid_mask
Each type has 5 data files

Generates:
1. Player average TOTAL score vs episode and valid mask (4 subplots, 2 curves each)
2. Total average TOTAL score vs episode and valid mask (1 plot, 2 curves)
3. Player average BALL score vs episode and valid mask (4 subplots, 2 curves each)
4. Total average BALL score vs episode and valid mask (1 plot, 2 curves)

Valid mask indicates whether the observation space uses a validity mask for missing data
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
MASK_COLORS = {
    'no_valid_mask': '#E74C3C',    # Red - no valid mask
    'use_valid_mask': '#3498DB'     # Blue - use valid mask
}

MASK_NAMES = {
    'no_valid_mask': 'No Valid Mask',
    'use_valid_mask': 'Use Valid Mask'
}

# Ball collection sources
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_data_files(base_dir):
    """
    Load all data files organized by mask type
    Returns: dict {mask_type: [list of file paths]}
    """
    data_files = {
        'no_valid_mask': [],
        'use_valid_mask': []
    }
    
    base_path = Path(base_dir)
    
    # Find files by pattern
    for file in base_path.glob("*.csv"):
        fname = file.name
        if "valid_mask_analysis.py" in fname:
            continue
        if fname.startswith("no_valid_mask"):
            data_files['no_valid_mask'].append(str(file))
        elif fname.startswith("use_valid_mask"):
            data_files['use_valid_mask'].append(str(file))
    
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


def aggregate_data_by_mask(data_files):
    """
    Aggregate data for each mask type across all files
    Returns: dict {mask_type: aggregated_data}
    """
    aggregated = {}
    
    for mask_type, file_list in data_files.items():
        print(f"\nProcessing {MASK_NAMES[mask_type]} ({len(file_list)} files)...")
        
        if len(file_list) == 0:
            print(f"  [!] Warning: No files found for {mask_type}")
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
        aggregated[mask_type] = {
            'df': combined_df,
            'max_episodes': max_episodes,
            'num_files': len(file_list)
        }
    
    return aggregated


def compute_episode_stats(aggregated_data):
    """
    Compute per-episode statistics for each mask type
    Returns: dict with player total scores, ball scores, etc.
    """
    stats = {}
    
    for mask_type, data in aggregated_data.items():
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
            
            # Compute scores per player
            player_total_scores = {}
            player_ball_scores = {}
            
            for player_id in all_players:
                player_df = ep_df[ep_df['player_id'] == player_id]
                
                # Total score (sum of all values)
                total_score = player_df['value'].sum()
                player_total_scores[player_id] = total_score / num_files
                
                # Ball collection score (only BALL_SOURCES)
                ball_df = player_df[player_df['source'].isin(BALL_SOURCES)]
                ball_score = ball_df['value'].sum()
                player_ball_scores[player_id] = ball_score / num_files
                
                ep_data[f'player_{player_id}_total_score'] = player_total_scores[player_id]
                ep_data[f'player_{player_id}_ball_score'] = player_ball_scores[player_id]
            
            # Total across all players
            total_total_score = sum(player_total_scores.values())
            total_ball_score = sum(player_ball_scores.values())
            
            ep_data['total_total_score'] = total_total_score
            ep_data['total_ball_score'] = total_ball_score
            
            episode_stats.append(ep_data)
        
        stats_df = pd.DataFrame(episode_stats)
        stats[mask_type] = {
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


def plot_player_total_score_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 1: Each player's average TOTAL score vs episode and valid mask
    4 subplots (one per player), each with 2 curves (2 mask types)
    """
    # Get player list from first mask type
    first_mask = list(stats.keys())[0]
    players = stats[first_mask]['players']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Player Average Total Score vs Episode and Valid Mask', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_total_score'
        
        # Plot each mask type
        for mask_type in ['no_valid_mask', 'use_valid_mask']:
            if mask_type not in stats:
                continue
            
            df = stats[mask_type]['df']
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
                        label=MASK_NAMES[mask_type],
                        color=MASK_COLORS[mask_type],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Total Score', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_total_total_score_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 2: All players' average TOTAL score vs episode and valid mask
    1 plot with 2 curves (2 mask types)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each mask type
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type not in stats:
            continue
        
        df = stats[mask_type]['df']
        episodes = df['episode_id'].values
        total_scores = df['total_total_score'].values
        
        # Apply smoothing
        if smooth_window > 1 and len(total_scores) > smooth_window:
            smoothed_scores = smooth_curve(total_scores, smooth_window)
            smoothed_episodes = episodes[smooth_window-1:]
        else:
            smoothed_scores = total_scores
            smoothed_episodes = episodes
        
        sns.lineplot(x=smoothed_episodes, y=smoothed_scores,
                    label=MASK_NAMES[mask_type],
                    color=MASK_COLORS[mask_type],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Score', fontsize=12)
    ax.set_title('Total Average Score vs Episode and Valid Mask', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Valid Mask')
    ax.set_xlim(0, None)
    
    # Add statistics annotation
    stats_text = "Mean Total Score:\n"
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type in stats:
            avg_score = stats[mask_type]['df']['total_total_score'].mean()
            stats_text += f"{MASK_NAMES[mask_type]}: {avg_score:.1f}\n"
    
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


def plot_player_ball_score_comparison(stats, save_path=None, smooth_window=5):
    """
    Plot 3: Each player's average BALL score vs episode and valid mask
    4 subplots (one per player), each with 2 curves (2 mask types)
    """
    # Get player list from first mask type
    first_mask = list(stats.keys())[0]
    players = stats[first_mask]['players']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Player Average Ball Collection Score vs Episode and Valid Mask', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(players):
        ax = axes[idx]
        col_name = f'player_{player_id}_ball_score'
        
        # Plot each mask type
        for mask_type in ['no_valid_mask', 'use_valid_mask']:
            if mask_type not in stats:
                continue
            
            df = stats[mask_type]['df']
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
                        label=MASK_NAMES[mask_type],
                        color=MASK_COLORS[mask_type],
                        linewidth=2,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Score', fontsize=11)
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
    Plot 4: All players' average BALL score vs episode and valid mask
    1 plot with 2 curves (2 mask types)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Plot each mask type
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type not in stats:
            continue
        
        df = stats[mask_type]['df']
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
                    label=MASK_NAMES[mask_type],
                    color=MASK_COLORS[mask_type],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Score', fontsize=12)
    ax.set_title('Total Ball Collection Score vs Episode and Valid Mask', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Valid Mask')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Add statistics annotation
    stats_text = "Mean Ball Collection Score:\n"
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type in stats:
            avg_score = stats[mask_type]['df']['total_ball_score'].mean()
            stats_text += f"{MASK_NAMES[mask_type]}: {avg_score:.1f}\n"
    
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


def save_summary_csv(stats, output_dir):
    """Save summary statistics to CSV files"""
    summary_dir = Path(output_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    for mask_type, data in stats.items():
        df = data['df']
        output_path = summary_dir / f"{mask_type}_episode_stats.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    # Create overall summary
    summary_data = []
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type not in stats:
            continue
        
        df = stats[mask_type]['df']
        row = {
            'mask_type': mask_type,
            'mask_name': MASK_NAMES[mask_type],
            'mean_total_total_score': df['total_total_score'].mean(),
            'mean_total_ball_score': df['total_ball_score'].mean(),
            'max_episodes': stats[mask_type]['max_episodes']
        }
        
        # Add per-player stats
        players = stats[mask_type]['players']
        for player_id in players:
            row[f'player_{player_id}_mean_total_score'] = df[f'player_{player_id}_total_score'].mean()
            row[f'player_{player_id}_mean_ball_score'] = df[f'player_{player_id}_ball_score'].mean()
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = summary_dir / "overall_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    return summary_df


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("VALID MASK ANALYSIS SUMMARY")
    print("="*70)
    
    for mask_type in ['no_valid_mask', 'use_valid_mask']:
        if mask_type not in stats:
            continue
        
        data = stats[mask_type]
        df = data['df']
        players = data['players']
        
        print(f"\n{MASK_NAMES[mask_type]}:")
        print(f"  Episodes: {data['max_episodes']}")
        print(f"  Mean Total Score: {df['total_total_score'].mean():.2f}")
        print(f"  Mean Ball Collection Score: {df['total_ball_score'].mean():.2f}")
        
        print("  Per-player stats:")
        for player_id in players:
            total_col = f'player_{player_id}_total_score'
            ball_col = f'player_{player_id}_ball_score'
            print(f"    Player {player_id}: Total={df[total_col].mean():.2f}, Ball={df[ball_col].mean():.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\valid_mask_comparison"
    summary_dir = Path(base_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Valid Mask Analysis (Seaborn Style)")
    print("="*70)
    print(f"Ball sources counted: {BALL_SOURCES}")
    
    # Step 1: Load data file paths
    print("\n[1/6] Loading data file paths...")
    data_files = load_data_files(base_dir)
    for mt, files in data_files.items():
        print(f"  {MASK_NAMES[mt]}: {len(files)} files")
    
    # Step 2: Aggregate data
    print("\n[2/6] Aggregating data...")
    aggregated = aggregate_data_by_mask(data_files)
    
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
    
    # Plot 1: Player total score comparison
    print("  Creating player total score comparison...")
    fig1 = plot_player_total_score_comparison(stats, 
                                        save_path=summary_dir / "player_total_score_comparison.png",
                                        smooth_window=5)
    
    # Plot 2: Total total score comparison
    print("  Creating total total score comparison...")
    fig2 = plot_total_total_score_comparison(stats, 
                                       save_path=summary_dir / "total_total_score_comparison.png",
                                       smooth_window=5)
    
    # Plot 3: Player ball score comparison
    print("  Creating player ball score comparison...")
    fig3 = plot_player_ball_score_comparison(stats, 
                                             save_path=summary_dir / "player_ball_score_comparison.png",
                                             smooth_window=5)
    
    # Plot 4: Total ball score comparison
    print("  Creating total ball score comparison...")
    fig4 = plot_total_ball_score_comparison(stats, 
                                            save_path=summary_dir / "total_ball_score_comparison.png",
                                            smooth_window=5)
    
    # Step 6: Print summary
    print("\n[6/6] Printing summary...")
    print_summary(stats)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nGenerated files in {summary_dir}:")
    print("  - player_total_score_comparison.png")
    print("  - total_total_score_comparison.png")
    print("  - player_ball_score_comparison.png")
    print("  - total_ball_score_comparison.png")
    print("  - *_episode_stats.csv (per-mask episode data)")
    print("  - overall_summary.csv")
    print("\nAll plots saved at 300 DPI for publication quality")


if __name__ == "__main__":
    main()
