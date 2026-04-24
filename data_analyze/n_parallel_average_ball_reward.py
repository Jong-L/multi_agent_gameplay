"""
Average Ball Reward Analysis Script
Function: Merge multiple parallel test data, calculate average BALL rewards per player and total
Only counts collect_ball_A and collect_ball_B sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ball collection sources
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_and_aggregate_ball_data(file_paths):
    """
    Load multiple CSV files, aggregate BALL scores by episode_id and player_id
    Only includes collect_ball_A and collect_ball_B sources
    Returns: dict of aggregated data per file {file_name: DataFrame(episode_id, player_id, total_ball_reward)}
    """
    file_data = {}
    
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"Error: File not found - {fp}")
            sys.exit(1)
        
        # Read CSV
        df = pd.read_csv(fp)
        
        # Filter only ball collection sources
        ball_df = df[df['source'].isin(BALL_SOURCES)]
        
        # Aggregate value by episode_id and player_id
        aggregated = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        aggregated.columns = ['episode_id', 'player_id', 'total_ball_reward']
        
        file_data[path.name] = aggregated
        print(f"Loaded: {path.name} - {len(df)} raw records -> {len(ball_df)} ball records -> {len(aggregated)} aggregated records")
    
    return file_data


def check_episode_consistency(file_data):
    """
    Check if all files have the same number of episodes
    Returns: max_episodes if consistent, None otherwise
    """
    episode_counts = {}
    for fname, df in file_data.items():
        max_ep = df['episode_id'].max() if len(df) > 0 else 0
        episode_counts[fname] = max_ep
    
    unique_counts = set(episode_counts.values())
    
    print("\n=== Episode Count Check ===")
    for fname, count in episode_counts.items():
        print(f"  {fname}: {count} episodes")
    
    if len(unique_counts) > 1:
        print(f"\n[!] Warning: Episode counts are inconsistent across files!")
        print(f"   Min episodes: {min(unique_counts)}")
        print(f"   Max episodes: {max(unique_counts)}")
        return None, episode_counts
    else:
        max_episodes = list(unique_counts)[0]
        print(f"\n[OK] All files have consistent episode count: {max_episodes} episodes")
        return max_episodes, episode_counts


def compute_average_ball_rewards(file_data, max_episodes):
    """
    Calculate average ball reward per player per episode across all files
    Rule: If a player doesn't appear in a file for an episode, their score is 0 for that file
    """
    num_files = len(file_data)
    
    # Collect all player IDs
    all_players = set()
    for df in file_data.values():
        all_players.update(df['player_id'].unique())
    all_players = sorted(list(all_players))
    
    print(f"\n=== Player Statistics ===")
    print(f"Players found: {all_players}")
    print(f"Number of data files: {num_files}")
    
    # Build result data structure
    results = []
    
    for episode in range(1, max_episodes + 1):
        episode_data = {'episode_id': episode}
        player_totals = {}
        
        # Collect ball scores for all players in this episode
        for player_id in all_players:
            player_sum = 0
            
            for fname, df in file_data.items():
                # Find player's ball score in this episode
                mask = (df['episode_id'] == episode) & (df['player_id'] == player_id)
                player_records = df[mask]
                
                if len(player_records) > 0:
                    player_sum += player_records['total_ball_reward'].values[0]
                # If not present, add 0 (default behavior)
            
            player_avg = player_sum / num_files
            player_totals[player_id] = player_avg
            episode_data[f'player_{player_id}_ball_avg'] = player_avg
        
        # Calculate total average across all players
        total_avg = sum(player_totals.values())
        episode_data['total_ball_avg_reward'] = total_avg
        
        results.append(episode_data)
    
    result_df = pd.DataFrame(results)
    return result_df, all_players


def main():
    # Define file paths (linear reward function - lrrs)
    file_paths = [
        r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log\rewards_lrrs_2026-04-24_11-20-48_pid20624.csv",
        r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log\rewards_lrrs_2026-04-24_11-20-50_pid26140.csv",
        r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log\rewards_lrrs_2026-04-24_11-20-51_pid7384.csv",
        r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log\rewards_lrrs_2026-04-24_11-20-52_pid24428.csv",
    ]
    
    output_path = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log\lr_average_ball_reward.csv"
    
    print("=" * 60)
    print("LR Average Ball Reward Analysis")
    print("=" * 60)
    print(f"Ball sources: {BALL_SOURCES}")
    
    # 1. Load and aggregate ball data
    print("\n[1/4] Loading data files...")
    file_data = load_and_aggregate_ball_data(file_paths)
    
    # 2. Check episode consistency
    print("\n[2/4] Checking data consistency...")
    max_episodes, episode_counts = check_episode_consistency(file_data)
    
    if max_episodes is None:
        print("\n[ERROR] Cannot proceed due to inconsistent episode counts.")
        print("   Please ensure all tests ran for the same number of episodes.")
        sys.exit(1)
    
    # 3. Compute average ball rewards
    print("\n[3/4] Computing average ball rewards...")
    result_df, all_players = compute_average_ball_rewards(file_data, max_episodes)
    
    # 4. Save results
    print("\n[4/4] Saving results...")
    result_df.to_csv(output_path, index=False)
    print(f"[OK] Results saved to: {output_path}")
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    
    print(f"\nTotal episodes: {max_episodes}")
    print(f"Number of players: {len(all_players)} (IDs: {all_players})")
    print(f"Ball sources counted: {BALL_SOURCES}")
    
    print("\n--- Per-Player Average Ball Reward Statistics ---")
    for player_id in all_players:
        col = f'player_{player_id}_ball_avg'
        avg_reward = result_df[col].mean()
        total_reward = result_df[col].sum()
        print(f"  Player {player_id}: avg per episode={avg_reward:.4f}, total={total_reward:.4f}")
    
    print("\n--- Total Average Ball Reward Statistics ---")
    total_avg_mean = result_df['total_ball_avg_reward'].mean()
    total_avg_sum = result_df['total_ball_avg_reward'].sum()
    print(f"  Avg per episode={total_avg_mean:.4f}, total across all episodes={total_avg_sum:.4f}")
    
    # Show preview of first 5 and last 5 episodes
    print("\n--- Data Preview (First 5 episodes) ---")
    preview_cols = ['episode_id'] + [f'player_{p}_ball_avg' for p in all_players] + ['total_ball_avg_reward']
    print(result_df[preview_cols].head(5).to_string(index=False))
    
    print("\n--- Data Preview (Last 5 episodes) ---")
    print(result_df[preview_cols].tail(5).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
