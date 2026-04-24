"""
Average Ball Reward Analysis Script for All Potential Functions
Function: Process linear (lrrs), inverse proportional (invprs), and exponential (exprs) reward data
Generate average ball rewards for each function type
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
        print(f"  Loaded: {path.name} - {len(df)} raw -> {len(ball_df)} ball -> {len(aggregated)} agg records")
    
    return file_data


def check_episode_consistency(file_data):
    """Check if all files have the same number of episodes"""
    episode_counts = {}
    for fname, df in file_data.items():
        max_ep = df['episode_id'].max() if len(df) > 0 else 0
        episode_counts[fname] = max_ep
    
    unique_counts = set(episode_counts.values())
    
    print("\n  Episode counts:")
    for fname, count in episode_counts.items():
        print(f"    {fname}: {count} episodes")
    
    if len(unique_counts) > 1:
        print(f"  [!] Warning: Inconsistent episode counts!")
        return None
    else:
        max_episodes = list(unique_counts)[0]
        print(f"  [OK] Consistent: {max_episodes} episodes")
        return max_episodes


def compute_average_ball_rewards(file_data, max_episodes):
    """Calculate average ball reward per player per episode across all files"""
    num_files = len(file_data)
    
    # Collect all player IDs
    all_players = set()
    for df in file_data.values():
        all_players.update(df['player_id'].unique())
    all_players = sorted(list(all_players))
    
    print(f"  Players: {all_players}, Files: {num_files}")
    
    # Build result data structure
    results = []
    
    for episode in range(1, max_episodes + 1):
        episode_data = {'episode_id': episode}
        player_totals = {}
        
        for player_id in all_players:
            player_sum = 0
            
            for fname, df in file_data.items():
                mask = (df['episode_id'] == episode) & (df['player_id'] == player_id)
                player_records = df[mask]
                
                if len(player_records) > 0:
                    player_sum += player_records['total_ball_reward'].values[0]
            
            player_avg = player_sum / num_files
            player_totals[player_id] = player_avg
            episode_data[f'player_{player_id}_ball_avg'] = player_avg
        
        total_avg = sum(player_totals.values())
        episode_data['total_ball_avg_reward'] = total_avg
        
        results.append(episode_data)
    
    result_df = pd.DataFrame(results)
    return result_df, all_players


def process_function_type(func_name, file_paths, output_dir):
    """Process one function type and save results"""
    print(f"\n{'='*60}")
    print(f"Processing {func_name.upper()}")
    print('='*60)
    
    # Load data
    print("[1/3] Loading data...")
    file_data = load_and_aggregate_ball_data(file_paths)
    
    # Check consistency
    print("[2/3] Checking consistency...")
    max_episodes = check_episode_consistency(file_data)
    
    if max_episodes is None:
        print("[ERROR] Skipping due to inconsistent episodes.")
        return None
    
    # Compute averages
    print("[3/3] Computing averages...")
    result_df, all_players = compute_average_ball_rewards(file_data, max_episodes)
    
    # Save results
    output_path = f"{output_dir}\\{func_name}_average_ball_reward.csv"
    result_df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")
    
    # Print summary
    print(f"\nSummary for {func_name}:")
    print(f"  Episodes: {max_episodes}, Players: {len(all_players)}")
    for player_id in all_players:
        col = f'player_{player_id}_ball_avg'
        avg = result_df[col].mean()
        print(f"  Player {player_id} avg: {avg:.2f}")
    print(f"  Total avg per episode: {result_df['total_ball_avg_reward'].mean():.2f}")
    
    return result_df


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    # Define file paths for each function type
    data = {
        'lr': [
            f"{base_dir}\\rewards_lrrs_2026-04-24_11-20-48_pid20624.csv",
            f"{base_dir}\\rewards_lrrs_2026-04-24_11-20-50_pid26140.csv",
            f"{base_dir}\\rewards_lrrs_2026-04-24_11-20-51_pid7384.csv",
            f"{base_dir}\\rewards_lrrs_2026-04-24_11-20-52_pid24428.csv",
        ],
        'invp': [
            f"{base_dir}\\rewards_invprs_2026-04-24_13-49-24_pid4740.csv",
            f"{base_dir}\\rewards_invprs_2026-04-24_13-49-25_pid4880.csv",
            f"{base_dir}\\rewards_invprs_2026-04-24_13-49-26_pid19964.csv",
            f"{base_dir}\\rewards_invprs_2026-04-24_13-49-28_pid22932.csv",
        ],
        'exp': [
            f"{base_dir}\\rewards_exprs_2026-04-24_13-28-26_pid24720.csv",
            f"{base_dir}\\rewards_exprs_2026-04-24_13-28-28_pid4208.csv",
            f"{base_dir}\\rewards_exprs_2026-04-24_13-28-29_pid9616.csv",
            f"{base_dir}\\rewards_exprs_2026-04-24_13-28-30_pid21048.csv",
        ],
    }
    
    print("="*60)
    print("Ball Reward Analysis - All Potential Functions")
    print("="*60)
    print(f"Ball sources: {BALL_SOURCES}")
    
    results = {}
    for func_name, file_paths in data.items():
        df = process_function_type(func_name, file_paths, base_dir)
        if df is not None:
            results[func_name] = df
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)
    print(f"\nGenerated files:")
    for func_name in results.keys():
        print(f"  - {func_name}_average_ball_reward.csv")


if __name__ == "__main__":
    main()
