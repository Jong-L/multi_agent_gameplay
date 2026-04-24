"""
Ball Reward Analysis: Nearest vs All Potential Calculation
Compare training with potential calculated for nearest ball only vs all balls
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
    """
    file_data = {}
    
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"Error: File not found - {fp}")
            sys.exit(1)
        
        df = pd.read_csv(fp)
        ball_df = df[df['source'].isin(BALL_SOURCES)]
        aggregated = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
        aggregated.columns = ['episode_id', 'player_id', 'total_ball_reward']
        
        file_data[path.name] = aggregated
        print(f"  Loaded: {path.name} - {len(df)} raw -> {len(ball_df)} ball -> {len(aggregated)} agg")
    
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
    
    all_players = set()
    for df in file_data.values():
        all_players.update(df['player_id'].unique())
    all_players = sorted(list(all_players))
    
    print(f"  Players: {all_players}, Files: {num_files}")
    
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


def process_scheme(scheme_name, file_paths, output_dir):
    """Process one scheme and save results"""
    print(f"\n{'='*60}")
    print(f"Processing {scheme_name.upper()}")
    print('='*60)
    
    print("[1/3] Loading data...")
    file_data = load_and_aggregate_ball_data(file_paths)
    
    print("[2/3] Checking consistency...")
    max_episodes = check_episode_consistency(file_data)
    
    if max_episodes is None:
        print("[ERROR] Skipping due to inconsistent episodes.")
        return None
    
    print("[3/3] Computing averages...")
    result_df, all_players = compute_average_ball_rewards(file_data, max_episodes)
    
    output_path = f"{output_dir}\\{scheme_name}_average_ball_reward.csv"
    result_df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")
    
    print(f"\nSummary for {scheme_name}:")
    print(f"  Episodes: {max_episodes}, Players: {len(all_players)}")
    for player_id in all_players:
        col = f'player_{player_id}_ball_avg'
        avg = result_df[col].mean()
        print(f"  Player {player_id} avg: {avg:.2f}")
    print(f"  Total avg per episode: {result_df['total_ball_avg_reward'].mean():.2f}")
    
    return result_df


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\game_reward_log"
    
    # Define file paths for each scheme
    data = {
        'nearest': [
            f"{base_dir}\\ball_rewards_nearest_lrrs_2026-04-24_15-07-06_pid8068.csv",
            f"{base_dir}\\ball_rewards_nearest_lrrs_2026-04-24_15-07-07_pid5484.csv",
            f"{base_dir}\\ball_rewards_nearest_lrrs_2026-04-24_15-07-09_pid27640.csv",
            f"{base_dir}\\ball_rewards_nearest_lrrs_2026-04-24_15-07-10_pid16604.csv",
        ],
        'all': [
            f"{base_dir}\\ball_rewards_all_lrrs_2026-04-24_16-01-45_pid20240.csv",
            f"{base_dir}\\ball_rewards_all_lrrs_2026-04-24_16-01-46_pid22036.csv",
            f"{base_dir}\\ball_rewards_all_lrrs_2026-04-24_16-01-47_pid30784.csv",
            f"{base_dir}\\ball_rewards_all_lrrs_2026-04-24_16-01-49_pid30280.csv",
        ],
    }
    
    print("="*60)
    print("Ball Reward Analysis: Nearest vs All Potential Calculation")
    print("="*60)
    print(f"Ball sources: {BALL_SOURCES}")
    
    results = {}
    for scheme_name, file_paths in data.items():
        df = process_scheme(scheme_name, file_paths, base_dir)
        if df is not None:
            results[scheme_name] = df
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)
    print(f"\nGenerated files:")
    for scheme_name in results.keys():
        print(f"  - {scheme_name}_average_ball_reward.csv")


if __name__ == "__main__":
    main()
