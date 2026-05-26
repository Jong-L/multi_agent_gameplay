"""
Potential Function Comparison V2 Analysis
Process linear (lrrs) and exponential (exprs) data from v2 experiment
Generate comparison plots with Seaborn styling
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
FUNC_COLORS = {
    'lrrs': '#0173B2',   # Blue
    'exprs': '#DE8F05'   # Orange
}

FUNC_NAMES = {
    'lrrs': 'Linear',
    'exprs': 'Exponential'
}

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']


def load_and_aggregate_ball_data(file_paths):
    """Load CSV files, aggregate ball scores by episode and player"""
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
        print(f"\n  [!] ERROR: Inconsistent episode counts!")
        print(f"      Min: {min(unique_counts)}, Max: {max(unique_counts)}")
        print(f"      Cannot proceed with analysis.")
        return None
    else:
        max_episodes = list(unique_counts)[0]
        print(f"  [OK] Consistent: {max_episodes} episodes")
        return max_episodes


def compute_average_ball_rewards(file_data, max_episodes):
    """Calculate average ball reward per player per episode"""
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


def process_function_type(func_name, file_paths, output_dir):
    """Process one function type and save results"""
    print(f"\n{'='*60}")
    print(f"Processing {FUNC_NAMES[func_name].upper()}")
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
    
    output_path = f"{output_dir}\\{func_name}_average_ball_reward.csv"
    result_df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")
    
    print(f"\nSummary for {func_name}:")
    print(f"  Episodes: {max_episodes}, Players: {len(all_players)}")
    for player_id in all_players:
        col = f'player_{player_id}_ball_avg'
        avg = result_df[col].mean()
        print(f"  Player {player_id} avg: {avg:.2f}")
    print(f"  Total avg per episode: {result_df['total_ball_avg_reward'].mean():.2f}")
    
    return result_df


def smooth_curve(y, window=10):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def get_player_columns(df):
    """Get player column names"""
    player_cols = [col for col in df.columns if col.startswith('player_') and col.endswith('_ball_avg')]
    return sorted(player_cols)


def plot_player_comparison(data, save_path=None, smooth_window=10):
    """Plot 4 subplots (one per player) comparing lrrs vs exprs"""
    first_func = list(data.keys())[0]
    player_cols = get_player_columns(data[first_func]['df'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle('Ball Collection Reward per Player\n(Linear vs Exponential Potential Function)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        for func_key in ['lrrs', 'exprs']:
            if func_key not in data:
                continue
            df = data[func_key]['df']
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
                        label=FUNC_NAMES[func_key],
                        color=FUNC_COLORS[func_key],
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


def plot_total_comparison(data, save_path=None, smooth_window=10):
    """Plot total average comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    for func_key in ['lrrs', 'exprs']:
        if func_key not in data:
            continue
        df = data[func_key]['df']
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
                    label=FUNC_NAMES[func_key],
                    color=FUNC_COLORS[func_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward', fontsize=12)
    ax.set_title('Total Ball Collection Reward Comparison\n(Linear vs Exponential)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Potential Function')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    
    # Statistics annotation
    stats_text = "Mean Reward per Episode:\n"
    for func_key in ['lrrs', 'exprs']:
        if func_key in data:
            avg_reward = data[func_key]['df']['total_ball_avg_reward'].mean()
            stats_text += f"{FUNC_NAMES[func_key]}: {avg_reward:.1f}\n"
    
    if 'lrrs' in data and 'exprs' in data:
        lrrs_avg = data['lrrs']['df']['total_ball_avg_reward'].mean()
        exprs_avg = data['exprs']['df']['total_ball_avg_reward'].mean()
        if exprs_avg > 0:
            improvement = ((lrrs_avg - exprs_avg) / exprs_avg) * 100
            stats_text += f"\nLinear vs Exp: {improvement:+.1f}%"
    
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
    print("SUMMARY: V2 Comparison (Linear vs Exponential)")
    print("="*60)
    
    for func_key in ['lrrs', 'exprs']:
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
    
    # Calculate improvement
    if 'lrrs' in data and 'exprs' in data:
        lrrs_avg = data['lrrs']['df']['total_ball_avg_reward'].mean()
        exprs_avg = data['exprs']['df']['total_ball_avg_reward'].mean()
        if exprs_avg > 0:
            improvement = ((lrrs_avg - exprs_avg) / exprs_avg) * 100
            print(f"\n{'='*60}")
            print(f"KEY FINDING:")
            print(f"  Linear vs Exponential: {improvement:+.1f}%")
            print(f"  ({lrrs_avg:.1f} vs {exprs_avg:.1f} average reward)")
            print("="*60)


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\potential_function_comparison_v2"
    summary_dir = f"{base_dir}\\summary"
    
    # Ensure summary directory exists
    Path(summary_dir).mkdir(exist_ok=True)
    
    # Define file paths
    data = {
        'lrrs': [
            f"{base_dir}\\rewards_lrrs_04-24_23-35_pid7148.csv",
            f"{base_dir}\\rewards_lrrs_04-24_23-35_pid9160.csv",
            f"{base_dir}\\rewards_lrrs_04-24_23-35_pid9376.csv",
            f"{base_dir}\\rewards_lrrs_04-24_23-35_pid23260.csv",
            f"{base_dir}\\rewards_lrrs_04-24_23-35_pid28104.csv",
        ],
        'exprs': [
            f"{base_dir}\\rewards_exprs_04-24_22-50_pid10312.csv",
            f"{base_dir}\\rewards_exprs_04-24_22-50_pid22964.csv",
            f"{base_dir}\\rewards_exprs_04-24_22-50_pid27480.csv",
            f"{base_dir}\\rewards_exprs_04-24_22-50_pid29444.csv",
            f"{base_dir}\\rewards_exprs_04-24_22-50_pid30596.csv",
        ],
    }
    
    print("="*60)
    print("Potential Function Comparison V2 Analysis")
    print("="*60)
    print(f"Ball sources: {BALL_SOURCES}")
    
    # Process each function type
    results = {}
    for func_name, file_paths in data.items():
        df = process_function_type(func_name, file_paths, summary_dir)
        if df is not None:
            results[func_name] = {'df': df, 'name': FUNC_NAMES[func_name]}
    
    if len(results) == 0:
        print("\n[ERROR] No data processed successfully!")
        return
    
    # Print summary
    print_summary(results)
    
    # Create plots
    print("\nCreating plots...")
    
    fig1 = plot_player_comparison(results, 
                                  save_path=f"{summary_dir}\\player_comparison.png",
                                  smooth_window=10)
    
    fig2 = plot_total_comparison(results, 
                                 save_path=f"{summary_dir}\\total_comparison.png",
                                 smooth_window=10)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nResults saved in: {summary_dir}")
    print(f"  - lrrs_average_ball_reward.csv")
    print(f"  - exprs_average_ball_reward.csv")
    print(f"  - player_comparison.png")
    print(f"  - total_comparison.png")
    
    print("\nDisplaying plots... (close windows to exit)")
    plt.show()


if __name__ == "__main__":
    main()
