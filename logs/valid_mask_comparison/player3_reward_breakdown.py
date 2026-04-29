"""
Player 3 Reward Source Breakdown Analysis
Compares reward sources for Player 3 between no_valid_mask and use_valid_mask
Shows which specific behaviors improved with valid mask
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Color scheme
MASK_COLORS = {
    'no_valid_mask': '#E74C3C',
    'use_valid_mask': '#3498DB'
}

MASK_NAMES = {
    'no_valid_mask': 'No Valid Mask',
    'use_valid_mask': 'Use Valid Mask'
}

# Reward sources to analyze
REWARD_SOURCES = [
    'collect_ball_A',      # +10
    'collect_ball_B',      # +15
    'bear_damage',         # -10
    'cause_damage_to_enemy',  # +10
    'cause_damage_to_player', # +15
    'kill_enemy',          # +30
    'kill_player',         # +45
    'run',                 # -0.001
    'attack',              # -0.05
    'died',                # -20
    'wall_collision'       # -0.5
]


def load_data_files(base_dir):
    """Load all data files organized by mask type"""
    data_files = {
        'no_valid_mask': [],
        'use_valid_mask': []
    }
    
    base_path = Path(base_dir)
    
    for file in base_path.glob("*.csv"):
        fname = file.name
        if "player3" in fname or "breakdown" in fname:
            continue
        if fname.startswith("no_valid_mask"):
            data_files['no_valid_mask'].append(str(file))
        elif fname.startswith("use_valid_mask"):
            data_files['use_valid_mask'].append(str(file))
    
    for key in data_files:
        data_files[key].sort()
    
    return data_files


def analyze_player3_sources(file_paths, mask_type):
    """Analyze Player 3's reward sources across files"""
    all_data = []
    
    for fp in file_paths:
        df = pd.read_csv(fp)
        # Filter for Player 3 only
        p3_df = df[df['player_id'] == 3]
        all_data.append(p3_df)
    
    # Combine all files
    combined = pd.concat(all_data, ignore_index=True)
    num_files = len(file_paths)
    
    # Get max episodes
    max_episodes = combined['episode_id'].max()
    
    # Analyze per episode
    episode_stats = []
    
    for episode in range(1, max_episodes + 1):
        ep_df = combined[combined['episode_id'] == episode]
        ep_data = {'episode_id': episode}
        
        # Sum values for each source
        for source in REWARD_SOURCES:
            source_df = ep_df[ep_df['source'] == source]
            # Sum and average across files
            total_value = source_df['value'].sum() / num_files
            # Count occurrences
            count = len(source_df) / num_files
            
            ep_data[f'{source}_value'] = total_value
            ep_data[f'{source}_count'] = count
        
        # Calculate totals
        total_positive = sum(ep_data.get(f'{s}_value', 0) for s in REWARD_SOURCES 
                            if ep_data.get(f'{s}_value', 0) > 0)
        total_negative = sum(ep_data.get(f'{s}_value', 0) for s in REWARD_SOURCES 
                            if ep_data.get(f'{s}_value', 0) < 0)
        total_score = sum(ep_data.get(f'{s}_value', 0) for s in REWARD_SOURCES)
        
        ep_data['total_positive'] = total_positive
        ep_data['total_negative'] = total_negative
        ep_data['total_score'] = total_score
        
        episode_stats.append(ep_data)
    
    return pd.DataFrame(episode_stats), max_episodes


def plot_source_comparison(no_mask_df, use_mask_df, save_path=None):
    """Plot comparison of reward sources between two mask types"""
    
    # Calculate mean values for each source
    sources = REWARD_SOURCES
    no_mask_means = []
    use_mask_means = []
    
    for source in sources:
        no_mask_means.append(no_mask_df[f'{source}_value'].mean())
        use_mask_means.append(use_mask_df[f'{source}_value'].mean())
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Source': sources,
        'No Valid Mask': no_mask_means,
        'Use Valid Mask': use_mask_means
    })
    
    # Calculate differences
    comparison_df['Difference'] = comparison_df['Use Valid Mask'] - comparison_df['No Valid Mask']
    comparison_df['Improvement'] = comparison_df['Difference'].apply(lambda x: '↑' if x > 0 else '↓' if x < 0 else '-')
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # Plot 1: Side-by-side bar chart
    ax1 = axes[0]
    x = np.arange(len(sources))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, no_mask_means, width, label='No Valid Mask', 
                    color=MASK_COLORS['no_valid_mask'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, use_mask_means, width, label='Use Valid Mask', 
                    color=MASK_COLORS['use_valid_mask'], alpha=0.8)
    
    ax1.set_xlabel('Reward Source', fontsize=11)
    ax1.set_ylabel('Average Value per Episode', fontsize=11)
    ax1.set_title('Player 3: Reward Source Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in sources], rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Difference bar chart
    ax2 = axes[1]
    colors = ['#27AE60' if d > 0 else '#E74C3C' for d in comparison_df['Difference']]
    bars = ax2.bar(x, comparison_df['Difference'], color=colors, alpha=0.8)
    
    ax2.set_xlabel('Reward Source', fontsize=11)
    ax2.set_ylabel('Difference (Use - No Valid Mask)', fontsize=11)
    ax2.set_title('Player 3: Improvement with Valid Mask', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('_', '\n') for s in sources], rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, comparison_df['Difference']):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig, comparison_df


def plot_source_timeline(no_mask_df, use_mask_df, save_path=None):
    """Plot timeline of key reward sources"""
    
    # Select key sources to plot
    key_sources = ['collect_ball_A', 'collect_ball_B', 'wall_collision', 'died', 'kill_enemy', 'kill_player']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    axes = axes.flatten()
    
    for idx, source in enumerate(key_sources):
        ax = axes[idx]
        
        # Smooth the data
        window = 10
        no_mask_values = no_mask_df[f'{source}_value'].values
        use_mask_values = use_mask_df[f'{source}_value'].values
        
        if len(no_mask_values) >= window:
            no_mask_smooth = np.convolve(no_mask_values, np.ones(window)/window, mode='valid')
            use_mask_smooth = np.convolve(use_mask_values, np.ones(window)/window, mode='valid')
            episodes = no_mask_df['episode_id'].values[window-1:]
        else:
            no_mask_smooth = no_mask_values
            use_mask_smooth = use_mask_values
            episodes = no_mask_df['episode_id'].values
        
        sns.lineplot(x=episodes, y=no_mask_smooth, label='No Valid Mask', 
                    color=MASK_COLORS['no_valid_mask'], linewidth=2, ax=ax)
        sns.lineplot(x=episodes, y=use_mask_smooth, label='Use Valid Mask', 
                    color=MASK_COLORS['use_valid_mask'], linewidth=2, ax=ax)
        
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Average Value', fontsize=10)
        ax.set_title(source.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, None)
    
    plt.suptitle('Player 3: Key Reward Sources Timeline', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def print_detailed_breakdown(no_mask_df, use_mask_df):
    """Print detailed breakdown of reward sources"""
    
    print("\n" + "="*80)
    print("PLAYER 3 REWARD SOURCE BREAKDOWN")
    print("="*80)
    
    print("\n{:<25} {:>15} {:>15} {:>15} {:>10}".format(
        "Source", "No Mask", "Use Mask", "Difference", "Improvement"))
    print("-"*80)
    
    for source in REWARD_SOURCES:
        no_val = no_mask_df[f'{source}_value'].mean()
        use_val = use_mask_df[f'{source}_value'].mean()
        diff = use_val - no_val
        
        # Determine if this is good or bad
        # Positive sources: higher is better
        # Negative sources: higher (less negative) is better
        is_positive_source = source in ['collect_ball_A', 'collect_ball_B', 'cause_damage_to_enemy', 
                                         'cause_damage_to_player', 'kill_enemy', 'kill_player']
        
        if is_positive_source:
            improvement = "OK" if diff > 0 else "BAD" if diff < 0 else "-"
        else:
            improvement = "OK" if diff > 0 else "BAD" if diff < 0 else "-"
        
        print("{:<25} {:>15.2f} {:>15.2f} {:>+15.2f} {:>10}".format(
            source, no_val, use_val, diff, improvement))
    
    print("-"*80)
    
    # Summary
    no_total = no_mask_df['total_score'].mean()
    use_total = use_mask_df['total_score'].mean()
    total_diff = use_total - no_total
    
    print("\n{:<25} {:>15.2f} {:>15.2f} {:>+15.2f}".format(
        "TOTAL SCORE", no_total, use_total, total_diff))
    print("="*80)
    
    # Key insights
    print("\n[KEY INSIGHTS]:")
    
    # Find biggest improvements
    improvements = []
    for source in REWARD_SOURCES:
        no_val = no_mask_df[f'{source}_value'].mean()
        use_val = use_mask_df[f'{source}_value'].mean()
        diff = use_val - no_val
        improvements.append((source, diff))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 3 improvements (higher value with Valid Mask):")
    for source, diff in improvements[:3]:
        if diff > 0:
            print(f"  + {source}: +{diff:.2f}")
    
    print("\nTop 3 declines (lower value with Valid Mask):")
    for source, diff in improvements[-3:]:
        if diff < 0:
            print(f"  - {source}: {diff:.2f}")


def main():
    base_dir = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\valid_mask_comparison"
    summary_dir = Path(base_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Player 3 Reward Source Breakdown Analysis")
    print("="*80)
    
    # Load data files
    print("\n[1/4] Loading data files...")
    data_files = load_data_files(base_dir)
    print(f"  No Valid Mask: {len(data_files['no_valid_mask'])} files")
    print(f"  Use Valid Mask: {len(data_files['use_valid_mask'])} files")
    
    # Analyze Player 3
    print("\n[2/4] Analyzing Player 3 reward sources...")
    no_mask_df, no_episodes = analyze_player3_sources(data_files['no_valid_mask'], 'no_valid_mask')
    use_mask_df, use_episodes = analyze_player3_sources(data_files['use_valid_mask'], 'use_valid_mask')
    print(f"  No Valid Mask: {no_episodes} episodes")
    print(f"  Use Valid Mask: {use_episodes} episodes")
    
    # Save detailed data
    print("\n[3/4] Saving detailed data...")
    no_mask_df.to_csv(summary_dir / "player3_no_valid_mask_breakdown.csv", index=False)
    use_mask_df.to_csv(summary_dir / "player3_use_valid_mask_breakdown.csv", index=False)
    print(f"  Saved: player3_*_breakdown.csv")
    
    # Generate plots
    print("\n[4/4] Generating plots...")
    
    # Plot 1: Source comparison
    print("  Creating source comparison chart...")
    fig1, comparison_df = plot_source_comparison(no_mask_df, use_mask_df, 
                                                   save_path=summary_dir / "player3_source_comparison.png")
    
    # Plot 2: Timeline
    print("  Creating source timeline...")
    fig2 = plot_source_timeline(no_mask_df, use_mask_df,
                                save_path=summary_dir / "player3_source_timeline.png")
    
    # Print detailed breakdown
    print_detailed_breakdown(no_mask_df, use_mask_df)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nGenerated files in {summary_dir}:")
    print("  - player3_source_comparison.png")
    print("  - player3_source_timeline.png")
    print("  - player3_no_valid_mask_breakdown.csv")
    print("  - player3_use_valid_mask_breakdown.csv")


if __name__ == "__main__":
    main()
