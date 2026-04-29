"""
Ball Reward Shaping Scheme Comparison Analysis
=============================================
Analyzes 4 potential shaping schemes across 10 independent runs each:
  Scheme 1 (LINEAR):       prefix=lrrs_
  Scheme 2 (EXPONENTIAL):  prefix=exprs_
  Scheme 3 (INVERSE):      prefix=invprs_
  Scheme 4 (DISTANCE_REWARD): prefix=distance_rs_

Outputs:
  - summary/*.csv  : aggregated data
  - summary/*.png  : seaborn plots (publication quality)

Data format: episode_id, player_id, source, value, game_time
Ball sources: collect_ball_A, collect_ball_B
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# ============================================================
# Configuration
# ============================================================
BASE_DIR = r"D:\schoolTour\softwares\multi-agent-gameplay\logs\ball_reward_shaping_v3"
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")

SCHEMES = {
    'LINEAR':           {'prefix': 'lrrs_',           'label': 'Linear',           'color': '#0173B2'},
    'EXPONENTIAL':      {'prefix': 'exprs_',          'label': 'Exponential',      'color': '#DE8F05'},
    'INVERSE':          {'prefix': 'invprs_',         'label': 'Inverse',          'color': '#029E73'},
    'DISTANCE_REWARD':  {'prefix': 'distance_rs_',    'label': 'Distance Reward',   'color': '#CC78BC'},
}
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
SMOOTH_WINDOW = 5


# ============================================================
# Data Loading
# ============================================================
def find_scheme_files(base_dir, prefix):
    """Find all CSV files matching a scheme prefix."""
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(prefix) and f.endswith('.csv'):
            files.append(os.path.join(base_dir, f))
    files.sort()
    return files


def load_and_aggregate_file(filepath):
    """
    Load a single CSV file and aggregate ball rewards per (episode_id, player_id).
    Returns: DataFrame with columns [episode_id, player_id, ball_reward]
             or None if file is empty/malformed.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(filepath)}: {e}")
        return None

    if df.empty or 'source' not in df.columns:
        return None

    # Filter ball sources
    ball_df = df[df['source'].isin(BALL_SOURCES)].copy()
    if ball_df.empty:
        # File has no ball records — return empty aggregated with correct columns
        return pd.DataFrame(columns=['episode_id', 'player_id', 'ball_reward'])

    # Aggregate: sum value per (episode_id, player_id)
    agg = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    agg.rename(columns={'value': 'ball_reward'}, inplace=True)
    return agg


def check_episode_consistency(file_list, scheme_name):
    """
    Check if all files in a scheme have the same max episode_id.
    Returns: max_episode (int) if consistent, None if inconsistent.
    """
    episode_counts = {}
    for fp in file_list:
        agg = load_and_aggregate_file(fp)
        if agg is None or agg.empty:
            max_ep = 0
        else:
            max_ep = agg['episode_id'].max()
        episode_counts[os.path.basename(fp)] = max_ep

    unique_counts = set(episode_counts.values())
    if len(unique_counts) > 1:
        print(f"\n  [ERROR] {scheme_name}: Episode counts are INCONSISTENT!")
        for fname, cnt in episode_counts.items():
            print(f"    {fname}: {cnt} episodes")
        return None

    max_ep = list(unique_counts)[0]
    print(f"  [OK] {scheme_name}: All {len(file_list)} files have {max_ep} episodes")
    return max_ep


def compute_scheme_average(file_list, scheme_name):
    """
    Compute per-episode per-player average ball reward across all files in a scheme.
    Rule: If a player is missing in a file's episode, score=0 for that file.
    Returns: DataFrame with columns [episode_id, player_0, player_1, ..., total_avg]
    """
    # Collect all (episode_id, player_id) pairs across all files
    all_records = []  # list of DataFrames, one per file
    for fp in file_list:
        agg = load_and_aggregate_file(fp)
        if agg is None:
            agg = pd.DataFrame(columns=['episode_id', 'player_id', 'ball_reward'])
        all_records.append(agg)

    # Get all unique (episode_id, player_id) combinations
    all_episodes = set()
    all_players = set()
    for rec in all_records:
        if not rec.empty:
            all_episodes.update(rec['episode_id'].unique())
            all_players.update(rec['player_id'].unique())
    all_episodes = sorted(list(all_episodes))
    all_players = sorted(list(all_players))

    if len(all_episodes) == 0:
        print(f"  [WARN] {scheme_name}: No data found!")
        return None

    num_files = len(file_list)
    results = []

    for ep in all_episodes:
        row = {'episode_id': ep}
        player_totals = {}
        for pid in all_players:
            total = 0.0
            for rec in all_records:
                mask = (rec['episode_id'] == ep) & (rec['player_id'] == pid)
                if mask.any():
                    total += rec.loc[mask, 'ball_reward'].values[0]
                # else: add 0 (player missing in this file's episode)
            player_totals[pid] = total / num_files
            row[f'player_{pid}'] = player_totals[pid]

        # Total = sum of all players' averages for this episode
        row['total_avg'] = sum(player_totals.values())
        results.append(row)

    result_df = pd.DataFrame(results)
    return result_df, all_players


# ============================================================
# Plotting
# ============================================================
def setup_seaborn_style():
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)


def smooth_curve(y, window=SMOOTH_WINDOW):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode='valid')


def plot_per_player(scheme_data, save_path):
    """
    Plot per-player average ball reward vs episode for all 4 schemes.
    4 subplots (one per player), each with 4 lines.
    """
    setup_seaborn_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Per-Player Average Ball Reward by Shaping Scheme',
                 fontsize=15, fontweight='bold', y=1.02)
    axes = axes.flatten()

    # Collect all player IDs from all schemes
    all_players = set()
    for scheme_key, (df, players) in scheme_data.items():
        all_players.update(players)
    all_players = sorted(list(all_players))

    for idx, pid in enumerate(all_players):
        ax = axes[idx]
        col = f'player_{pid}'

        for scheme_key, (df, players) in scheme_data.items():
            if col not in df.columns:
                continue
            episodes = df['episode_id'].values
            rewards = df[col].values

            # Smooth
            if len(rewards) >= SMOOTH_WINDOW:
                sm_rewards = smooth_curve(rewards, SMOOTH_WINDOW)
                sm_episodes = episodes[SMOOTH_WINDOW - 1:]
            else:
                sm_rewards = rewards
                sm_episodes = episodes

            ax.plot(sm_episodes, sm_rewards,
                    label=SCHEMES[scheme_key]['label'],
                    color=SCHEMES[scheme_key]['color'],
                    linewidth=2)

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Ball Reward', fontsize=11)
        ax.set_title(f'Player {pid}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_total(scheme_data, save_path):
    """
    Plot total average ball reward vs episode for all 4 schemes.
    1 plot with 4 lines.
    """
    setup_seaborn_style()
    fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=150)

    stats_text = "Mean Reward per Episode:\n"
    for scheme_key, (df, players) in scheme_data.items():
        episodes = df['episode_id'].values
        total = df['total_avg'].values

        if len(total) >= SMOOTH_WINDOW:
            sm_total = smooth_curve(total, SMOOTH_WINDOW)
            sm_episodes = episodes[SMOOTH_WINDOW - 1:]
        else:
            sm_total = total
            sm_episodes = episodes

        sns.lineplot(x=sm_episodes, y=sm_total,
                      label=SCHEMES[scheme_key]['label'],
                      color=SCHEMES[scheme_key]['color'],
                      linewidth=2.5, ax=ax)

        mean_val = np.mean(total)
        stats_text += f"  {SCHEMES[scheme_key]['label']}: {mean_val:.2f}\n"

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Average Ball Reward (all players)', fontsize=12)
    ax.set_title('Total Ball Collection Reward by Shaping Scheme',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, title='Scheme')
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    # Stats annotation
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                       edgecolor='gray'),
            fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_individual_player_comparison(scheme_data, player_id, save_path):
    """
    Standalone plot for a single player: 4 schemes compared.
    Useful for detailed per-player analysis.
    """
    setup_seaborn_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    col = f'player_{player_id}'
    for scheme_key, (df, players) in scheme_data.items():
        if col not in df.columns:
            continue
        episodes = df['episode_id'].values
        rewards = df[col].values

        if len(rewards) >= SMOOTH_WINDOW:
            sm_rewards = smooth_curve(rewards, SMOOTH_WINDOW)
            sm_episodes = episodes[SMOOTH_WINDOW - 1:]
        else:
            sm_rewards = rewards
            sm_episodes = episodes

        ax.plot(sm_episodes, sm_rewards,
                label=SCHEMES[scheme_key]['label'],
                color=SCHEMES[scheme_key]['color'],
                linewidth=2.5)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'Player {player_id} Average Ball Reward', fontsize=12)
    ax.set_title(f'Player {player_id} Ball Reward by Shaping Scheme',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Ball Reward Shaping Scheme Comparison Analysis")
    print("=" * 60)

    # Ensure summary directory exists
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    print(f"\nOutput directory: {SUMMARY_DIR}")

    scheme_data = {}  # key: scheme_key, value: (df, players)

    # --------------------------------------------------------
    # Step 1: Load and process each scheme
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 1/4] Loading and processing scheme data...")
    print("-" * 60)

    for scheme_key, scheme_info in SCHEMES.items():
        prefix = scheme_info['prefix']
        label = scheme_info['label']

        print(f"\n  Processing: {label} (prefix={prefix})")
        files = find_scheme_files(BASE_DIR, prefix)
        print(f"    Found {len(files)} files.")

        if len(files) == 0:
            print(f"  [WARN] No files found for {label}, skipping.")
            continue

        # Check episode consistency
        max_ep = check_episode_consistency(files, label)
        if max_ep is None:
            print(f"  [SKIP] {label} due to inconsistent episode counts.")
            continue

        # Compute average across files
        result = compute_scheme_average(files, label)
        if result is None:
            continue
        df, players = result
        scheme_data[scheme_key] = (df, players)
        print(f"    Computed average: {len(df)} episodes, players={players}")

        # Save per-scheme CSV
        csv_path = os.path.join(SUMMARY_DIR, f"{scheme_key.lower()}_average.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved CSV: {os.path.basename(csv_path)}")

    if len(scheme_data) == 0:
        print("\n[ERROR] No valid scheme data loaded. Exiting.")
        sys.exit(1)

    # --------------------------------------------------------
    # Step 2: Print summary statistics
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 2/4] Summary Statistics")
    print("-" * 60)

    for scheme_key, (df, players) in scheme_data.items():
        label = SCHEMES[scheme_key]['label']
        print(f"\n  {label}:")
        print(f"    Episodes: {len(df)}")
        for pid in players:
            col = f'player_{pid}'
            if col in df.columns:
                avg = df[col].mean()
                print(f"    Player {pid} avg: {avg:.2f}")
        total_avg = df['total_avg'].mean()
        print(f"    Total avg per episode: {total_avg:.2f}")

    # --------------------------------------------------------
    # Step 3: Generate plots
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 3/4] Generating plots...")
    print("-" * 60)

    # Plot 1: Per-player (4 subplots)
    print("\n  Plot 1: Per-player comparison (4 subplots)...")
    plot_per_player(scheme_data,
                    os.path.join(SUMMARY_DIR, "per_player_comparison.png"))

    # Plot 2: Total comparison
    print("\n  Plot 2: Total comparison...")
    plot_total(scheme_data,
               os.path.join(SUMMARY_DIR, "total_comparison.png"))

    # Plot 3-N: Individual player plots
    all_players = set()
    for scheme_key, (df, players) in scheme_data.items():
        all_players.update(players)
    for pid in sorted(list(all_players)):
        print(f"\n  Plot: Player {pid} standalone...")
        plot_individual_player_comparison(
            scheme_data, pid,
            os.path.join(SUMMARY_DIR, f"player_{pid}_comparison.png"))

    # --------------------------------------------------------
    # Step 4: Save combined CSV for further analysis
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Step 4/4] Saving combined data...")
    print("-" * 60)

    # Build a combined DataFrame with all schemes
    combined = None
    for scheme_key, (df, players) in scheme_data.items():
        label = SCHEMES[scheme_key]['label']
        df_copy = df.copy()
        df_copy['scheme'] = label
        if combined is None:
            combined = df_copy
        else:
            combined = pd.concat([combined, df_copy], ignore_index=True)

    if combined is not None:
        combined_path = os.path.join(SUMMARY_DIR, "all_schemes_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  [SAVED] {combined_path}")

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print(f"\n  Output files in: {SUMMARY_DIR}")
    print("  - per_player_comparison.png  : 4-subplot per-player comparison")
    print("  - total_comparison.png      : total average comparison")
    print("  - player_X_comparison.png  : individual player plots")
    print("  - *_average.csv             : per-scheme aggregated data")
    print("  - all_schemes_combined.csv  : combined data for further analysis")


if __name__ == "__main__":
    main()
