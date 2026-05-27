"""
Nearest vs All Potential Calculation Comparison
=================================================
比较 nearest-ball 和 all-balls 势能方案的球收集奖励。
使用两级平滑管线，产出论文级图表。

Window 1: 4 subplots (one per player), each with 2 curves
Window 2: 1 subplot with total average, 2 curves
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from publication_plot_utils import (
    setup_style, save_figure,
    prepare_curve, plot_with_fill, add_stats_box, style_axes,
    COLORS,
)

# ── 配置 ────────────────────────────────────────────
SCHEME_COLORS = {
    'nearest': COLORS['blue'],    # #0173B2
    'all':     COLORS['orange'],  # #DE8F05
}
SCHEME_NAMES = {
    'nearest': '仅最近球',
    'all':     '所有球',
}


def load_data(base_dir):
    """Load both scheme data"""
    data = {}
    for scheme in ['nearest', 'all']:
        file_path = f"{base_dir}\\{scheme}_average_ball_reward.csv"
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            data[scheme] = {'df': df, 'name': SCHEME_NAMES[scheme]}
            print(f"已加载 {scheme} 方案数据：{len(df)} 回合")
        else:
            print(f"警告：{file_path} 未找到")
    return data


def get_player_columns(df):
    """Get player column names"""
    return sorted([c for c in df.columns
                   if c.startswith('player_') and c.endswith('_ball_avg')])


def plot_player_comparison(data, save_path=None, smooth_window=5):
    """4 subplots (one per player), each with 2 curves"""
    first_scheme = list(data.keys())[0]
    player_cols = get_player_columns(data[first_scheme]['df'])
    
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('奖励球势能计算方式对比\n（各智能体）',
                 fontsize=16, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    for idx, player_col in enumerate(player_cols):
        ax = axes[idx]
        player_id = player_col.split('_')[1]
        
        for scheme_key, color in SCHEME_COLORS.items():
            if scheme_key not in data:
                continue
            df = data[scheme_key]['df']
            episodes = df['episode_id'].values
            rewards = df[player_col].values
            
            half_band = np.max(np.abs(rewards)) * 0.05
            errors = np.full_like(rewards, half_band, dtype=float)
            
            x_s, y_s, y_l, y_u = prepare_curve(
                episodes, rewards, errors=errors,
                smooth_window=smooth_window,
            )
            
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=color, label=SCHEME_NAMES[scheme_key])
        
        style_axes(ax, xlabel='训练回合', ylabel='平均球收集奖励',
                   title=f'智能体 {player_id}',
                   legend_kwargs={'title': '势能计算方式', 'fontsize': 8})
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_total_comparison(data, save_path=None, smooth_window=5):
    """1 subplot with total average, 2 curves"""
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    stats_text = "回合均分:\n"
    
    for scheme_key, color in SCHEME_COLORS.items():
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        episodes = df['episode_id'].values
        total = df['total_ball_avg_reward'].values
        
        half_band = np.max(np.abs(total)) * 0.05
        errors = np.full_like(total, half_band, dtype=float)
        
        x_s, y_s, y_l, y_u = prepare_curve(
            episodes, total, errors=errors,
            smooth_window=smooth_window,
        )
        
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=color, label=SCHEME_NAMES[scheme_key])
        
        avg_reward = total.mean()
        stats_text += f"{SCHEME_NAMES[scheme_key]}: {avg_reward:.1f}\n"
    
    # 计算改善幅度
    if 'nearest' in data and 'all' in data:
        nearest_avg = data['nearest']['df']['total_ball_avg_reward'].mean()
        all_avg = data['all']['df']['total_ball_avg_reward'].mean()
        improvement = ((nearest_avg - all_avg) / all_avg) * 100
        stats_text += f"\n提升幅度：+{improvement:.1f}%"
    
    style_axes(ax, xlabel='训练回合', ylabel='全场平均球收集奖励',
               title='全场球收集奖励对比',
               legend_kwargs={'title': '势能计算方式'})
    
    add_stats_box(ax, stats_text)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def print_summary(data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("摘要：奖励球势能计算方式对比")
    print("="*60)
    for scheme_key in ['nearest', 'all']:
        if scheme_key not in data:
            continue
        df = data[scheme_key]['df']
        print(f"\n{SCHEME_NAMES[scheme_key]}方案：")
        print(f"  回合数：{len(df)}")
        for col in get_player_columns(df):
            pid = col.split('_')[1]
            print(f"  智能体 {pid} 均值：{df[col].mean():.2f}")
        print(f"  全场回合均值：{df['total_ball_avg_reward'].mean():.2f}")
    
    if 'nearest' in data and 'all' in data:
        nearest_avg = data['nearest']['df']['total_ball_avg_reward'].mean()
        all_avg = data['all']['df']['total_ball_avg_reward'].mean()
        improvement = ((nearest_avg - all_avg) / all_avg) * 100
        print(f"\n{'='*60}")
        print(f"核心发现：")
        print(f"  仅最近球方案球收集提升 {improvement:.1f}%")
        print(f"  （仅最近球 {nearest_avg:.1f} vs 所有球 {all_avg:.1f}）")
        print("="*60)


def main():
    base_dir = r"experiment_data\game_reward_log"
    
    print("="*60)
    print("奖励球势能计算方式对比：仅最近球 vs 所有球")
    print("="*60)
    
    print("\n正在加载数据...")
    data = load_data(base_dir)
    
    if len(data) == 0:
        print("未找到数据文件！")
        return
    
    print_summary(data)
    
    print("\n正在生成图表...")
    plot_player_comparison(data,
                           save_path=f"{base_dir}\\nearest_vs_all_player_comparison.png",
                           smooth_window=5)
    plot_total_comparison(data,
                          save_path=f"{base_dir}\\nearest_vs_all_total_comparison.png",
                          smooth_window=5)
    
    print("\n" + "="*60)
    print("图表已生成（300 DPI）！")
    print("="*60)


if __name__ == "__main__":
    main()
