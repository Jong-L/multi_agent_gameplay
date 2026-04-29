"""
Wall Shaping Comparison Analysis
比较3种障碍物奖励塑形方案：wall_min / wall_average / wall_weighted_average
每组5个并行运行数据文件，分析吃球得分、总得分、撞墙次数与回合数的关系

产出4张图表：
  1. 每个玩家的平均吃球得分 vs 回合数 (2x2子图, 3条曲线)
  2. 所有玩家平均总得分 vs 回合数 (1图, 3条曲线)
  3. 每个玩家的平均撞墙次数 vs 回合数 (2x2子图, 3条曲线)
  4. 所有玩家平均撞墙次数 vs 回合数 (1图, 3条曲线)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(r"D:\schoolTour\softwares\multi-agent-gameplay\logs\wall_shaping_comparison")
SUMMARY_DIR = BASE_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# 三组实验方案
SCHEMES = {
    'wall_min': {
        'label': 'Wall Min (Nearest)',
        'prefix': 'wall_min_',
    },
    'wall_average': {
        'label': 'Wall Average',
        'prefix': 'wall_average_',
    },
    'wall_weighted_average': {
        'label': 'Wall Weighted Average',
        'prefix': 'wall_wieghted_average_',  # 注意拼写typo
    },
}

# 颜色方案 (colorblind-friendly)
SCHEME_COLORS = {
    'wall_min': '#0173B2',              # Blue
    'wall_average': '#DE8F05',          # Orange
    'wall_weighted_average': '#029E73',  # Green
}

SCHEME_LINESTYLES = {
    'wall_min': '-',
    'wall_average': '--',
    'wall_weighted_average': '-.',
}

# 数据列定义
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
WALL_SOURCE = 'wall_collision'

# 移动平均窗口
SMOOTH_WINDOW = 10

# 玩家配置
NUM_PLAYERS = 4
PLAYER_IDS = [0, 1, 2, 3]

# ============================================================
# 步骤1: 发现数据文件
# ============================================================
def discover_files():
    """扫描目录，按scheme分组发现CSV文件"""
    scheme_files = defaultdict(list)
    
    for csv_file in BASE_DIR.glob("*.csv"):
        fname = csv_file.name
        for scheme_key, scheme_info in SCHEMES.items():
            if fname.startswith(scheme_info['prefix']):
                scheme_files[scheme_key].append(csv_file)
                break
    
    # 排序以确保一致性
    for key in scheme_files:
        scheme_files[key] = sorted(scheme_files[key])
    
    return scheme_files


# ============================================================
# 步骤2: 数据加载与聚合
# ============================================================
def load_and_aggregate_file(filepath):
    """
    加载单个CSV文件，按 (episode_id, player_id) 聚合：
      - total_score: 所有source的value求和
      - ball_score: collect_ball_A + collect_ball_B 的value求和
      - wall_count: wall_collision 事件计数
    """
    df = pd.read_csv(filepath)
    
    # 所有得分总和
    total_agg = df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    total_agg.columns = ['episode_id', 'player_id', 'total_score']
    
    # 吃球得分
    ball_df = df[df['source'].isin(BALL_SOURCES)]
    ball_agg = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    ball_agg.columns = ['episode_id', 'player_id', 'ball_score']
    
    # 撞墙次数
    wall_df = df[df['source'] == WALL_SOURCE]
    wall_agg = wall_df.groupby(['episode_id', 'player_id']).size().reset_index(name='wall_count')
    
    # 合并
    merged = total_agg.merge(ball_agg, on=['episode_id', 'player_id'], how='left')
    merged = merged.merge(wall_agg, on=['episode_id', 'player_id'], how='left')
    merged['ball_score'] = merged['ball_score'].fillna(0.0)
    merged['wall_count'] = merged['wall_count'].fillna(0.0)
    
    return merged, len(df)


def load_scheme_data(scheme_files):
    """
    加载一个scheme组的所有文件数据
    返回: list of DataFrames, 每个元素是一个文件的聚合数据
    """
    file_data_list = []
    for fp in scheme_files:
        agg_df, raw_count = load_and_aggregate_file(fp)
        file_data_list.append({
            'filename': fp.name,
            'df': agg_df,
            'raw_count': raw_count,
            'max_episode': agg_df['episode_id'].max(),
            'players': sorted(agg_df['player_id'].unique().tolist()),
        })
    return file_data_list


# ============================================================
# 步骤3: 回合数一致性检查
# ============================================================
def check_episode_consistency(all_scheme_data):
    """
    检查所有数据文件的回合数是否一致
    输出警告信息，但允许不一致时仍然进行分析
    返回: (is_consistent, max_episodes_dict, warnings)
    """
    issues = []
    all_max_eps = {}
    
    for scheme_key, file_data_list in all_scheme_data.items():
        max_eps = [fd['max_episode'] for fd in file_data_list]
        unique_max = set(max_eps)
        
        print(f"\n[{scheme_key}] 各文件最大回合数: {max_eps}")
        
        if len(unique_max) > 1:
            msg = (f"[警告] {scheme_key} 组内文件回合数不一致！"
                   f" (min={min(unique_max)}, max={max(unique_max)})")
            issues.append(msg)
            print(f"  {msg}")
        
        all_max_eps[scheme_key] = max(max_eps)
    
    # 跨组检查
    unique_across = set(all_max_eps.values())
    is_consistent = len(unique_across) == 1
    
    if not is_consistent:
        msg = (f"[警告] 跨组回合数不一致！各组最大回合数: {all_max_eps}")
        issues.append(msg)
        print(f"\n{msg}")
        global_max = max(all_max_eps.values())
    else:
        global_max = list(unique_across)[0]
        print(f"\n[OK] 所有数据文件回合数一致: {global_max} 回合")
    
    return is_consistent, all_max_eps, global_max, issues


# ============================================================
# 步骤4: 计算平均值（含缺失玩家补0逻辑）
# ============================================================
def compute_scheme_averages(file_data_list, max_episodes):
    """
    对一个scheme组的5个并行运行文件，计算每个回合每个玩家的平均值
    规则: 如果某玩家在某文件的某回合不存在，该文件贡献为0
    """
    num_files = len(file_data_list)
    
    results = []
    
    for episode in range(1, max_episodes + 1):
        row = {'episode_id': episode}
        
        for player_id in PLAYER_IDS:
            ball_sum = 0.0
            total_sum = 0.0
            wall_sum = 0.0
            
            for fd in file_data_list:
                df = fd['df']
                mask = (df['episode_id'] == episode) & (df['player_id'] == player_id)
                match = df[mask]
                
                if len(match) > 0:
                    ball_sum += match['ball_score'].values[0]
                    total_sum += match['total_score'].values[0]
                    wall_sum += match['wall_count'].values[0]
                # else: 贡献为0 (默认)
            
            row[f'player_{player_id}_ball'] = ball_sum / num_files
            row[f'player_{player_id}_total'] = total_sum / num_files
            row[f'player_{player_id}_wall'] = wall_sum / num_files
        
        # 所有玩家总和
        row['all_ball'] = sum(row[f'player_{p}_ball'] for p in PLAYER_IDS)
        row['all_total'] = sum(row[f'player_{p}_total'] for p in PLAYER_IDS)
        row['all_wall'] = sum(row[f'player_{p}_wall'] for p in PLAYER_IDS)
        
        results.append(row)
    
    return pd.DataFrame(results)


# ============================================================
# 步骤5: 数据处理主流程
# ============================================================
def process_all_data():
    """主数据处理流程"""
    print("=" * 70)
    print("障碍物奖励塑形方案对比 - 数据分析")
    print("=" * 70)
    
    # 5.1 发现文件
    print("\n[1/5] 扫描数据文件...")
    scheme_files = discover_files()
    for scheme_key, files in scheme_files.items():
        print(f"  {SCHEMES[scheme_key]['label']}: {len(files)} 个文件")
        for f in files:
            print(f"    - {f.name}")
    
    # 5.2 加载数据
    print("\n[2/5] 加载与聚合数据...")
    all_scheme_data = {}
    for scheme_key, files in scheme_files.items():
        file_data_list = load_scheme_data(files)
        all_scheme_data[scheme_key] = file_data_list
        
        total_raw = sum(fd['raw_count'] for fd in file_data_list)
        total_agg = sum(len(fd['df']) for fd in file_data_list)
        print(f"  {SCHEMES[scheme_key]['label']}: "
              f"{total_raw} 条原始记录 -> {total_agg} 条聚合记录")
    
    # 5.3 一致性检查
    print("\n[3/5] 回合数一致性检查...")
    is_consistent, all_max_eps, global_max, issues = check_episode_consistency(all_scheme_data)
    
    if issues:
        print(f"\n检测到 {len(issues)} 个警告:")
        for issue in issues:
            print(f"  - {issue}")
        if not is_consistent:
            print(f"\n将使用最大回合数 {global_max} 进行分析，"
                  f"回合数较少的文件在超出范围时贡献为0。")
    
    # 5.4 计算各scheme平均值
    print(f"\n[4/5] 计算各方案平均值 (每回合 {len(PLAYER_IDS)} 个玩家 × {5} 次并行)...")
    scheme_averages = {}
    for scheme_key, file_data_list in all_scheme_data.items():
        avg_df = compute_scheme_averages(file_data_list, global_max)
        scheme_averages[scheme_key] = avg_df
        
        # 保存中间CSV
        csv_path = SUMMARY_DIR / f"{scheme_key}_episode_stats.csv"
        avg_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  已保存: {csv_path.name} ({len(avg_df)} 回合)")
    
    # 5.5 生成汇总统计
    print(f"\n[5/5] 生成汇总统计...")
    summary_rows = []
    for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
        avg_df = scheme_averages[scheme_key]
        row = {'scheme': SCHEMES[scheme_key]['label']}
        
        for player_id in PLAYER_IDS:
            row[f'player_{player_id}_ball_avg'] = avg_df[f'player_{player_id}_ball'].mean()
            row[f'player_{player_id}_wall_avg'] = avg_df[f'player_{player_id}_wall'].mean()
        
        row['all_ball_avg'] = avg_df['all_ball'].mean()
        row['all_total_avg'] = avg_df['all_total'].mean()
        row['all_wall_avg'] = avg_df['all_wall'].mean()
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = SUMMARY_DIR / "overall_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format='%.2f')
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    print(f"\n{'方案':<30} {'平均吃球得分':>14} {'平均总得分':>14} {'平均撞墙次数':>14}")
    print("-" * 75)
    for _, row in summary_df.iterrows():
        print(f"{row['scheme']:<30} {row['all_ball_avg']:>14.2f} "
              f"{row['all_total_avg']:>14.2f} {row['all_wall_avg']:>14.2f}")
    
    # 分玩家统计
    print(f"\n--- 各玩家平均吃球得分 ---")
    header = f"{'方案':<30}"
    for p in PLAYER_IDS:
        header += f" {'Player ' + str(p):>12}"
    print(header)
    print("-" * (30 + 13 * 4))
    for _, row in summary_df.iterrows():
        line = f"{row['scheme']:<30}"
        for p in PLAYER_IDS:
            line += f" {row[f'player_{p}_ball_avg']:>12.2f}"
        print(line)
    
    print(f"\n--- 各玩家平均撞墙次数 ---")
    for _, row in summary_df.iterrows():
        line = f"{row['scheme']:<30}"
        for p in PLAYER_IDS:
            line += f" {row[f'player_{p}_wall_avg']:>12.2f}"
        print(line)
    
    return scheme_averages, global_max, is_consistent, issues


# ============================================================
# 步骤6: 可视化
# ============================================================
def smooth_curve(y, window=SMOOTH_WINDOW):
    """移动平均平滑"""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def setup_style():
    """设置seaborn论文风格"""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    # 使用colorblind友好的调色板
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def plot_player_ball_comparison(scheme_averages, global_max, save_path):
    """
    图表1: 每个玩家的平均吃球得分 vs 回合数
    2x2子图，每子图3条曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Average Ball Collection Score by Player\n(Wall Shaping Scheme Comparison)',
                 fontsize=15, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[idx]
        col = f'player_{player_id}_ball'
        
        for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
            df = scheme_averages[scheme_key]
            episodes = df['episode_id'].values
            values = df[col].values
            
            if SMOOTH_WINDOW > 1 and len(values) > SMOOTH_WINDOW:
                smoothed = smooth_curve(values, SMOOTH_WINDOW)
                smoothed_eps = episodes[SMOOTH_WINDOW - 1:]
            else:
                smoothed = values
                smoothed_eps = episodes
            
            sns.lineplot(x=smoothed_eps, y=smoothed,
                        label=SCHEMES[scheme_key]['label'],
                        color=SCHEME_COLORS[scheme_key],
                        linestyle=SCHEME_LINESTYLES[scheme_key],
                        linewidth=2.0,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Avg Ball Score', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
        ax.set_xlim(0, global_max)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存图表: {save_path.name}")
    plt.close(fig)


def plot_total_score_comparison(scheme_averages, global_max, save_path):
    """
    图表2: 所有玩家平均吃球得分总和 vs 回合数
    1图，3条曲线
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    
    for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
        df = scheme_averages[scheme_key]
        episodes = df['episode_id'].values
        values = df['all_ball'].values
        
        if SMOOTH_WINDOW > 1 and len(values) > SMOOTH_WINDOW:
            smoothed = smooth_curve(values, SMOOTH_WINDOW)
            smoothed_eps = episodes[SMOOTH_WINDOW - 1:]
        else:
            smoothed = values
            smoothed_eps = episodes
        
        sns.lineplot(x=smoothed_eps, y=smoothed,
                    label=SCHEMES[scheme_key]['label'],
                    color=SCHEME_COLORS[scheme_key],
                    linestyle=SCHEME_LINESTYLES[scheme_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Ball Collection Score (All Players)', fontsize=12)
    ax.set_title('Total Ball Collection Score Comparison: Wall Shaping Schemes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9,
             title='Wall Shaping Scheme')
    ax.set_xlim(0, global_max)
    
    # 添加统计注释
    stats_lines = ["Mean Ball Score:"]
    for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
        avg = scheme_averages[scheme_key]['all_ball'].mean()
        stats_lines.append(f"  {SCHEMES[scheme_key]['label']}: {avg:.1f}")
    
    ax.text(0.98, 0.02, '\n'.join(stats_lines),
           transform=ax.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=9, family='monospace')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存图表: {save_path.name}")
    plt.close(fig)


def plot_player_wall_comparison(scheme_averages, global_max, save_path):
    """
    图表3: 每个玩家的平均撞墙次数 vs 回合数
    2x2子图，每子图3条曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Average Wall Collision Count by Player\n(Wall Shaping Scheme Comparison)',
                 fontsize=15, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[idx]
        col = f'player_{player_id}_wall'
        
        for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
            df = scheme_averages[scheme_key]
            episodes = df['episode_id'].values
            values = df[col].values
            
            if SMOOTH_WINDOW > 1 and len(values) > SMOOTH_WINDOW:
                smoothed = smooth_curve(values, SMOOTH_WINDOW)
                smoothed_eps = episodes[SMOOTH_WINDOW - 1:]
            else:
                smoothed = values
                smoothed_eps = episodes
            
            sns.lineplot(x=smoothed_eps, y=smoothed,
                        label=SCHEMES[scheme_key]['label'],
                        color=SCHEME_COLORS[scheme_key],
                        linestyle=SCHEME_LINESTYLES[scheme_key],
                        linewidth=2.0,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Avg Wall Collisions', fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
        ax.set_xlim(0, global_max)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存图表: {save_path.name}")
    plt.close(fig)


def plot_total_wall_comparison(scheme_averages, global_max, save_path):
    """
    图表4: 所有玩家平均撞墙次数 vs 回合数
    1图，3条曲线
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    
    for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
        df = scheme_averages[scheme_key]
        episodes = df['episode_id'].values
        values = df['all_wall'].values
        
        if SMOOTH_WINDOW > 1 and len(values) > SMOOTH_WINDOW:
            smoothed = smooth_curve(values, SMOOTH_WINDOW)
            smoothed_eps = episodes[SMOOTH_WINDOW - 1:]
        else:
            smoothed = values
            smoothed_eps = episodes
        
        sns.lineplot(x=smoothed_eps, y=smoothed,
                    label=SCHEMES[scheme_key]['label'],
                    color=SCHEME_COLORS[scheme_key],
                    linestyle=SCHEME_LINESTYLES[scheme_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Wall Collision Count (All Players)', fontsize=12)
    ax.set_title('Wall Collision Comparison: Wall Shaping Schemes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9,
             title='Wall Shaping Scheme')
    ax.set_xlim(0, global_max)
    ax.set_ylim(bottom=0)
    
    # 添加统计注释
    stats_lines = ["Mean Wall Collisions:"]
    for scheme_key in ['wall_min', 'wall_average', 'wall_weighted_average']:
        avg = scheme_averages[scheme_key]['all_wall'].mean()
        stats_lines.append(f"  {SCHEMES[scheme_key]['label']}: {avg:.1f}")
    
    ax.text(0.98, 0.98, '\n'.join(stats_lines),
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=9, family='monospace')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存图表: {save_path.name}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================
def main():
    setup_style()
    
    # 处理数据
    scheme_averages, global_max, is_consistent, issues = process_all_data()
    
    if len(scheme_averages) == 0:
        print("[错误] 未找到任何数据文件！")
        sys.exit(1)
    
    # 生成图表
    print("\n" + "=" * 70)
    print("生成图表 (Seaborn 论文风格)...")
    print("=" * 70)
    
    plot_player_ball_comparison(
        scheme_averages, global_max,
        SUMMARY_DIR / "player_ball_score_comparison.png"
    )
    
    plot_total_score_comparison(
        scheme_averages, global_max,
        SUMMARY_DIR / "total_score_comparison.png"
    )
    
    plot_player_wall_comparison(
        scheme_averages, global_max,
        SUMMARY_DIR / "player_wall_count_comparison.png"
    )
    
    plot_total_wall_comparison(
        scheme_averages, global_max,
        SUMMARY_DIR / "total_wall_count_comparison.png"
    )
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n中间数据保存在: {SUMMARY_DIR}")
    print(f"  - wall_min_episode_stats.csv")
    print(f"  - wall_average_episode_stats.csv")
    print(f"  - wall_weighted_average_episode_stats.csv")
    print(f"  - overall_summary.csv")
    print(f"\n图表保存在: {SUMMARY_DIR}")
    print(f"  1. player_ball_score_comparison.png  - 各玩家吃球得分对比")
    print(f"  2. total_score_comparison.png        - 总吃球得分对比")
    print(f"  3. player_wall_count_comparison.png  - 各玩家撞墙次数对比")
    print(f"  4. total_wall_count_comparison.png   - 总撞墙次数对比")
    
    if issues:
        print(f"\n⚠ 数据警告 ({len(issues)}条):")
        for issue in issues:
            print(f"  {issue}")


if __name__ == "__main__":
    main()
