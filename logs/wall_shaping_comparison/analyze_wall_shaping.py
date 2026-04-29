"""
Wall Shaping Comparison Analysis - Extended
比较5种障碍物奖励塑形方案：
  原有3种: wall_min / wall_average / wall_weighted_average
  新增2种: wall_distance_penalty (额外距离惩罚) / wall_sparse (纯稀疏惩罚)
每组5个并行运行数据文件，分析吃球得分、碰墙次数与回合数的关系

产出图表(Group A - 所有5方案在公共回合数上的对比, 200ep):
  1. player_ball_score_comparison.png   - 各玩家吃球得分对比 (2x2子图, 5曲线)
  2. total_ball_score_comparison.png    - 总吃球得分对比 (1图, 5曲线)
  3. player_wall_count_comparison.png   - 各玩家碰墙次数对比 (2x2子图, 5曲线)
  4. total_wall_count_comparison.png    - 总碰墙次数对比 (1图, 5曲线)

产出图表(Group B - 新增2方案在完整回合数上的对比, 299ep):
  5. new_schemes_player_ball.png        - 各玩家吃球得分对比 (2x2子图, 2曲线)
  6. new_schemes_total_ball.png         - 总吃球得分对比 (1图, 2曲线)
  7. new_schemes_player_wall.png        - 各玩家碰墙次数对比 (2x2子图, 2曲线)
  8. new_schemes_total_wall.png         - 总碰墙次数对比 (1图, 2曲线)
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

# 所有实验方案
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
        'label': 'Wall Weighted Avg',
        'prefix': 'wall_wieghted_average_',  # 注意拼写typo
    },
    'wall_distance_penalty': {
        'label': 'Distance Penalty',
        'prefix': 'wall_distance_penalty_',
    },
    'wall_sparse': {
        'label': 'Sparse Penalty',
        'prefix': 'wall_sparse_',
    },
}

# 绘图顺序（控制显示顺序）
SCHEME_ORDER = [
    'wall_min',
    'wall_average',
    'wall_weighted_average',
    'wall_distance_penalty',
    'wall_sparse',
]

# 颜色方案 (colorblind-friendly, 5种颜色)
SCHEME_COLORS = {
    'wall_min': '#0173B2',              # Blue
    'wall_average': '#DE8F05',          # Orange
    'wall_weighted_average': '#029E73',  # Green
    'wall_distance_penalty': '#CC78BC',  # Purple
    'wall_sparse': '#D55E00',           # Reddish
}

SCHEME_LINESTYLES = {
    'wall_min': '-',
    'wall_average': '--',
    'wall_weighted_average': '-.',
    'wall_distance_penalty': ':',
    'wall_sparse': '-',
}

# 数据列定义
BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
WALL_COLLISION_SOURCE = 'wall_collision'  # 碰墙事件计数

# 移动平均窗口
SMOOTH_WINDOW = 10

# 玩家配置
PLAYER_IDS = [0, 1, 2, 3]

# ============================================================
# 步骤1: 发现数据文件
# ============================================================
def discover_files(schemes_dict):
    """扫描目录，按scheme分组发现CSV文件"""
    scheme_files = defaultdict(list)
    
    for csv_file in BASE_DIR.glob("*.csv"):
        fname = csv_file.name
        for scheme_key, scheme_info in schemes_dict.items():
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
      - wall_count: wall_collision 事件计数 (不是value值)
    """
    df = pd.read_csv(filepath)
    
    # 所有得分总和
    total_agg = df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    total_agg.columns = ['episode_id', 'player_id', 'total_score']
    
    # 吃球得分
    ball_df = df[df['source'].isin(BALL_SOURCES)]
    ball_agg = ball_df.groupby(['episode_id', 'player_id'])['value'].sum().reset_index()
    ball_agg.columns = ['episode_id', 'player_id', 'ball_score']
    
    # 碰墙次数 (计数wall_collision事件)
    wall_df = df[df['source'] == WALL_COLLISION_SOURCE]
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
    返回: list of DataFrames
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
    检查所有数据文件的回合数
    返回: (跨组是否一致, scheme最大回合数dict, 公共最小回合数, 警告列表)
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
    
    common_episodes = min(all_max_eps.values())
    
    if not is_consistent:
        msg = (f"[警告] 跨组回合数不一致！各组最大回合数: {all_max_eps}")
        issues.append(msg)
        print(f"\n{msg}")
        print(f"将使用公共最小回合数: {common_episodes}")
    else:
        print(f"\n[OK] 所有数据文件回合数一致: {common_episodes} 回合")
    
    return is_consistent, all_max_eps, common_episodes, issues


# ============================================================
# 步骤4: 计算平均值（含缺失玩家补0逻辑）
# ============================================================
def compute_scheme_averages(file_data_list, max_episodes):
    """
    对一个scheme组的所有并行运行文件，计算每个回合每个玩家的平均值
    规则: 如果某玩家在某文件的某回合不存在，该文件贡献为0
    """
    num_files = len(file_data_list)
    
    results = []
    
    for episode in range(1, max_episodes + 1):
        row = {'episode_id': episode}
        
        for player_id in PLAYER_IDS:
            ball_sum = 0.0
            total_sum = 0.0
            wall_count_sum = 0.0
            
            for fd in file_data_list:
                df = fd['df']
                mask = (df['episode_id'] == episode) & (df['player_id'] == player_id)
                match = df[mask]
                
                if len(match) > 0:
                    ball_sum += match['ball_score'].values[0]
                    total_sum += match['total_score'].values[0]
                    wall_count_sum += match['wall_count'].values[0]
                # else: 贡献为0 (默认)
            
            row[f'player_{player_id}_ball'] = ball_sum / num_files
            row[f'player_{player_id}_total'] = total_sum / num_files
            row[f'player_{player_id}_wall'] = wall_count_sum / num_files
        
        # 所有玩家总和
        row['all_ball'] = sum(row[f'player_{p}_ball'] for p in PLAYER_IDS)
        row['all_total'] = sum(row[f'player_{p}_total'] for p in PLAYER_IDS)
        row['all_wall'] = sum(row[f'player_{p}_wall'] for p in PLAYER_IDS)
        
        results.append(row)
    
    return pd.DataFrame(results)


# ============================================================
# 步骤5: 数据处理主流程
# ============================================================
def process_all_data(schemes_dict, scheme_order, description=""):
    """主数据处理流程"""
    print("=" * 70)
    print(f"障碍物奖励塑形方案对比 - {description}")
    print("=" * 70)
    
    # 5.1 发现文件
    print("\n[1/5] 扫描数据文件...")
    scheme_files = discover_files(schemes_dict)
    for scheme_key in scheme_order:
        if scheme_key in scheme_files:
            files = scheme_files[scheme_key]
            print(f"  {schemes_dict[scheme_key]['label']}: {len(files)} 个文件")
            for f in files:
                print(f"    - {f.name}")
        else:
            print(f"  {schemes_dict[scheme_key]['label']}: 未找到文件")
    
    # 5.2 加载数据
    print("\n[2/5] 加载与聚合数据...")
    all_scheme_data = {}
    for scheme_key in scheme_order:
        if scheme_key not in scheme_files:
            continue
        file_data_list = load_scheme_data(scheme_files[scheme_key])
        all_scheme_data[scheme_key] = file_data_list
        
        total_raw = sum(fd['raw_count'] for fd in file_data_list)
        total_agg = sum(len(fd['df']) for fd in file_data_list)
        print(f"  {schemes_dict[scheme_key]['label']}: "
              f"{total_raw} 条原始记录 -> {total_agg} 条聚合记录")
    
    # 5.3 一致性检查
    print("\n[3/5] 回合数一致性检查...")
    is_consistent, all_max_eps, common_episodes, issues = check_episode_consistency(all_scheme_data)
    
    if issues:
        print(f"\n检测到 {len(issues)} 个警告:")
        for issue in issues:
            print(f"  - {issue}")
    
    # 5.4 计算各scheme平均值
    print(f"\n[4/5] 计算各方案平均值 (每回合 {len(PLAYER_IDS)} 个玩家 × 5 次并行)...")
    scheme_averages = {}
    for scheme_key, file_data_list in all_scheme_data.items():
        avg_df = compute_scheme_averages(file_data_list, common_episodes)
        scheme_averages[scheme_key] = avg_df
        
        # 保存中间CSV
        csv_path = SUMMARY_DIR / f"{scheme_key}_episode_stats.csv"
        avg_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  已保存: {csv_path.name} ({len(avg_df)} 回合)")
    
    # 5.5 生成汇总统计
    print(f"\n[5/5] 生成汇总统计...")
    summary_rows = []
    for scheme_key in scheme_order:
        if scheme_key not in scheme_averages:
            continue
        avg_df = scheme_averages[scheme_key]
        row = {'scheme': schemes_dict[scheme_key]['label']}
        
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
    print(f"\n{'方案':<22} {'吃球得分':>10} {'总得分':>10} {'撞墙次数':>10}")
    print("-" * 55)
    for _, row in summary_df.iterrows():
        print(f"{row['scheme']:<22} {row['all_ball_avg']:>10.2f} "
              f"{row['all_total_avg']:>10.2f} {row['all_wall_avg']:>10.2f}")
    
    return scheme_averages, common_episodes, issues


# ============================================================
# 步骤6: 可视化工具函数
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
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 8.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def plot_player_metric(scheme_averages, scheme_order, schemes_dict,
                       global_max, save_path, metric='ball', ylabel='',
                       title='', legend_loc='upper left', ylim_bottom=None):
    """
    通用: 每个玩家的某指标 vs 回合数
    2x2子图，每条曲线对应一个scheme
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    
    axes = axes.flatten()
    
    for idx, player_id in enumerate(PLAYER_IDS):
        ax = axes[idx]
        col = f'player_{player_id}_{metric}'
        
        for scheme_key in scheme_order:
            if scheme_key not in scheme_averages:
                continue
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
                        label=schemes_dict[scheme_key]['label'],
                        color=SCHEME_COLORS[scheme_key],
                        linestyle=SCHEME_LINESTYLES[scheme_key],
                        linewidth=2.0,
                        ax=ax)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Player {player_id}', fontsize=12, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=8, frameon=True, framealpha=0.9)
        ax.set_xlim(0, global_max)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {save_path.name}")
    plt.close(fig)


def plot_total_metric(scheme_averages, scheme_order, schemes_dict,
                      global_max, save_path, metric='all_ball',
                      ylabel='', title='', legend_loc='upper left',
                      ylim_bottom=None, stats_prefix='Mean:'):
    """
    通用: 所有玩家的聚合指标 vs 回合数
    1图，每条曲线对应一个scheme
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    
    for scheme_key in scheme_order:
        if scheme_key not in scheme_averages:
            continue
        df = scheme_averages[scheme_key]
        episodes = df['episode_id'].values
        values = df[metric].values
        
        if SMOOTH_WINDOW > 1 and len(values) > SMOOTH_WINDOW:
            smoothed = smooth_curve(values, SMOOTH_WINDOW)
            smoothed_eps = episodes[SMOOTH_WINDOW - 1:]
        else:
            smoothed = values
            smoothed_eps = episodes
        
        sns.lineplot(x=smoothed_eps, y=smoothed,
                    label=schemes_dict[scheme_key]['label'],
                    color=SCHEME_COLORS[scheme_key],
                    linestyle=SCHEME_LINESTYLES[scheme_key],
                    linewidth=2.5,
                    ax=ax)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc=legend_loc, fontsize=9.5, frameon=True, framealpha=0.9,
             title='Wall Shaping Scheme')
    ax.set_xlim(0, global_max)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    
    # 添加统计注释
    stats_lines = [stats_prefix]
    for scheme_key in scheme_order:
        if scheme_key not in scheme_averages:
            continue
        avg = scheme_averages[scheme_key][metric].mean()
        stats_lines.append(f"  {schemes_dict[scheme_key]['label']}: {avg:.1f}")
    
    ax.text(0.98, 0.02, '\n'.join(stats_lines),
           transform=ax.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           fontsize=9, family='monospace')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {save_path.name}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================
def main():
    setup_style()
    
    # =============================================
    # GROUP A: 所有5方案对比 (公共回合数: 200)
    # =============================================
    all_schemes_dict = {
        k: SCHEMES[k] for k in ['wall_min', 'wall_average', 'wall_weighted_average',
                                'wall_distance_penalty', 'wall_sparse']
    }
    all_order = ['wall_min', 'wall_average', 'wall_weighted_average',
                 'wall_distance_penalty', 'wall_sparse']
    
    print("=" * 70)
    print("GROUP A: 所有5种方案对比")
    print("=" * 70)
    
    scheme_averages_all, common_ep_all, issues_all = process_all_data(
        all_schemes_dict, all_order, "所有5种方案"
    )
    
    if len(scheme_averages_all) == 0:
        print("[错误] 未找到任何数据文件！")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print(f"生成图表 - GROUP A ({common_ep_all} 回合)...")
    print("=" * 70)
    
    # A1: 各玩家吃球得分
    plot_player_metric(
        scheme_averages_all, all_order, all_schemes_dict, common_ep_all,
        SUMMARY_DIR / "player_ball_score_comparison.png",
        metric='ball', ylabel='Avg Ball Score',
        title='Average Ball Collection Score by Player\n(All Wall Shaping Schemes)',
        legend_loc='upper left', ylim_bottom=0
    )
    
    # A2: 总吃球得分
    plot_total_metric(
        scheme_averages_all, all_order, all_schemes_dict, common_ep_all,
        SUMMARY_DIR / "total_ball_score_comparison.png",
        metric='all_ball', ylabel='Average Ball Collection Score (All Players)',
        title='Total Ball Collection Score: All Wall Shaping Schemes',
        legend_loc='upper left', ylim_bottom=0, stats_prefix='Mean Ball Score:'
    )
    
    # A3: 各玩家碰墙次数
    plot_player_metric(
        scheme_averages_all, all_order, all_schemes_dict, common_ep_all,
        SUMMARY_DIR / "player_wall_count_comparison.png",
        metric='wall', ylabel='Avg Wall Collisions',
        title='Average Wall Collision Count by Player\n(All Wall Shaping Schemes)',
        legend_loc='upper right', ylim_bottom=0
    )
    
    # A4: 总碰墙次数
    plot_total_metric(
        scheme_averages_all, all_order, all_schemes_dict, common_ep_all,
        SUMMARY_DIR / "total_wall_count_comparison.png",
        metric='all_wall', ylabel='Average Wall Collision Count (All Players)',
        title='Total Wall Collision Count: All Wall Shaping Schemes',
        legend_loc='upper right', ylim_bottom=0, stats_prefix='Mean Wall Collisions:'
    )
    
    # =============================================
    # GROUP B: 新增2方案在完整回合数上对比 (299ep)
    # =============================================
    new_schemes_dict = {
        'wall_distance_penalty': SCHEMES['wall_distance_penalty'],
        'wall_sparse': SCHEMES['wall_sparse'],
    }
    new_order = ['wall_distance_penalty', 'wall_sparse']
    
    print("\n" + "=" * 70)
    print("GROUP B: 新增2方案完整对比 (299回合)")
    print("=" * 70)
    
    scheme_averages_new, common_ep_new, issues_new = process_all_data(
        new_schemes_dict, new_order, "新增2种方案完整对比"
    )
    
    print("\n" + "=" * 70)
    print(f"生成图表 - GROUP B ({common_ep_new} 回合)...")
    print("=" * 70)
    
    # B1: 新增方案 - 各玩家吃球得分
    plot_player_metric(
        scheme_averages_new, new_order, new_schemes_dict, common_ep_new,
        SUMMARY_DIR / "new_schemes_player_ball.png",
        metric='ball', ylabel='Avg Ball Score',
        title='Distance Penalty vs Sparse Penalty\n(Ball Collection Score by Player)',
        legend_loc='upper left', ylim_bottom=0
    )
    
    # B2: 新增方案 - 总吃球得分
    plot_total_metric(
        scheme_averages_new, new_order, new_schemes_dict, common_ep_new,
        SUMMARY_DIR / "new_schemes_total_ball.png",
        metric='all_ball', ylabel='Average Ball Collection Score (All Players)',
        title='Distance Penalty vs Sparse Penalty\n(Total Ball Collection Score)',
        legend_loc='upper left', ylim_bottom=0, stats_prefix='Mean Ball Score:'
    )
    
    # B3: 新增方案 - 各玩家碰墙次数
    plot_player_metric(
        scheme_averages_new, new_order, new_schemes_dict, common_ep_new,
        SUMMARY_DIR / "new_schemes_player_wall.png",
        metric='wall', ylabel='Avg Wall Collisions',
        title='Distance Penalty vs Sparse Penalty\n(Wall Collision Count by Player)',
        legend_loc='upper right', ylim_bottom=0
    )
    
    # B4: 新增方案 - 总碰墙次数
    plot_total_metric(
        scheme_averages_new, new_order, new_schemes_dict, common_ep_new,
        SUMMARY_DIR / "new_schemes_total_wall.png",
        metric='all_wall', ylabel='Average Wall Collision Count (All Players)',
        title='Distance Penalty vs Sparse Penalty\n(Total Wall Collision Count)',
        legend_loc='upper right', ylim_bottom=0, stats_prefix='Mean Wall Collisions:'
    )
    
    # =============================================
    # 完成报告
    # =============================================
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n中间数据保存在: {SUMMARY_DIR}")
    print(f"  - *_episode_stats.csv     (每组方案的逐回合聚合数据)")
    print(f"  - overall_summary.csv     (所有方案的汇总统计)")
    print(f"\nGROUP A - 所有5方案 ({common_ep_all}回合):")
    print(f"  1. player_ball_score_comparison.png  - 各玩家吃球得分")
    print(f"  2. total_ball_score_comparison.png   - 总吃球得分")
    print(f"  3. player_wall_count_comparison.png  - 各玩家碰墙次数")
    print(f"  4. total_wall_count_comparison.png   - 总碰墙次数")
    print(f"\nGROUP B - 新增2方案 ({common_ep_new}回合):")
    print(f"  5. new_schemes_player_ball.png       - 各玩家吃球得分")
    print(f"  6. new_schemes_total_ball.png        - 总吃球得分")
    print(f"  7. new_schemes_player_wall.png       - 各玩家碰墙次数")
    print(f"  8. new_schemes_total_wall.png        - 总碰墙次数")


if __name__ == "__main__":
    main()
