"""
Wall Penalty — 论文级重绘 (基于 wall_penalty_ball_reward_analysis.py)
=======================================================================
数据处理逻辑与原脚本完全一致, 仅替换绘图管线为 publication_plot_utils:
  conv → gaussian_filter1d(sigma=window×0.35) → B-spline(k=3, 200pts) → fill_between(α=0.18)

输出 4 张图:
1. player_ball_score_comparison.png  — 4 子图, 各玩家吃球得分
2. total_ball_score_comparison.png   — 全场平均吃球得分
3. player_wall_count_comparison.png  — 4 子图, 各玩家撞墙次数
4. total_wall_count_comparison.png   — 全场平均撞墙次数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "data_analyze"))
from publication_plot_utils import (
    setup_style, prepare_curve, plot_with_fill,
    add_stats_box, style_axes, save_figure, COLORS
)

# ============================================================
# 配置 (与原脚本一致)
# ============================================================
DATA_DIR = Path(__file__).parent / "experiment_data" / "wall_penalty_comparison"
OUTPUT_DIR = DATA_DIR / "summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PENALTY_TYPES = ['00p5', '0p5', '5']
PENALTY_LABELS = {'00p5': 'Penalty=0.05', '0p5': 'Penalty=0.5', '5': 'Penalty=5.0'}
PENALTY_VALUES = {'00p5': 0.05, '0p5': 0.5, '5': 5.0}
PENALTY_COLORS = {
    '00p5': COLORS['blue'],
    '0p5':  COLORS['orange'],
    '5':     COLORS['green'],
}

BALL_SOURCES = ['collect_ball_A', 'collect_ball_B']
PLAYERS = [0, 1, 2, 3]


# ============================================================
# 数据加载 (与原脚本一致: concat → 按 episode 聚合 → /num_files)
# 额外: 按单文件处理得 SEM
# ============================================================

def discover_files():
    files = {p: [] for p in PENALTY_TYPES}
    for f in DATA_DIR.glob("*.csv"):
        fn = f.name
        if "00p5penalty" in fn:
            files['00p5'].append(str(f))
        elif "0p5penalty" not in fn and "5penalty" in fn:
            files['5'].append(str(f))
        elif "0p5penalty" in fn:
            files['0p5'].append(str(f))
    for p in files:
        files[p].sort()
    return files


def process_one_file(fp):
    """处理单个 CSV — 返回 {episode: {player_ball, player_walls}}"""
    df = pd.read_csv(fp)
    max_ep = int(df['episode_id'].max())
    result = {}
    for ep in range(1, max_ep + 1):
        ep_df = df[df['episode_id'] == ep]
        row = {}
        for pid in PLAYERS:
            p_df = ep_df[ep_df['player_id'] == pid]
            ball = p_df[p_df['source'].isin(BALL_SOURCES)]['value'].sum()
            walls = len(p_df[p_df['source'] == 'wall_collision'])
            row[f'p{pid}_ball'] = ball
            row[f'p{pid}_walls'] = walls
        result[ep] = row
    return result, max_ep


def aggregate_penalty(files):
    """聚合一个惩罚条件下的 5 次运行 → per-episode mean±SEM"""
    all_runs = []
    max_eps_list = []
    for fp in files:
        r, me = process_one_file(fp)
        all_runs.append(r)
        max_eps_list.append(me)

    if len(set(max_eps_list)) > 1:
        print(f"  WARNING: inconsistent episode counts {max_eps_list}")
        return None

    max_ep = max_eps_list[0]
    episodes = np.arange(1, max_ep + 1)

    def collect(key):
        mat = np.zeros((len(files), max_ep))
        for i, run in enumerate(all_runs):
            for ep in range(1, max_ep + 1):
                mat[i, ep - 1] = run[ep][key]
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(len(files)) if len(files) > 1 else np.zeros_like(mean)
        return mean, sem

    data = {'episodes': episodes}
    for pid in PLAYERS:
        bm, bs = collect(f'p{pid}_ball')
        wm, ws = collect(f'p{pid}_walls')
        data[f'p{pid}_ball_mean'] = bm
        data[f'p{pid}_ball_sem'] = bs
        data[f'p{pid}_walls_mean'] = wm
        data[f'p{pid}_walls_sem'] = ws

    # Total
    total_ball_mean = np.zeros(max_ep)
    total_ball_sem = np.zeros(max_ep)
    total_walls_mean = np.zeros(max_ep)
    total_walls_sem = np.zeros(max_ep)
    for pid in PLAYERS:
        total_ball_mean += data[f'p{pid}_ball_mean']
        total_ball_sem += data[f'p{pid}_ball_sem'] ** 2
        total_walls_mean += data[f'p{pid}_walls_mean']
        total_walls_sem += data[f'p{pid}_walls_sem'] ** 2
    data['total_ball_mean'] = total_ball_mean
    data['total_ball_sem'] = np.sqrt(total_ball_sem)
    data['total_walls_mean'] = total_walls_mean
    data['total_walls_sem'] = np.sqrt(total_walls_sem)

    return data


def load_all_data():
    files = discover_files()
    print(f"[DISCOVER] 找到 {sum(len(v) for v in files.values())} 个文件")
    data = {}
    for pt in PENALTY_TYPES:
        print(f"  {PENALTY_LABELS[pt]}: {len(files[pt])} files")
        data[pt] = aggregate_penalty(files[pt])
        if data[pt]:
            print(f"    Episodes: {len(data[pt]['episodes'])}")
    return data


# ============================================================
# 图 1: 各玩家吃球得分
# ============================================================

def plot_player_ball_score(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(PLAYERS):
        ax = axes[idx]
        for pt in PENALTY_TYPES:
            if pt not in data:
                continue
            d = data[pt]
            x_s, y_s, y_l, y_u = prepare_curve(
                d['episodes'],
                d[f'p{pid}_ball_mean'],
                d[f'p{pid}_ball_sem'],
                smooth_window=5, error_band=True
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=PENALTY_COLORS[pt],
                           label=PENALTY_LABELS[pt])

        style_axes(ax,
                   xlabel='Episode', ylabel='Ball Collection Score',
                   title=f'Player {pid}',
                   legend_kwargs={'loc': 'upper left', 'fontsize': 8})

    plt.tight_layout()
    save_figure(fig, str(OUTPUT_DIR / "player_ball_score_comparison.png"))
    return fig


# ============================================================
# 图 2: 全场平均吃球得分
# ============================================================

def plot_total_ball_score(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    for pt in PENALTY_TYPES:
        if pt not in data:
            continue
        d = data[pt]
        x_s, y_s, y_l, y_u = prepare_curve(
            d['episodes'],
            d['total_ball_mean'],
            d['total_ball_sem'],
            smooth_window=5, error_band=True
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=PENALTY_COLORS[pt],
                       label=PENALTY_LABELS[pt])

    style_axes(ax,
               xlabel='Episode', ylabel='Total Ball Collection Score',
               title='Total Ball Collection Score vs Episode and Penalty',
               legend_kwargs={'loc': 'upper left', 'fontsize': 10, 'title': 'Wall Penalty'})

    stats_text = "Mean Ball Score:\n"
    for pt in PENALTY_TYPES:
        if pt in data:
            stats_text += f"{PENALTY_LABELS[pt]}: {np.nanmean(data[pt]['total_ball_mean']):.1f}\n"
    add_stats_box(ax, stats_text, loc='lower right', fontsize=9)

    save_figure(fig, str(OUTPUT_DIR / "total_ball_score_comparison.png"))
    return fig


# ============================================================
# 图 3: 各玩家撞墙次数
# ============================================================

def plot_player_wall_count(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(PLAYERS):
        ax = axes[idx]
        for pt in PENALTY_TYPES:
            if pt not in data:
                continue
            d = data[pt]
            x_s, y_s, y_l, y_u = prepare_curve(
                d['episodes'],
                d[f'p{pid}_walls_mean'],
                d[f'p{pid}_walls_sem'],
                smooth_window=5, error_band=True
            )
            plot_with_fill(ax, x_s, y_s, y_l, y_u,
                           color=PENALTY_COLORS[pt],
                           label=PENALTY_LABELS[pt])

        style_axes(ax,
                   xlabel='Episode', ylabel='Wall Collision Count',
                   title=f'Player {pid}',
                   legend_kwargs={'loc': 'upper right', 'fontsize': 8})

    plt.tight_layout()
    save_figure(fig, str(OUTPUT_DIR / "player_wall_count_comparison.png"))
    return fig


# ============================================================
# 图 4: 全场平均撞墙次数
# ============================================================

def plot_total_wall_count(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    for pt in PENALTY_TYPES:
        if pt not in data:
            continue
        d = data[pt]
        x_s, y_s, y_l, y_u = prepare_curve(
            d['episodes'],
            d['total_walls_mean'],
            d['total_walls_sem'],
            smooth_window=5, error_band=True
        )
        plot_with_fill(ax, x_s, y_s, y_l, y_u,
                       color=PENALTY_COLORS[pt],
                       label=PENALTY_LABELS[pt])

    style_axes(ax,
               xlabel='Episode', ylabel='Total Wall Collision Count',
               title='Total Wall Collision Count vs Episode and Penalty',
               legend_kwargs={'loc': 'upper right', 'fontsize': 10, 'title': 'Wall Penalty'})

    stats_text = "Mean Wall Collisions:\n"
    for pt in PENALTY_TYPES:
        if pt in data:
            stats_text += f"{PENALTY_LABELS[pt]}: {np.nanmean(data[pt]['total_walls_mean']):.1f}\n"
    add_stats_box(ax, stats_text, loc='upper left', fontsize=9)

    save_figure(fig, str(OUTPUT_DIR / "total_wall_count_comparison.png"))
    return fig


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Wall Penalty — Publication Style Redraw")
    print("=" * 60)

    # 1. 设置论文风格
    setup_style(context='paper', font_scale=1.3)

    # 2. 加载数据
    data = load_all_data()

    # 3. 生成 4 张图
    print("\nGenerating figures ...")
    plot_player_ball_score(data)    # 图 1
    plot_total_ball_score(data)     # 图 2
    plot_player_wall_count(data)    # 图 3
    plot_total_wall_count(data)     # 图 4

    print(f"\nDone. Results in {OUTPUT_DIR}")