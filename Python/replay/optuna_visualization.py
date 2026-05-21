"""
Optuna 超参搜索结果可视化
==========================
读取 custom_ppo.py 训练产生的 SQLite 数据库，
生成多维度可视化图表，帮助分析超参搜索结果。

图表包括:
  1. 优化历史 (Trial value 随 trial number 变化)
  2. 参数重要性排名
  3. 参数与目标值关系 (平行坐标图 / 散点图)
  4. 超参分布 (每个参数的采样分布)
  5. 最佳 trial 参数汇总表
"""

import optuna
import optuna.visualization as vis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pathlib
import argparse
import json


# ╔══════════════════════════════════════════════════════════╗
# ║                    默认配置                               ║
# ╚══════════════════════════════════════════════════════════╝
DEFAULT_DB_PATH = str(
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "logs" / "optuna" / "custom_ppo.db"
)
DEFAULT_STUDY_NAME = "custom_ppo_optuna"
DEFAULT_OUTPUT_DIR = str(
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "logs" / "optuna" / "figures"
)


def load_study(db_path: str, study_name: str) -> optuna.Study:
    """从 SQLite 数据库加载 Optuna study。"""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"[OK] 已加载 study: {study.study_name}  "
          f"(trials={len(study.trials)}, direction={study.direction})")
    return study


def print_best_trial(study: optuna.Study) -> None:
    """打印最佳 trial 的参数汇总。"""
    trial = study.best_trial
    print("\n" + "=" * 60)
    print(f"  最佳 Trial #{trial.number}  |  value = {trial.value:.6f}")
    print("=" * 60)
    for key, val in trial.params.items():
        print(f"  {key:30s} = {val}")
    print("=" * 60 + "\n")


def plot_optimization_history(study: optuna.Study, output_dir: str) -> go.Figure:
    """优化历史：目标值随 trial number 变化。"""
    fig = vis.plot_optimization_history(study)
    fig.update_layout(
        title=dict(text="Optimization History", font=dict(size=18)),
        height=500,
    )
    path = pathlib.Path(output_dir) / "optimization_history.html"
    pio.write_html(fig, file=str(path))
    print(f"[saved] {path}")
    return fig


def plot_param_importances(study: optuna.Study, output_dir: str) -> go.Figure:
    """参数重要性排名。"""
    try:
        fig = vis.plot_param_importances(study)
        fig.update_layout(
            title=dict(text="Hyperparameter Importances", font=dict(size=18)),
            height=500,
        )
        path = pathlib.Path(output_dir) / "param_importances.html"
        pio.write_html(fig, file=str(path))
        print(f"[saved] {path}")
        return fig
    except (ValueError, ImportError) as e:
        print(f"[skip] 参数重要性图无法生成 (可能需要 scikit-learn): {e}")
        return None


def plot_parallel_coordinate(study: optuna.Study, output_dir: str) -> go.Figure:
    """平行坐标图：同时展示所有参数与目标值的关系。"""
    fig = vis.plot_parallel_coordinate(study)
    fig.update_layout(
        title=dict(text="Parallel Coordinate", font=dict(size=18)),
        height=600,
    )
    path = pathlib.Path(output_dir) / "parallel_coordinate.html"
    pio.write_html(fig, file=str(path))
    print(f"[saved] {path}")
    return fig


def plot_contour(study: optuna.Study, output_dir: str) -> go.Figure:
    """等高线图：两个参数组合与目标值的关系。"""
    try:
        fig = vis.plot_contour(study)
        fig.update_layout(
            title=dict(text="Parameter Contour", font=dict(size=18)),
            height=600,
        )
        path = pathlib.Path(output_dir) / "param_contour.html"
        pio.write_html(fig, file=str(path))
        print(f"[saved] {path}")
        return fig
    except ValueError as e:
        print(f"[skip] 等高线图无法生成 (需要足够 trial): {e}")
        return None


def plot_slice(study: optuna.Study, output_dir: str) -> go.Figure:
    """切片图：每个参数单独与目标值的关系。"""
    fig = vis.plot_slice(study)
    fig.update_layout(
        title=dict(text="Parameter Slice", font=dict(size=18)),
        height=max(400, 250 * min(len(study.trials), 10)),
    )
    path = pathlib.Path(output_dir) / "param_slice.html"
    pio.write_html(fig, file=str(path))
    print(f"[saved] {path}")
    return fig


def plot_edf(study: optuna.Study, output_dir: str) -> go.Figure:
    """经验分布函数：目标值的累积分布。"""
    fig = vis.plot_edf(study)
    fig.update_layout(
        title=dict(text="Empirical Distribution Function", font=dict(size=18)),
        height=500,
    )
    path = pathlib.Path(output_dir) / "edf.html"
    pio.write_html(fig, file=str(path))
    print(f"[saved] {path}")
    return fig


def plot_timeline(study: optuna.Study, output_dir: str) -> go.Figure:
    """时间线图：展示每个 trial 的状态和耗时。"""
    try:
        fig = vis.plot_timeline(study)
        fig.update_layout(
            title=dict(text="Trial Timeline", font=dict(size=18)),
            height=400,
        )
        path = pathlib.Path(output_dir) / "timeline.html"
        pio.write_html(fig, file=str(path))
        print(f"[saved] {path}")
        return fig
    except Exception as e:
        print(f"[skip] 时间线图无法生成: {e}")
        return None


def generate_all(study: optuna.Study, output_dir: str) -> None:
    """生成所有可视化图表。"""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- 正在生成可视化图表 ---")
    plot_optimization_history(study, output_dir)
    plot_param_importances(study, output_dir)
    plot_parallel_coordinate(study, output_dir)
    plot_contour(study, output_dir)
    plot_slice(study, output_dir)
    plot_edf(study, output_dir)
    plot_timeline(study, output_dir)

    print(f"\n[done] 所有图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Optuna 结果可视化工具")
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH,
        help=f"Optuna SQLite 数据库路径 (默认: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--study-name", type=str, default=DEFAULT_STUDY_NAME,
        help=f"Study 名称 (默认: {DEFAULT_STUDY_NAME})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"图表输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    study = load_study(args.db, args.study_name)
    print_best_trial(study)
    generate_all(study, args.output_dir)


if __name__ == "__main__":
    main()
