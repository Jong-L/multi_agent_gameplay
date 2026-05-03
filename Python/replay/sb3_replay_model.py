"""
SB3 模型回放脚本 — 加载训练好的模型并可视化运行
=============================================

配置方式: 直接修改下方 Config 数据类的默认值后运行
  python Python/sb3_replay_model.py
"""
import pathlib
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


@dataclass
class Config:
    """模型回放配置 — 修改默认值即可调整参数。"""

    model_path: str = r"savedmodels\wall-distance-penalty-model.zip"
    """要加载的模型文件路径 (.zip)。"""
    env_path: Optional[str] = None
    """Godot 可执行文件路径 (None 连接编辑器)。"""
    speedup: int = 1
    """物理引擎加速倍数 (1=正常速度)。"""
    viz: bool = True
    """显示游戏窗口。"""
    reward_norm: bool = True
    """启用 VecNormalize 奖励归一化 (需与训练配置一致)。"""
    obs_norm: bool = False
    """同时归一化观测值。"""


def main():
    args = Config()

    model_path = pathlib.Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"加载模型: {model_path}")

    print("初始化 Godot 环境...")
    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        show_window=args.viz,
        speedup=args.speedup,
        n_parallel=1,
    )
    env = VecMonitor(env)

    if args.reward_norm:
        env = VecNormalize(
            env,
            norm_obs=args.obs_norm,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
        if vecnorm_path.exists():
            print(f"加载 VecNormalize 统计: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
        else:
            print(f"警告: VecNormalize 统计文件不存在 {vecnorm_path}")
            print("使用全新的归一化统计 (可能影响效果)")

    print("加载 PPO 模型...")
    model = PPO.load(model_path, env=env)

    print("开始回放... 按 Ctrl+C 停止。")
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
    except KeyboardInterrupt:
        print("\n回放被用户中断。")
    finally:
        env.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
