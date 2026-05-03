"""
DQN 模型回放脚本 — 加载训练好的 .pt 模型并可视化运行
===================================================

配置方式: 直接修改下方 Config 数据类的默认值后运行
  python Python/dqn_replay_model.py
"""
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# 复用 DQN 训练脚本中的组件
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from training.clean_rl_dqn import (
    DQNEnvWrapper, QNetwork, ObsSegmentDims, parse_godot_tres, layer_init,
)

@dataclass
class Config:
    """DQN 模型回放配置 — 修改默认值即可调整参数。"""

    model_path: str = "savedmodels/dqn_model.pt"
    """要加载的模型文件路径 (.pt)。"""
    env_path: Optional[str] = None
    """Godot 可执行文件路径 (None 连接编辑器)。"""
    speedup: int = 1
    """物理引擎加速倍数 (1=正常速度)。"""
    show_window: bool = True
    """显示游戏窗口。"""
    cuda: bool = True
    """是否启用 CUDA 加速 (回放推荐 CPU)。"""


def main():
    args = Config()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model_path = pathlib.Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"加载模型: {model_path}")
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    state_dict = checkpoint["q_network_state_dict"]
    train_args = checkpoint.get("args", {})

    # 从训练时的 config_path 读取 Godot 配置, 重建段维度
    config_path = train_args.get("config_path", "godot-game/configs/game_config.tres")
    seg = ObsSegmentDims.from_config(config_path)
    print(f"[Obs] segments: self={seg.self_dim} player={seg.player_dim} "
          f"ball={seg.ball_dim} enemy={seg.enemy_dim} map={seg.map_dim}")

    # 初始化环境 — 必须用 DQNEnvWrapper 处理 MultiDiscrete 动作
    print("初始化 Godot 环境...")
    envs = DQNEnvWrapper(
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
        seed=0,
        n_parallel=1,
    )

    # 用训练时的段维度重建网络
    q_network = QNetwork(envs, seg).to(device)
    q_network.load_state_dict(state_dict)
    q_network.eval()

    print("开始回放... 按 Ctrl+C 停止。")
    next_obs, _ = envs.reset()
    next_obs = np.array(next_obs, dtype=np.float32)

    try:
        while True:
            with torch.no_grad():
                obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
                q_values = q_network(obs_t.unsqueeze(0))
                actions = [int(q_values.argmax(dim=1).item())]

            next_obs, rewards, terms, truncs, infos = envs.step(
                np.array(actions)
            )
            next_obs = np.array(next_obs, dtype=np.float32)
    except KeyboardInterrupt:
        print("\n回放被用户中断。")
    finally:
        envs.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
