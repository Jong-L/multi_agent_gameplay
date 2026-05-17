"""
PPO 模型推理回放脚本
========================
加载 custom_ppo.py 训练产生的 .pt 模型文件，
连接到 Godot 环境进行可视化推理回放。

支持:
  - MLP / SegmentedMLP / GRU_MLP 三种网络架构
  - GRU_MLP 自动维护 RNN 隐藏态
  - 自动从 checkpoint 的 args 恢复观测维度配置
"""
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# 保证 training 目录下的模块可导入
_TRAINING_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "training")
sys.path.insert(0, _TRAINING_DIR)

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    ObsSegmentDims,
    layer_init,
)
from custom_ppo import (
    NetworkType,
    PPOAgent,
    SegmentedObsHelper,
    as_hidden_tuple,
    make_mlp,
)

# ╔══════════════════════════════════════════════════════════╗
# ║                    配  置                                ║
# ╚══════════════════════════════════════════════════════════╝
@dataclass
class ReplayConfig:
    """PPO 模型回放配置 — 直接修改默认值即可。"""

    model_path: str = "saved_models/ppo_mlp.pt"
    """要加载的模型文件路径 (.pt)。"""

    env_path: Optional[str] = "curriculum_envs/s4-enemy-only/build/game.exe"
    """Godot 可执行文件路径 (None 连接编辑器)。"""

    config_path: str = "curriculum_envs/s4-enemy-only/configs/game_config.tres"
    """game_config.tres 路径, 用于读取观测维度配置。"""

    speedup: int = 2
    """物理引擎加速倍数 (1=正常速度)。"""

    show_window: bool = True
    """显示游戏窗口。"""

    deterministic: bool = True
    """确定性推理 (argmax 而非采样)。"""

    cuda: bool = False
    """回放使用 GPU (默认 CPU)。"""

    max_episodes: int = 0
    """最大回放 episode 数 (0=无限, 按 Ctrl+C 停止)。"""

    seed: int = 0
    """随机种子。"""


# ╔══════════════════════════════════════════════════════════╗
# ║                  Checkpoint 加载                          ║
# ╚══════════════════════════════════════════════════════════╝

def load_checkpoint(model_path: pathlib.Path, device: torch.device) -> dict:
    """加载 save_pt_model() 保存的 .pt 模型文件。
    格式: {"args": {...str values...}, "agent_state_dict": state_dict}
    """
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    result = dict(checkpoint.get("args", {}))
    result["agent_state_dict"] = checkpoint["agent_state_dict"]
    if "reward_normalizer" in checkpoint:
        result["reward_normalizer"] = checkpoint["reward_normalizer"]
    return result


def build_replay_args(checkpoint_data: dict):
    """从 checkpoint 数据重建 PPO 训练 Args。

    当前 save_pt_model 格式保证 args 中所有字段均为纯 Python 字面量。
    若 checkpoint 缺字段直接报错，不提供 fallback。
    """
    from custom_ppo import Args as TrainArgs

    # checkpoint 中保存的 args 即为完整 TrainArgs 字段
    # 过滤掉不属于 Args 的运行时字段（如 reward_normalizer）
    args_fields = {f.name for f in TrainArgs.__dataclass_fields__.values()}
    merged = {k: v for k, v in checkpoint_data.items() if k in args_fields}

    # network_type 保存为字符串，转为 NetworkType 枚举
    nt = merged.get("network_type", "segmented_mlp")
    if isinstance(nt, str):
        merged["network_type"] = NetworkType(nt)

    return TrainArgs(**merged)


# ╔══════════════════════════════════════════════════════════╗
# ║                    主回放逻辑                             ║
# ╚══════════════════════════════════════════════════════════╝
def main():
    args = ReplayConfig()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model_path = pathlib.Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # ── 1. 加载 checkpoint ──
    print(f"加载模型: {model_path}")
    checkpoint_data = load_checkpoint(model_path, device)
    state_dict = checkpoint_data.pop("agent_state_dict")

    # ── 2. 重建网络配置 ──
    replay_args = build_replay_args(checkpoint_data)

    # 从 config_path 重建观测段维度
    config_path = replay_args.config_path or args.config_path
    seg = ObsSegmentDims.from_config(config_path)
    print(f"[Obs] segments: self={seg.self_dim} player={seg.player_dim} "
          f"ball={seg.ball_dim} enemy={seg.enemy_dim} map={seg.map_dim}")

    # ── 3. 重建网络 ──
    n_actions = 6  # Godot 游戏离散动作空间固定为 6
    agent = PPOAgent(n_actions, seg, replay_args).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"[PPO] network_type={replay_args.network_type}, "
          f"params={agent.num_params():,}, recurrent={agent.is_recurrent}")

    # ── 4. 初始化环境 ──
    print("初始化 Godot 环境...")
    envs = GodotDiscreteEnvWrapper(
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
        seed=args.seed,
        n_parallel=1,
    )

    num_envs = envs.num_envs

    # ── 5. 初始化隐藏态 (GRU_MLP) ──
    rnn_state = None
    if agent.is_recurrent:
        rnn_state = agent.get_initial_state(num_envs, device)
        print(f"[GRU] hidden_state_size={agent.recurrent_state_size}")

    # ── 6. 推理回放循环 ──
    print("开始回放... 按 Ctrl+C 停止。")
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = np.array(next_obs, dtype=np.float32)

    episode_count = 0
    episode_reward = 0.0
    step_count = 0

    try:
        while True:
            with torch.no_grad():
                obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
                # PPOAgent.get_action_and_value 返回 (action, logprob, entropy, value[, next_state])
                result = agent.get_action_and_value(
                    obs_t,
                    rnn_state=rnn_state,
                    return_state=agent.is_recurrent,
                )
                action = result[0]
                if agent.is_recurrent:
                    rnn_state = result[4]

                if args.deterministic:
                    # 确定性: 取 logits 最大概率动作
                    features = agent.encoder(obs_t, rnn_state)[0] if agent.is_recurrent else agent.encoder(obs_t)[0]
                    logits = agent.actor(features)
                    action = logits.argmax(dim=1)
                # 否则 result[0] 已是采样动作

                actions = [int(a.item()) for a in action]

            next_obs, rewards, terms, truncs, infos = envs.step(
                np.array(actions, dtype=np.int64)
            )
            next_obs = np.array(next_obs, dtype=np.float32)

            # 统计奖励
            reward_val = np.asarray(rewards, dtype=np.float32)
            done = np.logical_or(
                np.asarray(terms, dtype=bool),
                np.asarray(truncs, dtype=bool),
            )
            episode_reward += float(reward_val.sum())
            step_count += 1

            if done.any():
                episode_count += 1
                per_env = " ".join(
                    f"env_{i}={float(reward_val[i]):+.1f}"
                    for i in range(num_envs)
                )
                print(
                    f"[Ep {episode_count:4d}] "
                    f"步数={step_count:6d}  "
                    f"总奖励={episode_reward:+.1f}  "
                    f"{per_env}"
                )
                step_count = 0
                episode_reward = 0.0

                if 0 < args.max_episodes <= episode_count:
                    print(f"[Done] 达到最大 episode 数 {args.max_episodes}, 结束回放。")
                    break

                # 新 episode 重置 GRU 隐藏态
                if agent.is_recurrent:
                    rnn_state = agent.get_initial_state(num_envs, device)

    except KeyboardInterrupt:
        print("\n回放被用户中断。")
    finally:
        envs.close()
        print("环境已关闭。")


if __name__ == "__main__":
    main()
