"""
单智能体 DQN (Deep Q-Network) 算法 — 基于 CleanRL 风格实现
===========================================================
参考: Mnih et al. "Human-level control through deep reinforcement learning" (Nature, 2015)
CleanRL 风格参考: https://docs.cleanrl.dev/rl-algorithms/dqn/

本文件演示了如何使用 godot_rl_agents 的 CleanRLGodotEnv 封装，
手写一个标准的 DQN 算法来训练 Godot 游戏中的智能体。

与 CleanRL PPO/PQN 示例的核心差异:
  - 使用 Experience Replay Buffer (经验回放缓冲)
  - 使用 Target Network (目标网络，周期性软/硬更新)
  - ε-greedy 探索策略 (线性衰减)
  - TD(0) 目标: r + γ * max_a' Q_target(s', a') * (1 - done)

适用场景:
  - 离散动作空间 (本项目的 6 个离散动作: 上下左右/攻击/待机)
  - 单智能体训练 (多智能体对抗验证需在 Godot 端配置随机策略对手)

用法: 直接修改下方 Args 数据类的默认值后运行
  python Python/clean_rl_dqn_example.py
"""

import os
import pathlib
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    ObsSegmentDims,
    RewardNormalizer,
    init_training_setup,
    layer_init,
    save_pt_model,
)

#  训练配置
@dataclass
class Args:
    """DQN 训练配置 (dataclass 风格，兼容 CleanRL 惯例)"""

    #环境
    env_path: Optional[str] = None
    # env_path: Optional[str] = "godot-game\\build\\game.exe"
    """Godot 导出可执行文件路径。为 None 时连接 Godot 编辑器。"""
    config_path: str = "godot-game/configs/game_config.tres"
    """game_config.tres 路径, 用于读取 ray_count 和 use_observation_valid_mask 等动态参数"""
    n_parallel: int = 1
    """并行 Godot 进程数量"""
    seed: int = 1
    """随机种子"""
    show_window: bool = False
    """是否显示游戏窗口"""
    speedup: int = 8
    """物理引擎加速倍数"""

    #记录
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验名称 """
    experiment_dir: str = "logs/cleanrl_dqn"
    """TensorBoard 日志目录"""
    save_model_path: Optional[str] = f"saved_models/cleanrl_dqn_{time.strftime('%m%d-%H%M')}"
    """模型保存路径，不用后缀"""
    onnx_export_path: Optional[str] = None
    """导出 ONNX 模型的路径。"""
    track: bool = False
    """是否用 Weights & Biases 追踪实验。"""
    wandb_project_name: str = "cleanRL"
    """Weights & Biases 项目名称"""
    wandb_entity: Optional[str] = None
    
    capture_video: bool = False
    """是否录制视频"""

    #训练
    total_timesteps: int = 100_0000
    """总训练步数 (env steps)。"""
    learning_rate: float = 2.5e-4
    """Adam 优化器学习率。"""
    buffer_size: int = 100_000
    """经验回放缓冲区容量。"""
    gamma: float = 0.99
    """折扣因子 γ。"""
    tau: float = 1.0
    """目标网络软更新系数 (1.0 = 硬更新,参数完全被更新为主网络的参数)。"""
    target_network_frequency: int = 500
    """每隔多少步更新一次目标网络。"""
    batch_size: int = 128
    """从 replay buffer 采样的批量大小"""
    start_e: float = 1.0
    """ε-greedy 起始探索率。"""
    end_e: float = 0.05
    """ε-greedy 最终探索率。"""
    exploration_fraction: float = 0.5
    """从 start_e 衰减到 end_e 所用的步数占总步数的比例。"""
    learning_starts: int = 10_000
    """缓冲区至少需要积累多少条经验后才开始学习。"""
    train_frequency: int = 10
    """每收集多少步经验执行一次梯度更新。"""
    max_grad_norm: float = 10.0
    """梯度裁剪最大范数。"""
    torch_deterministic: bool = True
    """是否启用 PyTorch 确定性模式。"""
    cuda: bool = True
    """是否启用 CUDA 加速。"""
    reward_norm: bool = True
    """是否对奖励做 running z-score 归一化。"""
    reward_clip: float = 10.0
    """奖励归一化裁剪范围 (仅在 reward_norm=True 时生效)。"""


class QNetwork(nn.Module):
    """DQN Q 值网络, 对观测各段分别提取特征后融合。

    输入: 观测向量 (obs_dim,) — Godot 端已按固定顺序拼接:
          [self_state | nearby_players | nearby_balls | nearby_enemies | map_state]
    输出: 每个动作的 Q 值 (n_actions,)
    """
    def __init__(self, envs: GodotDiscreteEnvWrapper, seg: ObsSegmentDims):
        super().__init__()
        n_actions = int(envs.single_action_space.n)

        self.seg_self = seg.self_dim
        self.seg_player = seg.player_dim
        self.seg_ball = seg.ball_dim
        self.seg_enemy = seg.enemy_dim
        self.seg_map = seg.map_dim

        # 各段独立特征提取
        self.self_net = nn.Sequential(layer_init(nn.Linear(self.seg_self, 16)), nn.ReLU())
        self.player_net = nn.Sequential(layer_init(nn.Linear(self.seg_player, 64)), nn.ReLU())
        self.ball_net = nn.Sequential(layer_init(nn.Linear(self.seg_ball, 64)), nn.ReLU())
        self.enemy_net = nn.Sequential(layer_init(nn.Linear(self.seg_enemy, 32)), nn.ReLU())
        self.map_net = nn.Sequential(layer_init(nn.Linear(self.seg_map, 32)), nn.ReLU())

        self.output = layer_init(nn.Linear(16 + 64 + 64 + 32 + 32, n_actions), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        i = 0
        s = obs[:, i : i + self.seg_self];    i += self.seg_self
        p = obs[:, i : i + self.seg_player];  i += self.seg_player
        b = obs[:, i : i + self.seg_ball];    i += self.seg_ball
        e = obs[:, i : i + self.seg_enemy];   i += self.seg_enemy
        m = obs[:, i : i + self.seg_map]
        fused = torch.cat([
            self.self_net(s), self.player_net(p), self.ball_net(b),
            self.enemy_net(e), self.map_net(m),
        ], dim=1)

        return self.output(torch.relu(fused))

class ReplayBuffer:
    """经验回放缓冲区。
    存储 (state, action, reward, next_state, done) 五元组。
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """存入一条经验。"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device):
        """随机采样一个 batch 并转换为 PyTorch 张量。"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []

        for idx in indices:
            o, a, r, no, d = self.buffer[idx]
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            nobs_list.append(no)
            done_list.append(d)

        return (
            torch.tensor(np.array(obs_list), dtype=torch.float32).to(device),
            torch.tensor(np.array(act_list), dtype=torch.int64).to(device),
            torch.tensor(np.array(rew_list), dtype=torch.float32).to(device),
            torch.tensor(np.array(nobs_list), dtype=torch.float32).to(device),
            torch.tensor(np.array(done_list), dtype=torch.float32).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    """线性衰减调度器。在duration的时间内从 start 线性衰减到 end，之后保持不变。"""
    slope = (end - start) / duration
    return max(slope * t + start, end)

def select_actions_epsilon_greedy(
    q_network: QNetwork,
    obs_array: np.ndarray,
    epsilon: float,
    action_space: gym.spaces.Discrete,
    device: torch.device,
) -> list:
    """ε-greedy 动作选择 (多并行环境)。

    Args:
        q_network:    Q 值网络
        obs_array:    当前观测数组, 形状 (num_envs, obs_dim)
        epsilon:      探索率
        action_space: 离散动作空间
        device:       计算设备

    Returns:
        list[int]: 每个环境的动作
    """
    num_envs = obs_array.shape[0]
    actions = []

    for i in range(num_envs):
        if random.random() < epsilon:
            actions.append(action_space.sample())
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs_array[i], dtype=torch.float32).to(device)
                q_values = q_network(obs_t.unsqueeze(0))
                actions.append(int(q_values.argmax(dim=1).item()))

    return actions


def train_dqn_step(
    q_network: QNetwork,
    target_network: QNetwork,
    rb: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float]:
    """执行一次 DQN 梯度更新。

    从经验回放缓冲区采样一个 batch, 计算 TD 目标,
    使用 Huber loss 更新 Q 网络。

    Returns:
        (loss, current_q_mean): Huber loss 值, 当前 Q 值均值
    """
    # 采样
    obses, acts, rews, nobses, dones_batch = rb.sample(batch_size, device)

    # TD 目标
    with torch.no_grad():
        target_max, _ = target_network(nobses).max(dim=1)
        td_target = rews + gamma * target_max * (1.0 - dones_batch)

    # 当前 Q(s, a)
    current_q = q_network(obses).gather(1, acts.unsqueeze(1)).squeeze()

    # Huber loss
    loss = F.smooth_l1_loss(current_q, td_target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item(), current_q.mean().item()


def update_target_network(
    target_network: QNetwork,
    q_network: QNetwork,
    tau: float,
) -> None:
    """目标网络软更新: θ_target ← τ·θ + (1-τ)·θ_target。

    Args:
        target_network: 目标网络
        q_network:      在线 Q 网络
        tau:            更新系数 (1.0 = 硬更新)
    """
    for tp, p in zip(target_network.parameters(), q_network.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def log_dqn(
    writer,
    global_step: int,
    td_loss: float,
    q_mean: float,
    epsilon: float,
    episode_returns: deque,
    start_time: float,
) -> None:
    """将 DQN 训练指标写入 TensorBoard。

    Args:
        writer:          TensorBoard SummaryWriter
        global_step:     全局步数
        td_loss:         当前 TD 损失
        q_mean:          当前 Q 值均值
        epsilon:         探索率
        episode_returns: 近 100 回合的奖励列表
        start_time:      训练开始时间
    """
    writer.add_scalar("losses/td_loss", td_loss, global_step)
    writer.add_scalar("losses/q_values", q_mean, global_step)
    writer.add_scalar(
        "charts/SPS",
        int(global_step / (time.time() - start_time)),
        global_step,
    )
    writer.add_scalar("charts/epsilon", epsilon, global_step)

    if len(episode_returns) > 0:
        mean_ret = np.mean(np.array(episode_returns))
        writer.add_scalar("charts/episodic_return", mean_ret, global_step)


def export_dqn_onnx(
    q_network: QNetwork,
    onnx_path: str,
    obs_shape: tuple,
) -> None:
    """将 DQN Q 网络导出为 ONNX 模型。

    Args:
        q_network: 训练好的 QNetwork
        onnx_path: 导出路径 (不加后缀, 自动追加 .onnx)
        obs_shape: 观测空间形状, 如 (142,)
    """
    path_onnx = pathlib.Path(onnx_path).with_suffix(".onnx")
    print(f"[Export] Exporting ONNX to {path_onnx}")

    q_network.eval().to("cpu")
    dummy_input = torch.randn(1, int(np.prod(obs_shape)))

    torch.onnx.export(
        q_network,
        dummy_input,
        str(path_onnx),
        opset_version=15,
        input_names=["obs"],
        output_names=["q_values"],
        dynamic_axes={"obs": {0: "batch_size"}, "q_values": {0: "batch_size"}},
    )
    print(f"[Export] Done: {path_onnx}")


#  主训练循环

def train(
    args: Args,
    q_network: QNetwork,
    target_network: QNetwork,
    envs: GodotDiscreteEnvWrapper,
    optimizer: optim.Optimizer,
    rb: ReplayBuffer,
    device: torch.device,
    writer,
    reward_normalizer: Optional[RewardNormalizer],
    next_obs_array: np.ndarray,
) -> None:
    """DQN 主训练循环。"""
    global_step = 0
    start_time = time.time()
    episode_returns = deque(maxlen=100)
    accum_rewards = np.zeros(envs.num_envs)

    while global_step < args.total_timesteps:
        global_step += envs.num_envs

        # epsilon-greedy 动作选择
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        actions = select_actions_epsilon_greedy(
            q_network, next_obs_array, epsilon,
            envs.single_action_space, device,
        )

        # 环境步进
        next_obs, rewards, terminations, truncations, infos = envs.step(
            np.array(actions, dtype=np.int64)
        )
        dones = np.logical_or(terminations, truncations)

        # 奖励归一化 + 存入回放缓冲区
        raw_rewards = np.asarray(rewards, dtype=np.float32)
        if reward_normalizer is not None:
            reward_normalizer.update_array(raw_rewards)
            norm_rewards = reward_normalizer.normalize_array(raw_rewards)
        else:
            norm_rewards = raw_rewards

        for i in range(envs.num_envs):
            rb.push(
                next_obs_array[i].copy(),
                actions[i],
                float(norm_rewards[i]),
                np.array(next_obs[i], dtype=np.float32),
                bool(dones[i]),
            )

        next_obs_array = np.array(next_obs, dtype=np.float32)

        # 回合奖励追踪 (使用原始奖励)
        accum_rewards += raw_rewards
        for i, d in enumerate(dones):
            if d:
                episode_returns.append(accum_rewards[i])
                accum_rewards[i] = 0.0

        # DQN 梯度更新
        if len(rb) >= args.learning_starts and global_step % args.train_frequency == 0:
            td_loss, q_mean = train_dqn_step(
                q_network, target_network, rb, optimizer,
                args.batch_size, args.gamma, args.max_grad_norm, device,
            )

            # TensorBoard 日志 (每 100 步)
            if global_step % 100 == 0:
                log_dqn(
                    writer, global_step, td_loss, q_mean,
                    epsilon, episode_returns, start_time,
                )

        # 目标网络更新
        if global_step % args.target_network_frequency == 0:
            update_target_network(target_network, q_network, args.tau)

        # 终端打印
        if len(episode_returns) > 0 and global_step % 1000 == 0:
            mean_ret = np.mean(np.array(episode_returns))
            sps = int(global_step / (time.time() - start_time))
            print(
                f"[Step {global_step:8d}] "
                f"SPS: {sps:5d}  "
                f"mean_return: {mean_ret:8.2f}  "
                f"epsilon: {epsilon:.3f}"
            )


#  主训练入口

def main():
    # 共享初始化
    args = Args()
    writer, device, envs, seg, run_name = init_training_setup(args)

    # DQN 配置
    n_actions = int(envs.single_action_space.n)
    print(f"[Obs] segments: self={seg.self_dim} player={seg.player_dim} "
          f"ball={seg.ball_dim} enemy={seg.enemy_dim} map={seg.map_dim} total={seg.total}")

    # 网络，目标网络，缓冲区，优化器
    q_network = QNetwork(envs, seg).to(device)
    target_network = QNetwork(envs, seg).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size)

    # 奖励归一化器
    reward_normalizer = None
    if args.reward_norm:
        reward_normalizer = RewardNormalizer(clip=args.reward_clip)
        print(f"[RewardNorm] enabled, clip={args.reward_clip}")

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs_array = np.array(next_obs_array, dtype=np.float32)

    try:
        train(
            args, q_network, target_network, envs, optimizer, rb,
            device, writer, reward_normalizer, next_obs_array,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] 训练被手动中断 (Ctrl+C)")
        if args.save_model_path is not None:
            print(f"[Interrupt] 保存检查点到 {args.save_model_path} ...")
            save_dict = {"q_network_state_dict": q_network.state_dict()}
            save_pt_model(args.save_model_path, save_dict, args, reward_normalizer)
        return
    finally:
        envs.close()
        writer.close()

    # 正常训练结束后的保存与导出
    if args.save_model_path is not None:
        save_dict = {"q_network_state_dict": q_network.state_dict()}
        save_pt_model(args.save_model_path, save_dict, args, reward_normalizer)

    if args.onnx_export_path is not None:
        export_dqn_onnx(
            q_network, args.onnx_export_path,
            envs.single_observation_space.shape,
        )


if __name__ == "__main__":
    main()
