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
from token import OP
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

@dataclass
class Args:
    """DQN 训练配置 (dataclass 风格，兼容 CleanRL 惯例)"""

    # ---- 环境 ----
    env_path: Optional[str] = None
    # env_path: Optional[str] = "godot-game\\build\\game.exe"
    """Godot 导出可执行文件路径。为 None 时连接 Godot 编辑器。"""
    n_parallel: int = 1
    """并行 Godot 进程数量"""
    seed: int = 1
    """随机种子"""
    show_window: bool = False
    """是否显示游戏窗口"""
    speedup: int = 8
    """物理引擎加速倍数"""

    # ---- 记录----
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验名称 """
    experiment_dir: str = "logs/cleanrl_dqn"
    """TensorBoard 日志目录"""
    save_model_path: Optional[str] = None
    """训练结束后保存模型的路径 (.zip)"""
    onnx_export_path: Optional[str] = None
    """导出 ONNX 模型的路径。"""
    track: bool = False
    """是否用 Weights & Biases 追踪实验。"""
    wandb_project_name: str = "cleanRL"
    """Weights & Biases 项目名称"""
    wandb_entity: Optional[str] = None
    
    capture_video: bool = False
    """是否录制视频"""

    # ---- 训练 ----
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
    """从 replay buffer 采样的批量大小。"""
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

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """正交初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    """DQN Q 值网络。

    输入: 观测向量 (obs_dim,)
    输出: 每个动作的 Q 值 (n_actions,)
    """
    def __init__(self, envs: CleanRLGodotEnv):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = int(envs.single_action_space.n)

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回所有动作的 Q 值向量。"""
        return self.network(x)

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

def main():
    args = Args()

    # ---- 初始化 ----
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"{args.experiment_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---- 环境 ----
    envs = CleanRLGodotEnv(
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
        seed=args.seed,
        n_parallel=args.n_parallel,
    )
    assert isinstance(
        envs.single_action_space, __import__("gymnasium").spaces.Discrete
    ), "DQN 只支持离散动作空间"

    print(f"[Env] num_envs={envs.num_envs}")
    print(f"[Env] obs_shape={envs.single_observation_space.shape}")
    print(f"[Env] n_actions={envs.single_action_space.n}")

    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())  # 初始同步
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(args.buffer_size)

    #训练状态
    global_step = 0
    start_time = time.time()
    episode_returns = deque(maxlen=100)  # 最近 100 个 episode 的回报
    accum_rewards = np.zeros(envs.num_envs)  # 每回合所有ai_control节点的累计奖励

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs_array = np.array(next_obs_array, dtype=np.float32)

    # 主训练循环
    while global_step < args.total_timesteps:
        #收集一帧经验
        global_step += envs.num_envs

        # ε-greedy 动作选择
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )

        actions = []
        for i in range(envs.num_envs):# 遍历所有智能体
            if random.random() < epsilon:
                actions.append(envs.single_action_space.sample())
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(next_obs_array[i], dtype=torch.float32).to(device)#shape (obs_dim,)
                    q_values = q_network(obs_t.unsqueeze(0))#shape (1, obs_dim)
                    actions.append(int(q_values.argmax(dim=1).item()))

        # 执行动作
        next_obs, rewards, terminations, truncations, infos = envs.step(
            np.array(actions)
        )
        dones = np.logical_or(terminations, truncations)

        # 存入回放缓冲区
        for i in range(envs.num_envs):
            rb.push(
                next_obs_array[i].copy(),
                actions[i],
                float(rewards[i]),
                np.array(next_obs[i], dtype=np.float32),
                bool(dones[i]),
            )

        next_obs_array = np.array(next_obs, dtype=np.float32)

        # Episode 回报统计
        accum_rewards += np.array(rewards)
        for i, d in enumerate(dones):# 遍历所有智能体
            if d:
                episode_returns.append(accum_rewards[i])
                accum_rewards[i] = 0.0

        #训练步骤 
        if len(rb) >= args.learning_starts:
            if global_step % args.train_frequency == 0:
                # 采样一个 batch
                obses, acts, rews, nobses, dones_batch = rb.sample(
                    args.batch_size, device
                )

                with torch.no_grad():
                    # TD target
                    target_max, _ = target_network(nobses).max(dim=1)
                    td_target = rews + args.gamma * target_max * (1.0 - dones_batch)

                # 当前 Q(s, a)
                current_q = q_network(obses).gather(
                    1, acts.unsqueeze(1)#shape (batch_size, 1)
                ).squeeze()

                # Huber loss 
                loss = F.smooth_l1_loss(current_q, td_target)

                optimizer.zero_grad()# 清空梯度
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)# 梯度裁剪
                optimizer.step()

                # 日志
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar(
                        "losses/q_values", current_q.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

        #目标网络更新
        if global_step % args.target_network_frequency == 0:
            for tp, p in zip(
                target_network.parameters(), q_network.parameters()
            ):
                tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

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
            writer.add_scalar("charts/episodic_return", mean_ret, global_step)

    # 保存+清理
    envs.close()
    writer.close()

    # 保存模型
    if args.save_model_path is not None:
        save_path = pathlib.Path(args.save_model_path).with_suffix(".pt")
        torch.save(
            {
                "q_network_state_dict": q_network.state_dict(),
                "args": vars(args),
            },
            str(save_path),
        )
        print(f"[Save] Model saved to {save_path}")

    # 导出 ONNX (可选)
    if args.onnx_export_path is not None:
        path_onnx = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
        print(f"[Export] Exporting ONNX to {path_onnx}")
        q_network.eval().to("cpu")
        dummy_input = torch.randn(
            1, int(np.prod(envs.single_observation_space.shape))
        )
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


if __name__ == "__main__":
    main()
