# PPO算法实现 - 用于Godot强化学习环境
# 文档和实验结果: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# 这个文件实现了PPO (Proximal Policy Optimization) 算法，用于训练Godot游戏中的智能体
# 支持连续动作空间，使用CleanRL框架风格实现
#
# 配置方式: 直接修改下方 Args 数据类的默认值后运行
#   python Python/clean_rl_example.py
import os
import pathlib
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal  # 正态分布，用于连续动作采样
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录

from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv  # Godot环境包装器


@dataclass
class Args:
    """PPO 训练配置 — 修改默认值即可调整训练参数。"""

    # ---- 实验配置 ----
    experiment_dir: str = "logs/cleanrl"
    """TensorBoard 日志存储目录。"""
    experiment_name: str = os.path.basename(__file__).rstrip(".py")
    """实验名称，在 TensorBoard 中显示。"""
    seed: int = 0
    """随机种子，保证实验可重复性。"""
    torch_deterministic: bool = True
    """启用 PyTorch 确定性模式。"""
    cuda: bool = True
    """启用 CUDA 加速。"""

    # ---- 追踪 ----
    track: bool = False
    """使用 Weights & Biases 追踪实验。"""
    wandb_project_name: str = "cleanRL"
    """W&B 项目名称。"""
    wandb_entity: Optional[str] = None
    """W&B 团队/实体名称。"""
    capture_video: bool = False
    """录制智能体表现视频。"""
    onnx_export_path: Optional[str] = None
    """训练结束后导出 ONNX 模型的路径。"""

    # ---- 算法超参数 ----
    env_path: Optional[str] = None
    """Godot 环境可执行文件路径 (None 连接编辑器)。"""
    speedup: int = 8
    """Godot 环境加速倍数。"""
    total_timesteps: int = 1_000_000
    """总训练步数。"""
    learning_rate: float = 3e-4
    """优化器学习率。"""
    num_steps: int = 32
    """每次 rollout 在每个环境中运行的步数。"""
    anneal_lr: bool = True
    """对学习率进行线性退火。"""
    gamma: float = 0.99
    """折扣因子 γ。"""
    gae_lambda: float = 0.95
    """GAE 的 λ 参数，平衡偏差和方差。"""
    num_minibatches: int = 8
    """小批量数量。"""
    update_epochs: int = 10
    """每次更新遍历数据的轮数 (K epochs)。"""
    norm_adv: bool = True
    """对优势函数进行标准化。"""
    clip_coef: float = 0.2
    """PPO 裁剪系数 ε。"""
    clip_vloss: bool = True
    """对价值函数损失使用裁剪。"""
    ent_coef: float = 0.0001
    """熵系数，鼓励探索。"""
    vf_coef: float = 0.5
    """价值函数损失系数。"""
    max_grad_norm: float = 0.5
    """梯度裁剪最大范数。"""
    target_kl: Optional[float] = None
    """目标 KL 散度阈值，用于早停 (None 禁用)。"""
    n_parallel: int = 1
    """并行 Godot 环境实例数量。"""
    viz: bool = False
    """显示 Godot 游戏窗口。"""

    # 运行时计算的衍生值 (不在 dataclass __init__ 中设置)
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    神经网络层初始化函数
    使用正交初始化
    
    Args:
        layer: 要初始化的网络层
        std: 权重初始化的标准差，默认√2
        bias_const: 偏置初始化的常数值，默认0
    
    Returns:
        初始化后的网络层
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    PPO智能体网络架构
    包含两个核心网络：
    1. Actor（策略网络）：输出动作的概率分布（均值和对数标准差）
    2. Critic（价值网络）：估计状态价值V(s)
    
    网络结构：
    - 输入：观测空间维度
    - 隐藏层：2层全连接，每层64个神经元，Tanh激活
    - Actor输出：动作空间的均值向量 + 可学习的对数标准差
    - Critic输出：单个标量价值估计
    """
    def __init__(self, envs):
        super().__init__()
        
        # ==================== Critic 价值网络 ====================
        # 估计状态的价值函数 V(s)，用于计算优势函数
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # 输出层使用较小的std以稳定训练
        )
        
        # ==================== Actor 策略网络 ====================
        # 输出连续动作的均值，配合对数标准差形成正态分布
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),  # 输出层使用很小的std
        )
        
        # 可学习的对数标准差参数，对所有动作维度共享
        # 初始化为0，即标准差为exp(0)=1
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """
        获取状态价值估计
        
        Args:
            x: 观测状态张量
        Returns:
            状态价值 V(s)
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        根据观测状态获取动作和相关统计信息
        
        Args:
            x: 观测状态张量
            action: 可选的外部动作，如果为None则从策略中采样
        
        Returns:
            action: 采样的动作或给定动作
            log_prob: 动作的对数概率
            entropy: 策略的熵（衡量随机性）
            value: 状态价值估计
        """
        action_mean = self.actor_mean(x)  # 计算动作均值
        action_logstd = self.actor_logstd.expand_as(action_mean)  # 扩展对数标准差到匹配均值维度
        action_std = torch.exp(action_logstd)  # 转换为标准差
        probs = Normal(action_mean, action_std)  # 构建正态分布
        
        if action is None:
            action = probs.sample()  # 从分布中采样动作
        
        # 返回：动作、对数概率（求和）、熵（求和）、状态价值
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    # ==================== 第1步：初始化配置 ====================
    args = Args()
    run_name = f"{args.experiment_name}__{args.seed}__{int(time.time())}"  # 生成唯一运行名称
    
    # 可选：使用Weights & Biases进行实验追踪
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,  # 同步TensorBoard日志
            config=vars(args),  # 记录所有超参数
            name=run_name,
            save_code=True,  # 保存代码快照
        )
    
    # 初始化TensorBoard日志写入器
    writer = SummaryWriter(f"{args.experiment_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ==================== 第2步：设置随机种子 ====================
    # 保证实验的可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 选择计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ==================== 第3步：环境初始化 ====================
    envs = env = CleanRLGodotEnv(
        env_path=args.env_path, show_window=args.viz, speedup=args.speedup, seed=args.seed, n_parallel=args.n_parallel
    )
    args.num_envs = envs.num_envs  # 并行环境数量
    args.batch_size = int(args.num_envs * args.num_steps)  # 批次大小 = 环境数 × 步数
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 小批量大小
    
    # 初始化智能体和优化器
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ==================== 第4步：分配存储空间 ====================
    # 预分配张量用于存储rollout数据，提高训练效率
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ==================== 第5步：开始训练循环 ====================
    global_step = 0  # 全局步数计数器
    start_time = time.time()  # 记录开始时间
    next_obs, _ = envs.reset(seed=args.seed)  # 重置环境获取初始观测
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)  # 初始化done标志
    num_updates = args.total_timesteps // args.batch_size  # 计算总更新次数
    video_filenames = set()

    # 用于统计episode回报的队列（保留最近20个episode）
    episode_returns = deque(maxlen=20)
    accum_rewards = np.zeros(args.num_envs)  # 累积每个环境的奖励

    # ==================== 主训练循环 ====================
    for update in range(1, num_updates + 1):
        # 学习率退火：随训练进程线性降低学习率
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ==================== 数据采集阶段（Rollout）====================
        # 收集 num_steps 步的经验数据
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # 使用当前策略选择动作（不计算梯度）
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # 执行动作并观察结果
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)  # 合并终止条件
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # 累积奖励用于计算episode总回报
            accum_rewards += np.array(reward)

            # 当episode结束时，记录总回报并重置累积奖励
            for i, d in enumerate(done):
                if d:
                    episode_returns.append(accum_rewards[i])
                    accum_rewards[i] = 0

        # ==================== 第6步：计算优势函数和回报 ====================
        # 使用GAE（广义优势估计）计算优势函数
        with torch.no_grad():
            # 对最后一个状态进行bootstrap（如果未完成）
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            # 从后向前计算GAE
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # TD误差：δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # GAE公式：A_t = δ_t + γ*λ*A_{t+1}
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values  # 回报 = 优势 + 价值

        # ==================== 第7步：展平批数据 ====================
        # 将(num_steps, num_envs, ...)重塑为(batch_size, ...)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ==================== 第8步：PPO策略和价值网络优化 ====================
        b_inds = np.arange(args.batch_size)  # 创建索引数组
        clipfracs = []  # 记录裁剪比例
        
        # 多轮epoch更新（PPO的关键特性：多次复用同一批数据）
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # 打乱数据顺序
            
            # 小批量梯度下降
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # 获取当前小批量索引

                # 前向传播：获取新策略下的对数概率、熵和价值
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # 计算重要性采样比率：r(θ) = π_θ(a|s) / π_θ_old(a|s)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 计算KL散度近似值（用于监控）
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 获取小批量优势并标准化（如果启用）
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ==================== 策略损失（Policy Loss）====================
                # PPO裁剪目标：min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)
                pg_loss1 = -mb_advantages * ratio  # 未裁剪的损失
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # 裁剪后的损失
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 取两者最大值（保守更新）

                # ==================== 价值损失（Value Loss）====================
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # 裁剪的价值损失：防止价值函数剧烈变化
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 熵损失（鼓励探索）
                entropy_loss = entropy.mean()
                
                # 总损失 = 策略损失 - 熵系数×熵 + 价值系数×价值损失
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # 梯度裁剪
                optimizer.step()

            # KL散度早停：如果策略偏离太大则提前停止
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # ==================== 第9步：计算解释方差 ====================
        # 评估价值函数的拟合程度（1表示完美拟合，0表示不如预测均值）
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ==================== 第10步：记录训练指标到TensorBoard ====================
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # 打印和记录episode回报
        if len(episode_returns) > 0:
            sps = int(global_step / (time.time() - start_time))  # 每秒步数
            mean_return = np.mean(np.array(episode_returns))
            print("SPS:", sps, "Returns:", mean_return)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/episodic_return", mean_return, global_step)

    # ==================== 第11步：清理资源 ====================
    envs.close()
    writer.close()

    # ==================== 第12步：导出ONNX模型（可选）====================
    # 将训练好的策略网络导出为ONNX格式，便于部署到其他平台
    if args.onnx_export_path is not None:
        path_onnx = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
        print("Exporting onnx to: " + os.path.abspath(path_onnx))

        agent.eval().to("cpu")  # 切换到评估模式并转移到CPU

        # 定义简化的ONNX导出策略（只包含actor_mean，不包含随机采样）
        class OnnxPolicy(torch.nn.Module):
            def __init__(self, actor_mean):
                super().__init__()
                self.actor_mean = actor_mean

            def forward(self, obs, state_ins):
                """
                ONNX推理时的前向传播
                
                Args:
                    obs: 观测输入
                    state_ins: 状态输入（占位符，用于RNN等场景）
                Returns:
                    action_mean: 确定的动作均值
                    state_outs: 状态输出
                """
                action_mean = self.actor_mean(obs)
                return action_mean, state_ins

        onnx_policy = OnnxPolicy(agent.actor_mean)
        dummy_input = torch.unsqueeze(torch.tensor(envs.single_observation_space.sample()), 0)

        # 导出为ONNX格式
        torch.onnx.export(
            onnx_policy,
            args=(dummy_input, torch.zeros(1).float()),
            f=str(path_onnx),
            opset_version=15,  # ONNX算子版本
            input_names=["obs", "state_ins"],
            output_names=["output", "state_outs"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "state_ins": {0: "batch_size"},  # 可变长度轴
                "output": {0: "batch_size"},
                "state_outs": {0: "batch_size"},
            },
        )
