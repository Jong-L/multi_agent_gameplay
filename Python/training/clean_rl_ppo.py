"""
离散动作 PPO (Proximal Policy Optimization)
"""

import os
import pathlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

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
    """训练配置 """
    # 环境 
    # env_path: Optional[str] = None
    env_path: Optional[str] = "godot-game\\build\\game.exe"
    """Godot环境路径"""
    config_path: str = "godot-game/configs/game_config.tres"
    """game_config.tres 路径, 用于读取观测维度配置"""
    n_parallel: int = 1
    """并行 Godot 进程数量"""
    seed: int = 1
    """随机种子。"""
    show_window: bool = True
    """是否显示游戏窗口。"""
    speedup: int = 8
    """物理引擎加速倍数"""

    #记录
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验名称, 在 TensorBoard 中显示。"""
    experiment_dir: str = "logs/cleanrl_ppo"
    """TensorBoard 日志目录。"""
    save_model_path: Optional[str] = None
    """模型保存路径 (不加后缀)。"""
    onnx_export_path: Optional[str] = None
    """导出 ONNX 模型的路径。"""
    track: bool = False
    """是否使用 Weights & Biases 追踪实验。"""
    wandb_project_name: str = "cleanRL"
    """W&B 项目名称。"""
    wandb_entity: Optional[str] = None
    """W&B 团队 / 实体名称。"""

    # PPO 算法超参数
    total_timesteps: int = 1_000_000
    """总训练步数 (环境步数)。"""
    learning_rate: float = 3e-4
    """Adam 优化器学习率。"""
    num_steps: int = 512
    """每个智能体（环境）每个 rollout 的步数"""
    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE 的 λ 参数"""
    num_minibatches: int = 10
    """小批量数量"""
    update_epochs: int = 4
    """每次更新遍历数据的轮数，即同一批经验的使用次数"""
    clip_coef: float = 0.2
    """PPO 裁剪系数 ε。"""
    ent_coef: float = 0.001
    """熵系数, 鼓励探索。"""
    vf_coef: float = 0.5
    """价值函数损失系数。"""
    max_grad_norm: float = 4.0
    """梯度裁剪最大范数。"""
    norm_adv: bool = True
    """对优势函数进行标准化。"""
    clip_vloss: bool = True
    """对价值函数损失使用裁剪。"""
    anneal_lr: bool = False
    """对学习率进行线性退火。"""
    target_kl: Optional[float] = None
    """目标 KL 散度阈值, 用于早停 (None = 不启用)"""
    torch_deterministic: bool = True
    """启用 PyTorch 确定性模式。"""
    cuda: bool = True
    """启用 CUDA 加速。"""
    reward_norm: bool = True
    """是否对奖励做 running z-score 归一化。"""
    reward_clip: float = 10.0
    """奖励归一化裁剪范围 (仅在 reward_norm=True 时生效)。"""

    # 运行时计算的衍生值
    num_envs: int = 0
    batch_size: int = 0
    """每次采样的样本数量"""
    minibatch_size: int = 0

#  PPO 网络
class PPOAgent(nn.Module):
    """离散 PPO 智能体
    输入: 观测向量 (obs_dim,):
          [self_state | nearby_players | nearby_balls | nearby_enemies | map_state]
    输出: 离散动作 (0~n_actions-1) + 对数概率 + 熵 + 状态价值
    """

    def __init__(self, n_actions: int, seg: ObsSegmentDims):
        super().__init__()

        #记录各段维度
        self.seg_self = seg.self_dim      # 自身状态维度
        self.seg_player = seg.player_dim  # 附近玩家维度
        self.seg_ball = seg.ball_dim      # 附近球维度
        self.seg_enemy = seg.enemy_dim    # 附近敌人维度
        self.seg_map = seg.map_dim        # 地图状态维度

        # 各段独立特征提取子网络
        self.self_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_self, 16)), nn.ReLU()
        )
        self.player_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_player, 64)), nn.ReLU()
        )
        self.ball_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_ball, 64)), nn.ReLU()
        )
        self.enemy_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_enemy, 32)), nn.ReLU()
        )
        self.map_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_map, 36)), nn.ReLU()
        )

        #融合层维度计算
        fused_dim = 16 + 64 + 64 + 32 + 36  # = 212

        # 共享躯干
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(fused_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )

        #  Actor 头
        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)

        # Critic 头
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    # 前向传播
    def _forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        """将观测向量按段切片, 送入各子网络, 融合后经过共享躯干。"""
        i = 0
        s = obs[:, i: i + self.seg_self];     i += self.seg_self
        p = obs[:, i: i + self.seg_player];   i += self.seg_player
        b = obs[:, i: i + self.seg_ball];     i += self.seg_ball
        e = obs[:, i: i + self.seg_enemy];    i += self.seg_enemy
        m = obs[:, i: i + self.seg_map]

        fused = torch.cat([
            self.self_net(s),
            self.player_net(p),
            self.ball_net(b),
            self.enemy_net(e),
            self.map_net(m),
        ], dim=1)

        return self.trunk(fused)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """获取状态价值 V(s)。"""
        features = self._forward_features(obs)
        return self.critic(features)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        """根据观测采样动作并计算相关统计量。
        """
        features = self._forward_features(obs)
        logits = self.actor(features)
        probs = Categorical(logits=logits) #转换为动作概率分布

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),# 计算动作对数概率
            probs.entropy(),# 计算策略熵
            self.critic(features), 
        )


#  Rollout 数据结构
@dataclass
class RolloutData:
    """单次 rollout 收集的经验数据。
    所有张量均在指定 device
    """
    obs: torch.Tensor   #shape(num_steps, num_envs, obs_dim)
    actions: torch.Tensor   #shape(num_steps, num_envs)
    logprobs: torch.Tensor  #shape(num_steps, num_envs)
    rewards: torch.Tensor   #shape(num_steps, num_envs)
    dones: torch.Tensor #shape(num_steps, num_envs)
    values: torch.Tensor    #shape(num_steps, num_envs)
    next_obs: torch.Tensor  #shape(num_envs, obs_dim)   — 最后一步后的观测
    next_done: torch.Tensor #shape(num_envs,)   — 最后一步后的 done 标志


#  经验收集 (rollout)
def collect_rollout(
    agent: PPOAgent,
    envs: GodotDiscreteEnvWrapper,
    num_steps: int,
    device: torch.device,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    global_step: int,
    episode_returns: deque,
    accum_rewards: np.ndarray,
    reward_normalizer: Optional[RewardNormalizer] = None,
) -> tuple[RolloutData, int]:
    """使用当前策略收集 num_steps 步经验。
    加上最后一步，实际包括num_steps+1步
    """
    num_envs = envs.num_envs
    obs_dim = envs.single_observation_space.shape

    # 预分配缓冲区
    obs = torch.zeros((num_steps, num_envs) + obs_dim).to(device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    for step in range(num_steps):
        global_step += num_envs
        obs[step] = next_obs
        dones[step] = next_done #dones[t]表示s_t是否终止，dones[0]为初始值0

        # 用当前策略采样动作并用旧网络计算状态值
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()#将(1, num_envs)转换为(num_envs,)

        actions[step] = action
        logprobs[step] = logprob

        # 执行动作.next_obs_raw形状为(num_envs, obs_dim)
        next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)

        # 奖励归一化
        reward_arr = np.asarray(reward, dtype=np.float32)
        if reward_normalizer is not None:
            reward_normalizer.update_array(reward_arr)
            reward_arr = reward_normalizer.normalize_array(reward_arr)

        rewards[step] = torch.tensor(reward_arr, dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs_raw, dtype=np.float32)).to(device)
        next_done = torch.tensor(done, dtype=torch.float32).to(device)

        # 追踪回合奖励 (使用原始奖励)
        accum_rewards += np.asarray(reward, dtype=np.float64)
        for i, d in enumerate(np.asarray(done)):
            if d:
                episode_returns.append(accum_rewards[i])
                accum_rewards[i] = 0.0

    return (
        RolloutData(obs, actions, logprobs, rewards, dones, values,next_obs, next_done),
        global_step,
    )


#  GAE 优势估计
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Generalized Advantage Estimation (GAE)。
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = (
            rewards[t]
            + gamma * nextvalues * nextnonterminal
            - values[t]
        )
        advantages[t] = lastgaelam = (
            delta
            + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    target_values = advantages + values
    return advantages, target_values # (num_steps, num_envs)


#  PPO 损失函数
def compute_actor_loss(
    new_logprob: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    norm_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """计算裁剪actor loss。
    """
    logratio = new_logprob - old_logprobs
    ratio = logratio.exp()

    # KL 散度近似
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

    # 优势标准化
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 裁剪策略损失
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return pg_loss, approx_kl, clipfrac


def compute_critic_loss(
    new_value: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    clip_coef: float,
    clip_vloss: bool = True,
) -> torch.Tensor:
    """计算critic loss。
    """
    if clip_vloss:
        v_loss_unclipped = (new_value - returns) ** 2
        v_clipped = old_values + torch.clamp(
            new_value - old_values, -clip_coef, clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()

    return v_loss

#  日志 + 模型导出
def log_ppo(
    writer,
    global_step: int,
    optimizer: optim.Optimizer,
    v_loss: torch.Tensor,
    pg_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    approx_kl: torch.Tensor,
    clipfracs: list,
    explained_var: float,
    episode_returns: deque,
    start_time: float,
    update: int = -1,
    num_updates: int = -1,
) -> None:
    """将 PPO 训练指标写入 TensorBoard 并打印终端日志。
    """
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

    if len(episode_returns) > 0:
        sps = int(global_step / (time.time() - start_time))
        mean_return = np.mean(np.array(episode_returns))
        if update > 0 and num_updates > 0:
            print(
                f"[Update {update:4d}/{num_updates}] "
                f"SPS: {sps:5d}  "
                f"return: {mean_return:8.2f}  "
                f"kl: {approx_kl.item():.4f}"
            )
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/episodic_return", mean_return, global_step)


def export_ppo_onnx(agent: PPOAgent, onnx_path: str, obs_shape: tuple) -> None:
    """将 PPO 智能体导出为 ONNX 模型 (确定性策略)。

    导出时会包装一层 OnnxPolicy, 只输出动作 logits,
    用于 Godot 端的推理 (不需要概率采样)。

    Args:
        agent:     训练好的 PPOAgent
        onnx_path: 导出路径 (不加后缀, 自动追加 .onnx)
        obs_shape: 观测空间形状, 如 (142,)
    """
    path_onnx = pathlib.Path(onnx_path).with_suffix(".onnx")
    print(f"[Export] Exporting ONNX to {path_onnx}")

    agent.eval().to("cpu")

    class OnnxPolicy(torch.nn.Module):
        def __init__(self, ppo_agent):
            super().__init__()
            self.ppo_agent = ppo_agent

        def forward(self, obs, state_ins):
            features = self.ppo_agent._forward_features(obs)
            logits = self.ppo_agent.actor(features)
            return logits, state_ins

    onnx_policy = OnnxPolicy(agent)
    dummy_input = torch.randn(1, int(np.prod(obs_shape)))

    torch.onnx.export(
        onnx_policy,
        args=(dummy_input, torch.zeros(1).float()),
        f=str(path_onnx),
        opset_version=15,
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "state_ins": {0: "batch_size"},
            "output": {0: "batch_size"},
            "state_outs": {0: "batch_size"},
        },
    )
    print(f"[Export] Done: {path_onnx}")


#  主训练入口w

def main():
    #初始化
    args = Args()
    writer, device, envs, seg, run_name = init_training_setup(args)

    # PPO配置 
    args.num_envs = envs.num_envs #智能体（并行环境）的数量
    args.batch_size = args.num_envs * args.num_steps # 每个rollout总采样步数
    args.minibatch_size = args.batch_size // args.num_minibatches
    n_actions = int(envs.single_action_space.n)

    # 智能体 + 优化器
    agent = PPOAgent(n_actions, seg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #训练状态
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    episode_returns = deque(maxlen=100) # 最近100个回合奖励
    accum_rewards: np.ndarray = np.zeros(args.num_envs) # 每个环境每回合的累计奖励

    # 奖励归一化器
    reward_normalizer = None
    if args.reward_norm:
        reward_normalizer = RewardNormalizer(clip=args.reward_clip)
        print(f"[RewardNorm] enabled, clip={args.reward_clip}")

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device)#shape(args.num_envs, obs_dim)
    next_done = torch.zeros(args.num_envs).to(device)#shape(args.num_envs,)

    # 训练循环
    for update in range(1, num_updates + 1):
        # 学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = progress * args.learning_rate

        # 收集经验
        rollout, global_step = collect_rollout(
            agent, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards,
            reward_normalizer=reward_normalizer,
        )
        next_obs = rollout.next_obs
        next_done = rollout.next_done

        # GAE 优势估计
        with torch.no_grad():
            next_value = agent.get_value(rollout.next_obs).reshape(1, -1)

            advantages, target_values = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            next_value, rollout.next_done,
            args.gamma, args.gae_lambda,
        )

        # 展平 rollout 数据,统一形状为(batch_size, *)
        b_obs = rollout.obs.reshape((-1,) + envs.single_observation_space.shape)#shape (num_steps * num_envs, obs_dim)
        b_actions = rollout.actions.reshape(-1) #shape (num_steps * num_envs,)
        b_logprobs = rollout.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_target_values = target_values.reshape(-1)
        b_values = rollout.values.reshape(-1)

        b_inds = np.arange(args.batch_size)#batch indices
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 用当前网络采样动作并计算价值
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_value = new_value.view(-1)

                # Actor loss
                pg_loss, approx_kl, clipfrac = compute_actor_loss(
                    new_logprob,
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    args.clip_coef,
                    args.norm_adv,
                )
                clipfracs.append(clipfrac)

                # Critic loss
                v_loss = compute_critic_loss(
                    new_value,
                    b_target_values[mb_inds],
                    b_values[mb_inds],
                    args.clip_coef,
                    args.clip_vloss,
                )

                #优化loss
                loss = pg_loss- args.ent_coef * entropy.mean()+ v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # KL散度过大时结束本轮更新
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 计算解释方差
        y_pred = b_values.cpu().numpy() #旧网络估计状态值
        y_true = b_target_values.cpu().numpy() #目标状态值
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        # 日志
        log_ppo(
            writer, global_step, optimizer,
            v_loss, pg_loss, entropy.mean(),
            approx_kl, clipfracs, explained_var,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
        )

    #  清理 + 保
    envs.close()
    writer.close()

    if args.save_model_path is not None:
        save_dict = {"agent_state_dict": agent.state_dict()}
        if reward_normalizer is not None:
            save_dict["reward_normalizer"] = reward_normalizer.state_dict()
        save_pt_model(args.save_model_path, save_dict, args)

    if args.onnx_export_path is not None:
        export_ppo_onnx(agent, args.onnx_export_path,
                        envs.single_observation_space.shape)


if __name__ == "__main__":
    main()
