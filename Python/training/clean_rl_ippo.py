"""
Independent PPO (IPPO)
每个智能体拥有独立的 PPO 网络、优化器和超参数配置，
在同一个 Godot 环境中同时训练多个智能体。
该脚本只支持一个godot进程，n_parallel为1
"""

import os
import pathlib
import time
from collections import deque
from dataclasses import dataclass, field
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
    save_pt_model,
    layer_init,
)


#单个智能体的独立超参数配置
@dataclass
class AgentConfig:
    agent_id: int                  # 映射到 Godot 端 player_id (0~3)

    #训练开关
    train: bool = True     # True=用RL训练; False=强制IDLE

    #网络架构
    self_hidden: int = 32
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hidden: tuple = (256, 128)

    # 优化器
    learning_rate: float = 3e-4

    #超参
    gamma: float = 0.99 # 折扣因子
    gae_lambda: float = 0.95 # GAE 参数
    clip_coef: float = 0.2 # 策略剪辑阈值
    ent_coef: float = 0.001 # 熵项系数
    vf_coef: float = 0.5 # 值函数系数
    max_grad_norm: float = 4.0

    #奖励归一化
    reward_norm: bool = True
    reward_clip: float = 10.0


@dataclass
class IppoArgs:
    """全局训练配置。
    """

    #环境
    # env_path: Optional[str] = None
    env_path: Optional[str] = "godot-game/build/game.exe"
    config_path: str = "godot-game/configs/game_config.tres"
    n_parallel: int = 1 #只能为1
    seed: int = 1
    show_window: bool = True
    speedup: int = 10

    # 训练
    total_timesteps: int = 5_000_000
    num_steps: int = 512
    """每个智能体（环境）每个 rollout 的步数"""
    num_minibatches: int = 8
    update_epochs: int = 10
    norm_adv: bool = True # 是否归一化优势函数
    clip_vloss: bool = True
    anneal_lr: bool = False
    target_kl: Optional[float] = None
    torch_deterministic: bool = True
    cuda: bool = True

    #智能体配置
    agents: list[AgentConfig] = field(default_factory=lambda: [
        AgentConfig(agent_id=0, train=True),
        AgentConfig(agent_id=1,train=True),
        AgentConfig(agent_id=2,train=True),
        AgentConfig(agent_id=3,train=True),
    ])

    #日志 + 保存
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    experiment_dir: str = "logs/cleanrl_ippo"
    # save_model_path: Optional[str] = "saved-models/clean_rl_ippo"
    save_model_path: Optional[str] = None
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None

    #运行时衍生数据
    num_agents: int = 0
    """智能体数量"""
    num_envs: int = 0
    """环境数量"""
    batch_size: int = 0
    """每个智能体共用的样本数量"""
    minibatch_size: int = 0
    """每个智能体共用的小批量样本数量"""

#  Per-Agent Rollout 数据结构
@dataclass
class AgentRolloutData:
    """单个智能体在一次 rollout 中收集的经验。"""
    obs: torch.Tensor        # (num_steps, obs_dim)
    actions: torch.Tensor    # (num_steps,)
    logprobs: torch.Tensor   # (num_steps,)
    rewards: torch.Tensor    # (num_steps,)
    dones: torch.Tensor      # (num_steps,)  — 所有智能体同步的 done 标志
    values: torch.Tensor     # (num_steps,)
    next_obs: torch.Tensor   # (obs_dim,) 最后一步观测
    next_done: torch.Tensor  # scalar  — 最后一步的 done 标志
    next_value: Optional[torch.Tensor] = None  # V(next_obs), GAE 阶段填充

class IPPOAgent(nn.Module):
    """离散 PPO 智能体
    输入: 观测向量 (obs_dim,):
          [self_state | nearby_players | nearby_balls | nearby_enemies | map_state]
    输出: 离散动作 (0~n_actions-1) + 对数概率 + 熵 + 状态价值
    """

    def __init__(
        self,
        n_actions: int,
        seg: ObsSegmentDims,
        self_hidden: int = 16,
        player_hidden: int = 64,
        ball_hidden: int = 64,
        enemy_hidden: int = 32,
        map_hidden: int = 36,
        trunk_hidden: tuple = (256, 128),
    ):
        super().__init__()

        self.seg_self = seg.self_dim
        self.seg_player = seg.player_dim
        self.seg_ball = seg.ball_dim
        self.seg_enemy = seg.enemy_dim
        self.seg_map = seg.map_dim

        #各段独立特征提取子网络
        self.self_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_self, self_hidden)), nn.ReLU()
        )
        self.player_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_player, player_hidden)), nn.ReLU()
        )
        self.ball_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_ball, ball_hidden)), nn.ReLU()
        )
        self.enemy_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_enemy, enemy_hidden)), nn.ReLU()
        )
        self.map_net = nn.Sequential(
            layer_init(nn.Linear(self.seg_map, map_hidden)), nn.ReLU()
        )

        fused_dim = self_hidden + player_hidden + ball_hidden + enemy_hidden + map_hidden

        #共享躯干
        trunk_layers = []
        in_dim = fused_dim
        for h in trunk_hidden:
            trunk_layers.append(layer_init(nn.Linear(in_dim, h)))
            trunk_layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = trunk_hidden[-1] if trunk_hidden else fused_dim

        self.actor = layer_init(nn.Linear(self._trunk_out_dim, n_actions), std=0.01) # Actor head
        self.critic = layer_init(nn.Linear(self._trunk_out_dim, 1), std=1.0) # Critic head

    def _forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        i = 0
        s = obs[:, i: i + self.seg_self];      i += self.seg_self
        p = obs[:, i: i + self.seg_player];    i += self.seg_player
        b = obs[:, i: i + self.seg_ball];      i += self.seg_ball
        e = obs[:, i: i + self.seg_enemy];     i += self.seg_enemy
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
        return self.critic(self._forward_features(obs))

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """根据观测采样动作，计算对数概率、熵和状态价值。"""

        features = self._forward_features(obs)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(features),
        )

#  GAE
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,#最后一步(idx:num_steps)状态价值
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (GAE)。"""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

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
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    return advantages, advantages + values


def compute_actor_loss(
    new_logprob: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    norm_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """裁剪 PPO actor loss。"""
    logratio = new_logprob - old_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    """裁剪 PPO critic loss。"""
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

#  多智能体 Rollout 收集
def collect_rollout_ippo(
    agents_cfg: list[AgentConfig],
    agents: list[IPPOAgent],
    envs: GodotDiscreteEnvWrapper,
    num_steps: int,
    device: torch.device,
    next_obs_all: torch.Tensor,
    next_done_all: torch.Tensor,
    global_step: int,
    episode_returns: list[deque],
    accum_rewards: list,
    reward_normalizers: list[Optional[RewardNormalizer]],
) -> tuple[list[AgentRolloutData], int]:
    """实际包括num_steps+1步"""
    n_agents = len(agents_cfg)
    obs_dim = envs.single_observation_space.shape
    active_agents = [i for i in range(n_agents) if agents_cfg[i].train]

    #预分配 per-agent 缓冲区
    buffers: list[dict] = []
    for _ in range(n_agents):
        buffers.append({
            "obs": torch.zeros((num_steps, obs_dim[0])).to(device),
            "actions": torch.zeros((num_steps,), dtype=torch.long).to(device),
            "logprobs": torch.zeros((num_steps,)).to(device),
            "rewards": torch.zeros((num_steps,)).to(device),
            "dones": torch.zeros((num_steps,)).to(device),
            "values": torch.zeros((num_steps,)).to(device),
        })

    next_obs_all = next_obs_all.clone()#所有智能体的观测
    next_done_all = next_done_all.clone()

    for step in range(num_steps):
        global_step += n_agents

        # 每个智能体独立推理动作
        actions_list = [4] * n_agents  # 默认 IDLE (action index 5)

        for i in range(n_agents):
            obs_i = next_obs_all[i].unsqueeze(0)  # shape(1, obs_dim)

            # 存入缓冲区
            buffers[i]["obs"][step] = obs_i.squeeze(0)#shape(obs_dim,)
            buffers[i]["dones"][step] = next_done_all[i]

            if agents_cfg[i].train:
                with torch.no_grad():
                    action, logprob, _, value = agents[i].get_action_and_value(obs_i)
                    buffers[i]["actions"][step] = action.item()
                    buffers[i]["logprobs"][step] = logprob.item()
                    buffers[i]["values"][step] = value.item()
                
                actions_list[i] = int(action.item())
            else:
                # 非训练智能体: IDLE, 无价值估计
                buffers[i]["actions"][step] = 5
                buffers[i]["logprobs"][step] = 0.0
                buffers[i]["values"][step] = 0.0

        # 所有智能体一起执行动作
        next_obs_raw, rewards_raw, terminations, truncations, infos = envs.step(np.array(actions_list,dtype=np.int64))
        dones_raw = np.logical_or(terminations, truncations)#反应的是第i+1步状态是否终止

        #转换为张量
        next_obs_tensor = torch.tensor(np.array(next_obs_raw, dtype=np.float32)).to(device)
        next_done_tensor = torch.tensor(dones_raw, dtype=torch.float32).to(device)

        for i in range(n_agents):
            r = float(rewards_raw[i])

            # 奖励归一化
            if agents_cfg[i].train and reward_normalizers[i] is not None:
                reward_normalizers[i].update(r)
                r = reward_normalizers[i].normalize(r)

            buffers[i]["rewards"][step] = r

        # 回合奖励追踪 (使用原始奖励)
        for i in range(n_agents):
            if agents_cfg[i].train:
                accum_rewards[i] += float(rewards_raw[i])
                if dones_raw[i]:
                    episode_returns[i].append(accum_rewards[i])
                    accum_rewards[i] = 0.0

        next_obs_all = next_obs_tensor
        next_done_all = next_done_tensor 

    #包装为 AgentRolloutData
    rollouts = []
    for i in range(n_agents):
        next_val = None
        next_done_i = next_done_all[i]
        if agents_cfg[i].train:
            with torch.no_grad():
                next_val = agents[i].get_value(next_obs_all[i].unsqueeze(0)).item()
        rollouts.append(AgentRolloutData(
            obs=buffers[i]["obs"],
            actions=buffers[i]["actions"],
            logprobs=buffers[i]["logprobs"],
            rewards=buffers[i]["rewards"],
            dones=buffers[i]["dones"],
            values=buffers[i]["values"],
            next_obs=next_obs_all[i],
            next_done=next_done_i,
            next_value=next_val,
        ))

    return rollouts, global_step


#  日志系统
def log_ippo(
    writer,
    global_step: int,
    agents_cfg: list[AgentConfig],
    optimizers: list[optim.Optimizer],
    losses: list[dict],         # [{pg_loss, v_loss, entropy, approx_kl, clipfrac}]
    explained_vars: list[float],
    episode_returns: list[deque],
    start_time: float,
    update: int = -1,
    num_updates: int = -1,
) -> None:
    """将 IPPO 训练指标写入 TensorBoard 并打印终端日志。"""
    # 全局指标
    sps = int(global_step / (time.time() - start_time)) if start_time > 0 else 0 #steps per second 
    writer.add_scalar("charts/SPS", sps, global_step)

    # per-agent 指标
    for i in range(len(agents_cfg)):
        tag = f"agent_{i}"

        # 学习率
        writer.add_scalar(
            f"{tag}/learning_rate", optimizers[i].param_groups[0]["lr"], global_step
        )

        # Loss 指标
        if losses[i] is not None:
            writer.add_scalar(f"{tag}/losses/policy_loss", losses[i]["pg_loss"], global_step)
            writer.add_scalar(f"{tag}/losses/value_loss", losses[i]["v_loss"], global_step)
            writer.add_scalar(f"{tag}/losses/entropy", losses[i]["entropy"], global_step)
            writer.add_scalar(f"{tag}/losses/approx_kl", losses[i]["approx_kl"], global_step)
            writer.add_scalar(f"{tag}/losses/clipfrac", losses[i]["clipfrac"], global_step)

        # Explained variance
        writer.add_scalar(f"{tag}/losses/explained_variance", explained_vars[i], global_step)

        # Episode return
        if len(episode_returns[i]) > 0:
            mean_ret = np.mean(np.array(episode_returns[i]))
            writer.add_scalar(f"{tag}/charts/episodic_return", mean_ret, global_step)

    # 终端日志
    if update > 0 and num_updates > 0:
        return_strs = []
        for i in range(len(agents_cfg)):
            if agents_cfg[i].train and len(episode_returns[i]) > 0:
                mean_ret = np.mean(np.array(episode_returns[i]))
                return_strs.append(f"p{i}:{mean_ret:.1f}")

        returns_summary = "  ".join(return_strs)

        kl_summary = "  ".join(
            f"p{i}:{losses[i]['approx_kl']:.3f}"
            for i in range(len(agents_cfg))
            if agents_cfg[i].train and losses[i] is not None
        )

        ev_summary = "  ".join(
            f"p{i}:{explained_vars[i]:.3f}"
            for i in range(len(agents_cfg))
            if agents_cfg[i].train and losses[i] is not None
        )

        print(
            f"[Update {update:4d}/{num_updates}] "
            # f"SPS: {sps:5d}  "
            f"returns [{returns_summary}]  "
            f"kl [{kl_summary}]  "
            f"ev [{ev_summary}]"
        )


#  模型保存
def save_ippo_model(
    save_path: str,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    reward_normalizers: list[Optional[RewardNormalizer]],
    args: IppoArgs,
) -> None:
    """保存 IPPO 所有智能体的模型检查点。
    """
    save_path = pathlib.Path(save_path).with_suffix(".pt")#自动追加 .pt 后缀
    agent_states = {}
    for i, agent in enumerate(agents):
        agent_states[f"agent_{i}_state_dict"] = agent.state_dict() #网络参数
        agent_states[f"agent_{i}_optimizer"] = optimizers[i].state_dict() #优化器参数
        if reward_normalizers[i] is not None:
            agent_states[f"agent_{i}_reward_norm"] = reward_normalizers[i].state_dict() #奖励归一化器参数

    torch.save({"args": vars(args), **agent_states},str(save_path))
    print(f"[Save] IPPO model saved to {save_path}")


#  Per-Agent 训练更新
def train_agent_update(
    agent: IPPOAgent,
    optimizer: optim.Optimizer,
    rollout: AgentRolloutData,
    cfg: AgentConfig,
    args: IppoArgs,
    device: torch.device,
) -> dict:
    """对单个智能体执行一次 PPO 更新 (多个 epoch + minibatch)
    """
    num_steps = rollout.obs.shape[0]
    batch_size = num_steps  # per-agent: num_steps x 1 env

    #GAE
    with torch.no_grad():
        next_value_t = torch.tensor([rollout.next_value], device=device)
        next_done_t = rollout.next_done.unsqueeze(0).to(device)

        advantages, target_values = compute_gae(
            rollout.rewards.unsqueeze(1),     # (num_steps, 1)
            rollout.values.unsqueeze(1),       # (num_steps, 1)
            rollout.dones.unsqueeze(1),        # (num_steps, 1)
            next_value_t.unsqueeze(0),         # (1, 1)
            next_done_t.unsqueeze(0),          # (1, 1)
            cfg.gamma,
            cfg.gae_lambda,
        )
    # Flatten
    b_obs = rollout.obs              # (num_steps, obs_dim)
    b_actions = rollout.actions      # (num_steps,)
    b_logprobs = rollout.logprobs    # (num_steps,)
    b_advantages = advantages.squeeze(1)  # (num_steps,)
    b_target_values = target_values.squeeze(1)  # (num_steps,)
    b_values = rollout.values        # (num_steps,)

    b_inds = np.arange(batch_size)#batch indices
    clipfracs = []
    final_metrics = {}

    minibatch_size = max(1, batch_size // args.num_minibatches)

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            new_value = new_value.view(-1)  # 转换为一维向量(minibatch_size,)

            # Actor loss
            pg_loss, approx_kl, clipfrac = compute_actor_loss(
                new_logprob,
                b_logprobs[mb_inds],
                b_advantages[mb_inds],
                cfg.clip_coef,
                args.norm_adv,
            )
            clipfracs.append(clipfrac)

            # Critic loss
            v_loss = compute_critic_loss(
                new_value,
                b_target_values[mb_inds],
                b_values[mb_inds],
                cfg.clip_coef,
                args.clip_vloss,
            )

            # Total loss
            loss = pg_loss - cfg.ent_coef * entropy.mean() + v_loss * cfg.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            final_metrics = {
                "pg_loss": pg_loss.item(),
                "v_loss": v_loss.item(),
                "entropy": entropy.mean().item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
            }

        # KL 早停
        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    #Explained variance
    y_pred = b_values.cpu().numpy()
    y_true = b_target_values.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    final_metrics["explained_var"] = explained_var

    return final_metrics


#  主训练循环

def train(
    args: IppoArgs,
    agents: list[IPPOAgent],
    optimizers: list[optim.Optimizer],
    envs: GodotDiscreteEnvWrapper,
    device: torch.device,
    writer,
    reward_normalizers: list[Optional[RewardNormalizer]],
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
) -> None:
    """IPPO 主训练循环。"""
    n_agents = len(args.agents)
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    episode_returns = [deque(maxlen=100) for _ in range(n_agents)]
    accum_rewards = np.zeros(n_agents, dtype=np.float64)

    for update in range(1, num_updates + 1):
        # 学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            for i in range(n_agents):
                if args.agents[i].train:
                    optimizers[i].param_groups[0]["lr"] = progress * args.agents[i].learning_rate

        # 经验采集
        rollouts, global_step = collect_rollout_ippo(
            args.agents, agents, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards, reward_normalizers,
        )

        next_obs = torch.stack([r.next_obs for r in rollouts])
        next_done = torch.stack([r.next_done for r in rollouts])

        # 独立更新
        losses = []
        explained_vars = []

        for i in range(n_agents):
            if args.agents[i].train:
                metrics = train_agent_update(agents[i], optimizers[i], rollouts[i], args.agents[i], args, device)
                losses.append(metrics)
                explained_vars.append(metrics.get("explained_var", 0.0))
            else:
                losses.append(None)
                explained_vars.append(0.0)

        # 日志
        log_ippo(
            writer, global_step, args.agents, optimizers,
            losses, explained_vars,
            episode_returns, start_time,
            update=update, num_updates=num_updates,
        )


#  主训练入口
def main():
    # 初始化
    args = IppoArgs()
    writer, device, envs, seg, run_name = init_training_setup(args)

    n_agents = len(args.agents)
    args.num_agents = n_agents
    args.num_envs = envs.num_envs
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = max(1, args.batch_size // args.num_minibatches)

    n_actions = int(envs.single_action_space.n)

    # 创建 per-agent 网络、优化器、奖励归一化器
    agents: list[IPPOAgent] = []
    optimizers: list[optim.Optimizer] = []
    reward_normalizers: list[Optional[RewardNormalizer]] = []

    for cfg in args.agents:
        agent = IPPOAgent(
            n_actions=n_actions,
            seg=seg,
            self_hidden=cfg.self_hidden,
            player_hidden=cfg.player_hidden,
            ball_hidden=cfg.ball_hidden,
            enemy_hidden=cfg.enemy_hidden,
            map_hidden=cfg.map_hidden,
            trunk_hidden=cfg.trunk_hidden,
        ).to(device)

        agents.append(agent)
        optimizers.append(optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5))

        if cfg.train and cfg.reward_norm:
            reward_normalizers.append(RewardNormalizer(clip=cfg.reward_clip))
        else:
            reward_normalizers.append(None)

    # 初始观测
    next_obs_array, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device)
    next_done = torch.zeros(n_agents).to(device)

    try:
        train(
            args, agents, optimizers, envs, device, writer,
            reward_normalizers, next_obs, next_done,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] 训练被手动中断")
        if args.save_model_path is not None:
            print(f"[Interrupt] 保存检查点到 {args.save_model_path} ...")
            save_ippo_model(args.save_model_path, agents, optimizers, reward_normalizers, args)
        return
    finally:
        envs.close()
        writer.close()

    # 正常训练结束后的保存
    if args.save_model_path is not None:
        save_ippo_model(args.save_model_path, agents, optimizers, reward_normalizers, args)


if __name__ == "__main__":
    main()
