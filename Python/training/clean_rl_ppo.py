"""
离散动作 PPO (Proximal Policy Optimization)
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
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from godot_env_wrapper import (
    GodotDiscreteEnvWrapper,
    parse_godot_tres,
    ObsSegmentDims,
    layer_init,
)

@dataclass
class Args:
    """训练配置 """
    # 环境 
    # env_path: Optional[str] = None
    env_path: Optional[str] = "godot-game\\build\\game.exe"
    """Godot 导出可执行文件路径"""
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
    num_steps: int = 128
    """每个智能体（环境）每个 rollout 的步数"""
    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE 的 λ 参数"""
    num_minibatches: int = 4
    """小批量数量"""
    update_epochs: int = 4
    """每次更新遍历数据的轮数，即同一批经验的使用次数"""
    clip_coef: float = 0.2
    """PPO 裁剪系数 ε。"""
    ent_coef: float = 0.01
    """熵系数, 鼓励探索。"""
    vf_coef: float = 0.5
    """价值函数损失系数。"""
    max_grad_norm: float = 0.5
    """梯度裁剪最大范数。"""
    norm_adv: bool = True
    """对优势函数进行标准化。"""
    clip_vloss: bool = True
    """对价值函数损失使用裁剪。"""
    anneal_lr: bool = False
    """对学习率进行线性退火。"""
    target_kl: Optional[float] = None
    """目标 KL 散度阈值, 用于早停 (None = 不启用)。"""
    torch_deterministic: bool = True
    """启用 PyTorch 确定性模式。"""
    cuda: bool = True
    """启用 CUDA 加速。"""

    # 运行时计算的衍生值
    num_envs: int = 0
    batch_size: int = 0
    """每次采样的样本数量"""
    minibatch_size: int = 0

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


# 主训练入口
def main():
    #初始化
    args = Args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:# 启用 WandB 记录
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = GodotDiscreteEnvWrapper(
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
        seed=args.seed,
        n_parallel=args.n_parallel,
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "只支持 Discrete 动作空间"

    args.num_envs = envs.num_envs #智能体（并行环境）数量
    args.batch_size = args.num_envs * args.num_steps #每次用于更新的样本数量
    args.minibatch_size = args.batch_size // args.num_minibatches#

    n_actions = int(envs.single_action_space.n)

    # print(f"[Env] num_envs={args.num_envs}")
    # print(f"[Env] obs_dim={envs.single_observation_space.shape}")
    # print(f"[Env] n_actions={n_actions}")

    #观测维度分段
    seg = ObsSegmentDims.from_config(args.config_path)

    # print(
    #     f"[Obs] segments: self={seg.self_dim} player={seg.player_dim} "
    #     f"ball={seg.ball_dim} enemy={seg.enemy_dim} map={seg.map_dim} "
    #     f"total={seg.total}"
    # )

    #初始化智能体，优化器
    agent = PPOAgent(n_actions, seg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #预分配 Rollout 缓冲
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape #shape (num_steps, num_envs, obs_dim)
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    #-------
    #训练循环 
    #---------


    global_step = 0 #全局步数
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size #总更新次数

    episode_returns = deque(maxlen=100)#近100个episode的奖励
    accum_rewards = np.zeros(args.num_envs) #每个智能体（并行环境）的每回合累计奖励

    # 初始观测 
    next_obs_array, _ = envs.reset(seed=args.seed)#shape (num_envs, obs_dim)，每个智能体的观测
    next_obs = torch.tensor(np.array(next_obs_array, dtype=np.float32)).to(device) 
    next_done = torch.zeros(args.num_envs).to(device)

    for update in range(1, num_updates + 1):
        #学习率退火
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates
            lrnow = progress * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        #对于每个rollout，收集 num_steps 步经验
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()#从 (num_envs, 1) 转换为 (num_envs,)

            actions[step] = action
            logprobs[step] = logprob

            # 执行动作
            next_obs_raw, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminations, truncations)#shape (num_envs,)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs = torch.tensor(
                np.array(next_obs_raw, dtype=np.float32)
            ).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            # 累加回合奖励
            accum_rewards += np.array(reward)
            for i, d in enumerate(done):
                if d:
                    episode_returns.append(accum_rewards[i])
                    accum_rewards[i] = 0.0

        # 反向遍历进行GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)#shape (1, num_envs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0 #

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal#回合结束时状态值为0
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam #最后一个样本或者是回合结束时的优势为0
                )

            returns = advantages + values

        #展平 Rollout 数据，统一形状为（batch_size,-1）
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)#batch_obs, shape (num_steps * num_envs, obs_dim)
        b_actions = actions.reshape(-1)#shape (num_steps * num_envs,)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #PPO 更新阶段 (K epochs, 小批量)
        b_inds: np.ndarray = np.arange(args.batch_size)#batch indices
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)#打乱数据

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds: np.ndarray = b_inds[start:end] #mini-batch indices

                # 用当前网络计算采样动作的对数概率、熵、价值
                _, newlogprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_value = new_value.view(-1) # 从 (batch_size, 1) 转换为 (batch_size,)

                # 重要性采样比率
                logratio = newlogprob - b_logprobs[mb_inds]#shape (mini_batch_size,)
                ratio = logratio.exp()

                # KL 近似
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()#当前策略近似旧策略的KL散度
                    approx_kl = ((ratio - 1) - logratio).mean()#kl散度的二阶近似
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]#被裁剪的样本的比率

                # 优势标准化
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                #policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value loss with clipping
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()


                entropy_loss = entropy.mean()

                #total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                #反向传播
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # KL 早停
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        #计算解释方差
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        #记录到 TensorBoard
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
            print(
                f"[Update {update:4d}/{num_updates}] "
                f"SPS: {sps:5d}  "
                f"return: {mean_return:8.2f}  "
                f"kl: {approx_kl.item():.4f}"
            )
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/episodic_return", mean_return, global_step)

    #清理+保存
    envs.close()
    writer.close()

    if args.save_model_path is not None:
        save_path = pathlib.Path(args.save_model_path).with_suffix(".pt")
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "args": vars(args),
            },
            str(save_path),
        )
        print(f"[Save] Model saved to {save_path}")

    #ONNX 导出
    if args.onnx_export_path is not None:
        path_onnx = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
        print(f"[Export] Exporting ONNX to {path_onnx}")

        agent.eval().to("cpu")

        class OnnxPolicy(torch.nn.Module):
            def __init__(self, ppo_agent):
                super().__init__()
                self.ppo_agent = ppo_agent

            def forward(self, obs, state_ins):
                # ONNX 推理时只输出动作 logits (确定性)
                features = self.ppo_agent._forward_features(obs)
                logits = self.ppo_agent.actor(features)
                return logits, state_ins

        onnx_policy = OnnxPolicy(agent)
        dummy_input = torch.randn(
            1, int(np.prod(envs.single_observation_space.shape))
        )

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


if __name__ == "__main__":
    main()
