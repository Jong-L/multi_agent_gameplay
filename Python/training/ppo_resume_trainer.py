"""
PPO 定期断点保存训练脚本

功能:
  1. 每 N 个 episode 自动保存 checkpoint（模型 + optimizer + global_step）
  2. --resume_from PATH: 完全续训（从上次保存的 checkpoint 继续）
  3. --load_model_path PATH: 仅加载模型参数，从头训练
"""

import argparse
import os
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─── 路径 ──────────────────────────────────────────────────────────────────
_TRAINING_DIR = str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, _TRAINING_DIR)

# ─── 顶层统一导入（消除局部导入）─────────────────────────────────────────
from custom_ppo import (  # noqa: E402
    Args as CustomPPOArgs,
    NetworkType,
    PPOAgent,
    RewardNormalizer,
    RolloutData,
    SegmentedObsHelper,
    collect_rollout,
    compute_actor_loss,
    compute_critic_loss,
    compute_gae,
    evaluate_recurrent_sequences,
    init_training_setup,
    log_ppo,
)
import godot_env_wrapper as _gew
from godot_env_wrapper import (  # noqa: E402
    GodotDiscreteEnvWrapper,
    ObsSegmentDims,
    safe_close,
)


@dataclass
class Args(CustomPPOArgs):
    """训练配置 (custom_ppo.Args + 断点续训字段)"""
    # ── 断点续训 ────────────────────────────────────────────────────────
    resume_from: Optional[str] = None
    """完全续训：加载模型 + optimizer + reward normalizer + 训练计数（默认 None）"""
    load_model_path: Optional[str] = None
    """仅加载模型：加载参数后从头训练"""
    save_every_n_episodes: int = 10
    """每 N 个完整 episode 保存一次 checkpoint；<=0 表示不保存中间 checkpoint"""
    max_checkpoints: int = 3
    """最多保留多少个中间 checkpoint；超出时删除最旧的"""

def parse_cli_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """解析命令行参数，返回 Namespace。"""
    return parser.parse_args()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建覆盖 Args dataclass 所有字段的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="PPO 定期断点保存训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 环境 ─────────────────────────────────────────────────────────────
    parser.add_argument("--env-path", type=str, dest="env_path", default=Args.env_path)
    parser.add_argument("--config-path", type=str, dest="config_path", default=Args.config_path)
    parser.add_argument("--n-parallel", type=int, dest="n_parallel", default=Args.n_parallel)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--show-window", action="store_true", dest="show_window", default=Args.show_window)
    parser.add_argument("--speedup", type=int, default=Args.speedup)

    # ── 记录 ─────────────────────────────────────────────────────────────
    parser.add_argument("--exp-name", type=str, dest="exp_name", default=Args.exp_name)
    parser.add_argument("--experiment-dir", type=str, dest="experiment_dir", default=Args.experiment_dir)
    parser.add_argument("--save-model-path", type=str, dest="save_model_path", default=Args.save_model_path)
    parser.add_argument("--track", action="store_true", default=Args.track)
    parser.add_argument("--wandb-project-name", type=str, dest="wandb_project_name", default=Args.wandb_project_name)
    parser.add_argument("--wandb-entity", type=str, dest="wandb_entity", default=Args.wandb_entity)

    # ── PPO 超参数 ──────────────────────────────────────────────────────
    parser.add_argument("--total-timesteps", type=int, dest="total_timesteps", default=Args.total_timesteps)
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=Args.learning_rate)
    parser.add_argument("--num-steps", type=int, dest="num_steps", default=Args.num_steps)
    parser.add_argument("--gamma", type=float, default=Args.gamma)
    parser.add_argument("--gae-lambda", type=float, dest="gae_lambda", default=Args.gae_lambda)
    parser.add_argument("--num-minibatches", type=int, dest="num_minibatches", default=Args.num_minibatches)
    parser.add_argument("--update-epochs", type=int, dest="update_epochs", default=Args.update_epochs)
    parser.add_argument("--recurrent-seq-len", type=int, dest="recurrent_seq_len", default=Args.recurrent_seq_len)
    parser.add_argument("--clip-coef", type=float, dest="clip_coef", default=Args.clip_coef)
    parser.add_argument("--ent-coef", type=float, dest="ent_coef", default=Args.ent_coef)
    parser.add_argument("--vf-coef", type=float, dest="vf_coef", default=Args.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, dest="max_grad_norm", default=Args.max_grad_norm)
    parser.add_argument("--norm-adv", action="store_true", dest="norm_adv", default=Args.norm_adv)
    parser.add_argument("--no-norm-adv", action="store_false", dest="norm_adv")
    parser.add_argument("--clip-vloss", action="store_true", dest="clip_vloss", default=Args.clip_vloss)
    parser.add_argument("--no-clip-vloss", action="store_false", dest="clip_vloss")
    parser.add_argument("--anneal-lr", action="store_true", dest="anneal_lr", default=Args.anneal_lr)
    parser.add_argument("--target-kl", type=float, dest="target_kl", default=Args.target_kl)
    parser.add_argument("--torch-deterministic", action="store_true", dest="torch_deterministic", default=Args.torch_deterministic)
    parser.add_argument("--no-torch-deterministic", action="store_false", dest="torch_deterministic")
    parser.add_argument("--cuda", action="store_true", default=Args.cuda)
    parser.add_argument("--no-cuda", action="store_false", dest="cuda")
    parser.add_argument("--reward-norm", action="store_true", dest="reward_norm", default=Args.reward_norm)
    parser.add_argument("--no-reward-norm", action="store_false", dest="reward_norm")
    parser.add_argument("--reward-clip", type=float, dest="reward_clip", default=Args.reward_clip)

    # ── 网络结构 ─────────────────────────────────────────────────────────
    parser.add_argument("--network-type", type=str, dest="network_type", default=Args.network_type.value,
                        choices=["mlp", "segmented_mlp", "gru_mlp"])
    parser.add_argument("--self-hidden", type=int, dest="self_hidden", default=Args.self_hidden)
    parser.add_argument("--player-hidden", type=int, dest="player_hidden", default=Args.player_hidden)
    parser.add_argument("--ball-hidden", type=int, dest="ball_hidden", default=Args.ball_hidden)
    parser.add_argument("--enemy-hidden", type=int, dest="enemy_hidden", default=Args.enemy_hidden)
    parser.add_argument("--map-hidden", type=int, dest="map_hidden", default=Args.map_hidden)
    parser.add_argument("--trunk-hidden", type=int, nargs="+", dest="trunk_hidden", default=list(Args.trunk_hidden))
    parser.add_argument("--mlp-hiddens", type=int, nargs="+", dest="mlp_hiddens", default=list(Args.mlp_hiddens))
    parser.add_argument("--gru-hidden", type=int, dest="gru_hidden", default=Args.gru_hidden)
    parser.add_argument("--gru-num-layers", type=int, dest="gru_num_layers", default=Args.gru_num_layers)
    parser.add_argument("--gru-input-layernorm", action="store_true", dest="gru_input_layernorm", default=Args.gru_input_layernorm)
    parser.add_argument("--no-gru-input-layernorm", action="store_false", dest="gru_input_layernorm")

    # ── 断点续训 ────────────────────────────────────────────────────────
    parser.add_argument("--resume-from", type=str, dest="resume_from", default=Args.resume_from)
    parser.add_argument("--load-model-path", type=str, dest="load_model_path", default=Args.load_model_path)
    parser.add_argument("--save-every-n-episodes", type=int, dest="save_every_n_episodes", default=Args.save_every_n_episodes)
    parser.add_argument("--max-checkpoints", type=int, dest="max_checkpoints", default=Args.max_checkpoints)

    return parser


def build_args_from_cli(cli_args: argparse.Namespace) -> Args:
    """将 CLI Namespace 合并到 Args dataclass，返回最终 Args。
    优先级：CLI 参数 > Args 默认值。脚本专有字段直接赋值。
    """
    args = Args()
    cli_vars = vars(cli_args)

    for key, value in cli_vars.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args.network_type = NetworkType(args.network_type)
    args.trunk_hidden = tuple(args.trunk_hidden)
    args.mlp_hiddens = tuple(args.mlp_hiddens)

    return args

def _count_completed_episodes(rollout: RolloutData) -> int:
    """Count episode boundaries observed during this rollout.

    The shared Godot episode ends when any controlled slot reports done. The
    current-step done flag is stored in the next row, with the final step stored
    in rollout.next_done.
    """
    done_rows = [rollout.next_done.unsqueeze(0)]
    if rollout.dones.shape[0] > 1:
        done_rows.insert(0, rollout.dones[1:])
    dones = torch.cat(done_rows, dim=0)
    return int(torch.any(dones > 0.5, dim=1).sum().item())

def run_training_with_resume(
    args: Args,
    agent: PPOAgent,
    envs,
    optimizer: optim.Optimizer,
    device: torch.device,
    writer,
    reward_normalizer: Optional[RewardNormalizer],
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    next_rnn_state: Optional[torch.Tensor],
    start_global_step: int,
    start_update: int,
    start_episode_count: int,
    start_episode_returns: Optional[deque] = None,
) -> dict:
    """执行 PPO 训练循环；训练逻辑与 custom_ppo.train 保持一致。"""
    global_step = start_global_step
    episode_count = start_episode_count
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    episode_returns = deque(start_episode_returns or [], maxlen=20)#最近20个回合的奖励
    accum_rewards: np.ndarray = np.zeros(args.num_envs)#每回合累计奖励
    next_checkpoint_episode = None
    if args.save_every_n_episodes > 0:
        interval = int(args.save_every_n_episodes)
        next_checkpoint_episode = ((episode_count // interval) + 1) * interval

    for update in range(start_update, num_updates + 1):
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
            rnn_state=next_rnn_state,
        )
        next_obs = rollout.next_obs
        next_done = rollout.next_done
        next_rnn_state = rollout.next_rnn_state
        completed_episodes = _count_completed_episodes(rollout)
        episode_count += completed_episodes

        # GAE 优势估计
        with torch.no_grad():
            next_value = agent.get_value(rollout.next_obs, rollout.next_rnn_state).reshape(1, -1)

            advantages, target_values = compute_gae(
                rollout.rewards, rollout.values, rollout.dones,
                next_value, rollout.next_done,
                args.gamma, args.gae_lambda,
            )

        # 展平 rollout 数据,统一形状为(batch_size, *)
        b_obs = rollout.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = rollout.actions.reshape(-1)
        b_logprobs = rollout.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_target_values = target_values.reshape(-1)
        b_values = rollout.values.reshape(-1)

        clipfracs = []

        if agent.is_recurrent:# 如果是循环神经网络
            seq_len = max(1, min(int(args.recurrent_seq_len), args.num_steps))
            seq_starts = []
            seq_ends = []
            seq_envs = []
            for env_i in range(args.num_envs):#对于每个智能体
                for start_t in range(0, args.num_steps, seq_len):
                    seq_starts.append(start_t)
                    seq_ends.append(min(start_t + seq_len, args.num_steps))
                    seq_envs.append(env_i)#env_i=0,1,2,...

            seq_starts = np.asarray(seq_starts)#序列在样本中的起始位置
            seq_ends = np.asarray(seq_ends)#序列在样本中的结束位置
            seq_envs = np.asarray(seq_envs)
            seq_inds = np.arange(len(seq_starts))#[0,1,2,...,len(seq_starts)-1]
            seqs_per_minibatch = max(1, (len(seq_inds) + args.num_minibatches - 1) // args.num_minibatches)
            """"每个minibatch包含多少个序列,等价于len(seq_inds)/args.num_minibatches向上取整"""

            for epoch in range(args.update_epochs):
                np.random.shuffle(seq_inds)#打乱序列索引

                #对每个minibatch中的所有子序列
                for start in range(0, len(seq_inds), seqs_per_minibatch):
                    mb_seq_inds = seq_inds[start:start + seqs_per_minibatch]#子序列组成一个minibatch的长序列
                    mb_inds, new_logprob, entropy, new_value = evaluate_recurrent_sequences(
                        agent,
                        rollout,
                        seq_starts[mb_seq_inds],
                        seq_ends[mb_seq_inds],
                        seq_envs[mb_seq_inds],
                        device,
                    )

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

                    # 优化 loss
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # KL散度过大时结束本轮更新（用最后一个epoch的KL）
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        else:
            b_inds = np.arange(args.batch_size)#batch_indices

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)# 打乱索引

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]#切出小批量mini batch indices

                    # 用当前网络采样动作并计算价值
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
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

                    # 优化 loss
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # KL散度过大时结束本轮更新
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        # 计算解释方差
        y_pred = b_values.cpu().numpy()
        y_true = b_target_values.cpu().numpy()
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

        # 每 N 个完整 episode 保存 checkpoint
        if (
            next_checkpoint_episode is not None
            and episode_count >= next_checkpoint_episode
        ):
            train_state = _build_train_state(
                global_step, update, episode_count,
                optimizer, episode_returns,
            )
            ckpt_path = _make_checkpoint_path(args.save_model_path, episode_count)
            if ckpt_path:
                _save_checkpoint(
                    ckpt_path, agent, optimizer,
                    args, reward_normalizer, train_state,
                )
                print(
                    f"[Checkpoint] episode={episode_count}, "
                    f"update={update}, step={global_step} -> {ckpt_path}"
                )
                _cleanup_old_checkpoints(args.save_model_path, args.max_checkpoints)
            while episode_count >= next_checkpoint_episode:
                next_checkpoint_episode += int(args.save_every_n_episodes)

    return _build_train_state(
        global_step, num_updates, episode_count,
        optimizer, episode_returns,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     Checkpoint 管理                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _build_train_state(
    global_step: int,
    update: int,
    episode_count: int,
    optimizer: optim.Optimizer,
    episode_returns: deque,
) -> dict:
    """打包当前训练状态为 dict，用于 checkpoint 保存"""
    state: dict = dict(
        global_step=int(global_step),
        update=int(update),
        episode_count=int(episode_count),
        lr=float(optimizer.param_groups[0]["lr"]),
    )
    state["episode_returns"] = list(episode_returns)
    return state


def _load_train_state(
    ckpt: dict,
) -> tuple[int, int, int, deque]:
    """从 checkpoint dict 恢复训练运行时变量。"""
    global_step = int(ckpt.get("global_step", 0))
    update = int(ckpt.get("update", 0))
    episode_count = int(ckpt.get("episode_count", 0))
    episode_returns = deque(ckpt.get("episode_returns", []), maxlen=20)
    return global_step, update, episode_count, episode_returns


def _restore_reward_normalizer(
    ckpt: dict,
    reward_normalizer: Optional[RewardNormalizer],
) -> None:
    """从 checkpoint 恢复奖励归一化器状态。"""
    if reward_normalizer is not None and "reward_normalizer" in ckpt:
        reward_normalizer.load_state_dict(ckpt["reward_normalizer"])


def _make_checkpoint_path(base_path: Optional[str], episode_count: int) -> Optional[str]:
    """生成带 episode 编号的 checkpoint 路径。"""
    if base_path is None:
        return None
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{base}_episode{episode_count}{ext}"


def _cleanup_old_checkpoints(base_path: Optional[str], max_keep: int) -> None:
    """删除超出保留数量的旧 checkpoint。"""
    if base_path is None or max_keep <= 0:
        return
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    prefix = os.path.basename(base) + "_episode"
    dir_name = os.path.dirname(base_path) or "."

    if not os.path.isdir(dir_name):
        return

    checkpoints: list[tuple[int, str]] = []
    for f in os.listdir(dir_name):
        if f.startswith(prefix) and f.endswith(ext):
            try:
                num_str = f[len(prefix):-len(ext)]
                update_num = int(num_str)
                checkpoints.append((update_num, os.path.join(dir_name, f)))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, old_path = checkpoints.pop(0)
        try:
            os.remove(old_path)
        except OSError:
            pass


def _save_checkpoint(
    ckpt_path: str,
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
    reward_normalizer: Optional[RewardNormalizer],
    extra: dict,
) -> None:
    """将模型 + optimizer + 训练状态保存到 checkpoint 文件。"""
    _gew.save_pt_model(
        ckpt_path,
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args,
        reward_normalizer=reward_normalizer,
        extra=extra,
    )

def create_agent_and_optimizer(
    n_actions: int,
    seg: ObsSegmentDims,
    args: Args,
    device: torch.device,
) -> tuple[PPOAgent, optim.Optimizer, Optional[RewardNormalizer]]:
    """创建智能体、优化器，以及可选的奖励归一化器。"""
    agent = PPOAgent(n_actions, seg, args).to(device)
    print(f"[PPO] network_type={args.network_type}, params={agent.num_params():,}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    reward_normalizer = None
    if args.reward_norm:
        reward_normalizer = RewardNormalizer(clip=args.reward_clip)
        print(f"[RewardNorm] enabled, clip={args.reward_clip}")

    return agent, optimizer, reward_normalizer


def load_checkpoint_if_requested(
    resume_path: Optional[str],
    is_resume: bool,
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    reward_normalizer: Optional[RewardNormalizer],
    device: torch.device,
) -> tuple[int, int, int, deque]:
    """从 checkpoint 恢复模型 / optimizer / 归一化器 / 训练计数。"""
    if not resume_path:
        return 0, 1, 0, deque(maxlen=20)

    print(f"[Resume] 加载 checkpoint: {resume_path}")
    ckpt = _gew.load_full_checkpoint(resume_path, device)
    agent.load_state_dict(ckpt["agent_state_dict"])

    if is_resume:
        # 完全续训：恢复 optimizer / 归一化器 / 训练计数；环境从新 episode reset。
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Resume] optimizer state restored")
        _restore_reward_normalizer(ckpt, reward_normalizer)
        start_global_step, start_update, start_episode_count, episode_returns = _load_train_state(ckpt)
        start_update += 1
        print(
            f"[Resume] 从 update {start_update} / step {start_global_step} "
            f"/ episode {start_episode_count} 继续"
        )
    else:
        # 仅加载模型：其余从头初始化
        print("[Load] 仅加载模型参数，其余从头初始化")
        return 0, 1, 0, deque(maxlen=20)

    return start_global_step, start_update, start_episode_count, episode_returns

def init_observation_state(
    args: Args,
    envs,
    agent: PPOAgent,
    next_obs: Optional[torch.Tensor],
    next_done: Optional[torch.Tensor],
    next_rnn_state: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """初始化观测 / done / RNN 隐藏态。

    episode 级 checkpoint 不恢复 Godot 内部状态，续训时也从新 episode reset。
    """
    if next_obs is None:
        next_obs_array, _ = envs.reset(seed=args.seed)
        next_obs = torch.tensor(
            np.array(next_obs_array, dtype=np.float32), device=device
        )
    if next_done is None:
        next_done = torch.zeros(args.num_envs, device=device)
    if next_rnn_state is None:
        next_rnn_state = agent.get_initial_state(args.num_envs, device)
    return next_obs, next_done, next_rnn_state


def save_final_model(
    save_model_path: Optional[str],
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
    reward_normalizer: Optional[RewardNormalizer],
    final_state: dict,
) -> None:
    """训练正常结束后保存最终模型。"""
    if save_model_path is None or final_state is None:
        return
    save_dict = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    _gew.save_pt_model(
        save_model_path, save_dict, args,
        reward_normalizer=reward_normalizer,
        extra=final_state,
    )
    print(f"[Done] 最终模型已保存到 {save_model_path}")

def main():
    # ── 1. 解析命令行参数（覆盖 Args 默认值）─────────────────────────────
    cli_ns = parse_cli_args(_build_arg_parser())
    args = build_args_from_cli(cli_ns)

    # ── 2. 初始化环境 / TensorBoard / 设备 ─────────────────────────────
    writer, device, envs, seg, _run_name = init_training_setup(args)
    args.num_envs = envs.num_envs
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    n_actions = int(envs.single_action_space.n)

    # ── 3. 创建智能体 / 优化器 / 归一化器 ────────────────────────────
    agent, optimizer, reward_normalizer = create_agent_and_optimizer(n_actions, seg, args, device,)

    # ── 4. 恢复 / 初始化训练状态 ──────────────────────────────────────
    resume_path = args.resume_from or args.load_model_path
    is_resume = bool(args.resume_from)
    start_global_step, start_update, start_episode_count, episode_returns = (
        load_checkpoint_if_requested(
            resume_path, is_resume,
            agent, optimizer, reward_normalizer,
            device,
        )
    )
    next_obs, next_done, next_rnn_state = init_observation_state(
        args, envs, agent,
        None, None, None, device,
    )

    # ── 5. 训练循环────────────────────────────────
    final_state = None
    try:
        final_state = run_training_with_resume(
            args, agent, envs, optimizer, device, writer,
            reward_normalizer, next_obs, next_done, next_rnn_state,
            start_global_step, start_update, start_episode_count,
            episode_returns,
        )
    finally:
        safe_close(envs)
        try:
            writer.close()
        except Exception:
            pass

    # ── 6. 保存最终模型 ──────────────────────────────────────────────
    save_final_model(
        args.save_model_path, agent, optimizer, args,
        reward_normalizer, final_state,
    )


if __name__ == "__main__":
    main()
