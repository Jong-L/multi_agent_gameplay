"""
PPO 定期断点保存训练脚本
========================
基于 custom_ppo.py 扩展，不修改原训练脚本。

功能:
  1. 每 N 个 update 自动保存 checkpoint（模型 + optimizer + 环境状态 + global_step）
  2. --resume_from PATH: 完全续训（从上次保存的 checkpoint 继续）
  3. --load_model_path PATH: 仅加载模型参数，从头训练

不再使用 Ctrl+C 中断保护与回滚逻辑：异常直接抛出，由外层安全关闭环境。

用法:
  # 新训练，每 10 个 update 保存一次
  python ppo_resume_trainer.py --save_every_n_updates 10

  # 从 checkpoint 完全续训
  python ppo_resume_trainer.py --resume_from saved_models/ppo_checkpoint_update100.pt

  # 仅加载模型，从头训练
  python ppo_resume_trainer.py --load_model_path saved_models/ppo_final.pt
"""

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

# ─── 路径 ───────────────────────────────────────────────
_TRAINING_DIR = str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, _TRAINING_DIR)

# ─── 导入 custom_ppo 全部内容 ───────────────────────────
from custom_ppo import (  # noqa: E402
    NetworkType,
    PPOAgent,
    RewardNormalizer,
    SegmentedObsHelper,
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


# ╔══════════════════════════════════════════════════════════╗
# ║              Args (custom_ppo.Args 扩展)                  ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class Args:
    """训练配置 (继承 custom_ppo.Args 全部字段 + 新增断点续训字段)"""

    # ── 环境 ──────────────────────────────────────────
    env_path: Optional[str] = "curriculum_envs\\s4-enemy-only\\build\\game.exe"
    config_path: str = "curriculum_envs\\s4-enemy-only\\configs\\game_config.tres"
    n_parallel: int = 4
    seed: int = 1
    show_window: bool = False
    speedup: int = 16

    # ── 记录 ──────────────────────────────────────────
    exp_name: str = "resume_trainer"
    experiment_dir: str = "logs/cleanrl_ppo"
    save_model_path: Optional[str] = "saved_models/ppo_gru_mlp"
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None

    # ── PPO 超参数 ─────────────────────────────────────
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 8
    recurrent_seq_len: int = 32
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 4.0
    norm_adv: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = False
    target_kl: Optional[float] = None
    torch_deterministic: bool = True
    cuda: bool = True
    reward_norm: bool = True
    reward_clip: float = 5.0

    # ── 网络结构 ───────────────────────────────────────
    network_type: NetworkType = NetworkType.GRU_MLP
    self_hidden: int = 32
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hidden: tuple = (128, 64)
    mlp_hiddens: tuple = (256, 128, 64)
    gru_hidden: int = 128
    gru_num_layers: int = 1
    gru_input_layernorm: bool = True

    # ── 运行时衍生 ─────────────────────────────────────
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0

    # ── 断点续训（新增）────────────────────────────────
    resume_from: Optional[str] = None
    """完全续训：加载模型 + optimizer + 环境状态 + global_step"""
    load_model_path: Optional[str] = None
    """仅加载模型：加载参数后从头训练"""
    save_every_n_updates: int = 1
    """每 N 个 update 保存一次 checkpoint；<=0 表示不保存中间 checkpoint"""
    max_checkpoints: int = 3
    """最多保留多少个中间 checkpoint；超出时删除最旧的"""


# ╔══════════════════════════════════════════════════════════╗
# ║              辅助函数                                      ║
# ╚══════════════════════════════════════════════════════════╝


def _to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """GPU/CPU tensor → numpy (float32)，None 保持不变。"""
    if t is None:
        return None
    return t.detach().cpu().numpy().astype(np.float32)


def _build_train_state(
    global_step: int,
    update: int,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    next_rnn_state: Optional[torch.Tensor],
    optimizer: optim.Optimizer,
    reward_normalizer: Optional[RewardNormalizer],
    episode_returns: deque,
) -> dict:
    """打包当前训练状态为 dict，用于 checkpoint 保存。"""
    state = dict(
        global_step=int(global_step),
        update=int(update),
        next_obs=_to_numpy(next_obs),
        next_done=_to_numpy(next_done),
        next_rnn_state=_to_numpy(next_rnn_state),
        lr=float(optimizer.param_groups[0]["lr"]),
    )
    if reward_normalizer is not None:
        state["reward_normalizer"] = reward_normalizer.state_dict()
    state["episode_returns"] = list(episode_returns)
    return state


def _load_train_state(
    ckpt: dict,
    num_envs: int,
    obs_dim: tuple,
    device: torch.device,
) -> tuple[int, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """从 checkpoint dict 恢复训练运行时变量。"""
    global_step = int(ckpt.get("global_step", 0))
    update = int(ckpt.get("update", 0))

    # 恢复 next_obs / next_done
    next_obs_np = ckpt.get("next_obs")
    next_done_np = ckpt.get("next_done")
    if next_obs_np is not None:
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    else:
        next_obs = torch.zeros((num_envs,) + obs_dim, device=device)
    if next_done_np is not None:
        next_done = torch.tensor(next_done_np, dtype=torch.float32).to(device)
    else:
        next_done = torch.zeros(num_envs, device=device)

    # 恢复 next_rnn_state
    next_rnn_state = None
    rnn_np = ckpt.get("next_rnn_state")
    if rnn_np is not None:
        next_rnn_state = torch.tensor(rnn_np, dtype=torch.float32).to(device)

    return global_step, update, next_obs, next_done, next_rnn_state


def _restore_reward_normalizer(
    ckpt: dict,
    reward_normalizer: Optional[RewardNormalizer],
) -> None:
    """从 checkpoint 恢复奖励归一化器状态。"""
    if reward_normalizer is not None and "reward_normalizer" in ckpt:
        reward_normalizer.load_state_dict(ckpt["reward_normalizer"])


def _make_checkpoint_path(base_path: Optional[str], update: int) -> Optional[str]:
    """生成带 update 编号的 checkpoint 路径。"""
    if base_path is None:
        return None
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{base}_update{update}{ext}"


def _cleanup_old_checkpoints(base_path: Optional[str], max_keep: int) -> None:
    """删除超出保留数量的旧 checkpoint。"""
    if base_path is None or max_keep <= 0:
        return
    base, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    prefix = os.path.basename(base) + "_update"
    dir_name = os.path.dirname(base_path) or "."

    if not os.path.isdir(dir_name):
        return

    checkpoints = []
    for f in os.listdir(dir_name):
        if f.startswith(prefix) and f.endswith(ext):
            try:
                # 提取 update 编号
                num_str = f[len(prefix):-len(ext)]
                update_num = int(num_str)
                checkpoints.append((update_num, os.path.join(dir_name, f)))
            except ValueError:
                continue

    # 按 update 编号排序，删除旧的
    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, old_path = checkpoints.pop(0)
        try:
            os.remove(old_path)
            print(f"[Cleanup] 删除旧 checkpoint: {old_path}")
        except OSError:
            pass


# ╔══════════════════════════════════════════════════════════╗
# ║                  训练循环（定期保存 checkpoint）            ║
# ╚══════════════════════════════════════════════════════════╝


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
) -> dict:
    """执行 PPO 训练循环，定期保存 checkpoint。

    不再处理 KeyboardInterrupt：异常直接向上抛出，由调用方负责关闭环境。
    """
    global_step = start_global_step
    start_time = time.time()
    num_updates_total = args.total_timesteps // args.batch_size
    episode_returns: deque = deque(maxlen=20)
    accum_rewards: np.ndarray = np.zeros(args.num_envs)

    for update in range(start_update, num_updates_total + 1):
        # ── 学习率退火 ──────────────────────────────
        if args.anneal_lr:
            progress = 1.0 - (update - 1.0) / num_updates_total
            optimizer.param_groups[0]["lr"] = progress * args.learning_rate

        # ═══ 阶段 1: 收集样本 ═══
        rollout, global_step = _custom_ppo_collect_rollout(
            agent, envs, args.num_steps, device,
            next_obs, next_done, global_step,
            episode_returns, accum_rewards,
            reward_normalizer=reward_normalizer,
            rnn_state=next_rnn_state,
        )

        # 更新 env 状态
        next_obs = rollout.next_obs
        next_done = rollout.next_done
        next_rnn_state = rollout.next_rnn_state

        # ═══ 阶段 2: GAE 优势估计 ═══
        with torch.no_grad():
            next_value = agent.get_value(
                rollout.next_obs, rollout.next_rnn_state
            ).reshape(1, -1)
            advantages, target_values = compute_gae(
                rollout.rewards, rollout.values, rollout.dones,
                next_value, rollout.next_done,
                args.gamma, args.gae_lambda,
            )

        # ═══ 阶段 3: 策略更新 ═══
        b_obs = rollout.obs.reshape(
            (-1,) + envs.single_observation_space.shape
        )
        b_actions = rollout.actions.reshape(-1)
        b_logprobs = rollout.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_target_values = target_values.reshape(-1)
        b_values = rollout.values.reshape(-1)
        clipfracs = []

        if agent.is_recurrent:
            seq_len = max(1, min(int(args.recurrent_seq_len), args.num_steps))
            seq_starts, seq_ends, seq_envs = [], [], []
            for env_i in range(args.num_envs):
                for start_t in range(0, args.num_steps, seq_len):
                    seq_starts.append(start_t)
                    seq_ends.append(min(start_t + seq_len, args.num_steps))
                    seq_envs.append(env_i)
            seq_starts = np.asarray(seq_starts)
            seq_ends = np.asarray(seq_ends)
            seq_envs = np.asarray(seq_envs)
            seq_inds = np.arange(len(seq_starts))
            seqs_per_minibatch = max(
                1, (len(seq_inds) + args.num_minibatches - 1) // args.num_minibatches
            )

            for epoch in range(args.update_epochs):
                np.random.shuffle(seq_inds)
                epoch_kls = []
                for start in range(0, len(seq_inds), seqs_per_minibatch):
                    mb_seq_inds = seq_inds[start:start + seqs_per_minibatch]
                    mb_inds, new_logprob, entropy, new_value = (
                        evaluate_recurrent_sequences(
                            agent, rollout,
                            seq_starts[mb_seq_inds],
                            seq_ends[mb_seq_inds],
                            seq_envs[mb_seq_inds],
                            device,
                        )
                    )
                    pg_loss, approx_kl, clipfrac = compute_actor_loss(
                        new_logprob, b_logprobs[mb_inds],
                        b_advantages[mb_inds], args.clip_coef, args.norm_adv,
                    )
                    clipfracs.append(clipfrac)
                    epoch_kls.append(approx_kl.item())

                    v_loss = compute_critic_loss(
                        new_value, b_target_values[mb_inds],
                        b_values[mb_inds], args.clip_coef, args.clip_vloss,
                    )
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if (args.target_kl is not None
                        and float(np.mean(epoch_kls)) > args.target_kl):
                    break
        else:
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds],
                    )
                    new_value = new_value.view(-1)

                    pg_loss, approx_kl, clipfrac = compute_actor_loss(
                        new_logprob, b_logprobs[mb_inds],
                        b_advantages[mb_inds], args.clip_coef, args.norm_adv,
                    )
                    clipfracs.append(clipfrac)

                    v_loss = compute_critic_loss(
                        new_value, b_target_values[mb_inds],
                        b_values[mb_inds], args.clip_coef, args.clip_vloss,
                    )
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                pass

        # ═══ 阶段 4: 日志 & 定期保存 ═══
        y_pred = b_values.cpu().numpy()
        y_true = b_target_values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )
        log_ppo(
            writer, global_step, optimizer,
            v_loss, pg_loss, entropy.mean(),
            approx_kl, clipfracs, explained_var,
            episode_returns, start_time,
            update=update, num_updates=num_updates_total,
        )

        # 每 N 个 update 保存 checkpoint
        if args.save_every_n_updates > 0 and update % args.save_every_n_updates == 0:
            train_state = _build_train_state(
                global_step, update,
                next_obs, next_done, next_rnn_state,
                optimizer, reward_normalizer, episode_returns,
            )
            ckpt_path = _make_checkpoint_path(args.save_model_path, update)
            if ckpt_path:
                _gew.save_pt_model(
                    ckpt_path,
                    {"agent_state_dict": agent.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict()},
                    args,
                    reward_normalizer=reward_normalizer,
                    extra=train_state,
                )
                print(f"[Checkpoint] update={update}, step={global_step} -> {ckpt_path}")
                _cleanup_old_checkpoints(args.save_model_path, args.max_checkpoints)

    # 返回最终训练状态
    return _build_train_state(
        global_step, num_updates_total,
        next_obs, next_done, next_rnn_state,
        optimizer, reward_normalizer, episode_returns,
    )


def _custom_ppo_collect_rollout(
    agent: PPOAgent,
    envs,
    num_steps: int,
    device: torch.device,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    global_step: int,
    episode_returns: deque,
    accum_rewards: np.ndarray,
    reward_normalizer: Optional[RewardNormalizer] = None,
    rnn_state: Optional[torch.Tensor] = None,
):
    """与 custom_ppo.collect_rollout 完全一致的副本。"""
    from custom_ppo import RolloutData  # 延迟导入避免循环

    num_envs = envs.num_envs
    obs_dim = envs.single_observation_space.shape

    obs = torch.zeros((num_steps, num_envs) + obs_dim).to(device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    rnn_states = None
    if agent.is_recurrent:
        rnn_states = torch.zeros(
            (num_steps, num_envs, agent.recurrent_state_size)
        ).to(device)
        if rnn_state is None:
            rnn_state = agent.get_initial_state(num_envs, device)

    for step in range(num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            if agent.is_recurrent:
                rnn_state = rnn_state * (1.0 - next_done).view(-1, 1)
                rnn_states[step] = rnn_state
            action, logprob, _, value, next_rnn_state = agent.get_action_and_value(
                next_obs, rnn_state=rnn_state, return_state=True,
            )
            values[step] = value.flatten()
            if agent.is_recurrent:
                rnn_state = next_rnn_state.detach()

        actions[step] = action
        logprobs[step] = logprob

        next_obs_raw, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        done = np.logical_or(terminations, truncations)

        reward_arr = np.asarray(reward, dtype=np.float32)
        if reward_normalizer is not None:
            reward_arr = reward_normalizer.normalize_array(reward_arr)
            reward_normalizer.update_array(np.asarray(reward, dtype=np.float32))

        rewards[step] = torch.tensor(reward_arr, dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs_raw, dtype=np.float32)).to(device)
        next_done = torch.tensor(done, dtype=torch.float32).to(device)

        accum_rewards += np.asarray(reward, dtype=np.float64)
        for i, d in enumerate(np.asarray(done)):
            if d:
                episode_returns.append(accum_rewards[i])
                accum_rewards[i] = 0.0

    from custom_ppo import RolloutData  # 局部导入避免循环
    return (
        RolloutData(
            obs, actions, logprobs, rewards, dones, values,
            next_obs, next_done, rnn_states, rnn_state,
        ),
        global_step,
    )


# ╔══════════════════════════════════════════════════════════╗
# ║                   主入口                                  ║
# ╚══════════════════════════════════════════════════════════╝


def main():
    # ── 解析命令行参数 ──────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(
        description="PPO 定期断点保存训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础字段
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=8)
    parser.add_argument("--recurrent_seq_len", type=int, default=32)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=4.0)
    parser.add_argument("--norm_adv", action="store_true", default=True)
    parser.add_argument("--no_norm_adv", action="store_false", dest="norm_adv")
    parser.add_argument("--clip_vloss", action="store_true", default=True)
    parser.add_argument("--no_clip_vloss", action="store_false", dest="clip_vloss")
    parser.add_argument("--anneal_lr", action="store_true", default=False)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--reward_norm", action="store_true", default=True)
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--network_type", type=str, default="mlp",
                        choices=["mlp", "segmented_mlp", "gru_mlp"])
    parser.add_argument("--n_parallel", type=int, default=4)
    parser.add_argument("--speedup", type=int, default=16)
    parser.add_argument("--show_window", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--track", action="store_true", default=False)
    parser.add_argument("--save_model_path", type=str, default="saved_models/ppo_mlp_checkpoint_test")
    parser.add_argument("--experiment_dir", type=str, default="logs/cleanrl_ppo")
    parser.add_argument("--seed", type=int, default=1)

    # 断点续训参数
    parser.add_argument("--resume_from", type=str, 
                        # default=None,
                        default="saved_models/ppo_mlp_checkpoint_test_update20.pt",
                        help="完全续训：加载模型 + optimizer + 环境状态")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="仅加载模型参数，从头训练")
    parser.add_argument("--save_every_n_updates", type=int, default=10,
                        help="每 N 个 update 保存一次 checkpoint；<=0 表示不保存中间 checkpoint")
    parser.add_argument("--max_checkpoints", type=int, default=3,
                        help="最多保留多少个中间 checkpoint")

    cli_args = parser.parse_args()

    # ── 构建 Args ──────────────────────────────────────
    args = Args()
    for key, value in vars(cli_args).items():
        if hasattr(args, key):
            setattr(args, key, value)

    # 设置本脚本专用字段
    args.save_every_n_updates = cli_args.save_every_n_updates
    args.max_checkpoints = cli_args.max_checkpoints

    # ── 初始化 ─────────────────────────────────────────
    writer, device, envs, seg, run_name = init_training_setup(args)
    args.num_envs = envs.num_envs
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    n_actions = int(envs.single_action_space.n)

    agent = PPOAgent(n_actions, seg, args).to(device)
    print(f"[PPO] network_type={args.network_type}, params={agent.num_params():,}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    reward_normalizer = None
    if args.reward_norm:
        reward_normalizer = RewardNormalizer(clip=args.reward_clip)
        print(f"[RewardNorm] enabled, clip={args.reward_clip}")

    # ── 续训 / 仅加载模型 ──────────────────────────────
    start_global_step = 0
    start_update = 1
    next_obs = None
    next_done = None
    next_rnn_state = None

    resume_path = args.resume_from or args.load_model_path

    if resume_path:
        print(f"[Resume] 加载 checkpoint: {resume_path}")
        ckpt = _gew.load_full_checkpoint(resume_path, device)
        agent.load_state_dict(ckpt["agent_state_dict"])

        if args.resume_from:
            # 完全续训：恢复 optimizer / reward_normalizer / 环境状态
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("[Resume] optimizer state restored")
            _restore_reward_normalizer(ckpt, reward_normalizer)
            start_global_step, start_update, next_obs, next_done, next_rnn_state = (
                _load_train_state(ckpt, args.num_envs,
                                  envs.single_observation_space.shape, device)
            )
            print(f"[Resume] 从 update {start_update} / step {start_global_step} 继续")
        else:
            # 仅加载模型：其余从头初始化
            print("[Load] 仅加载模型参数，其余从头初始化")

    # ── 初始观测（仅全新训练需要） ──────────────────────
    if next_obs is None:
        next_obs_array, _ = envs.reset(seed=args.seed)
        next_obs = torch.tensor(
            np.array(next_obs_array, dtype=np.float32), device=device
        )
    if next_done is None:
        next_done = torch.zeros(args.num_envs, device=device)
    if next_rnn_state is None:
        next_rnn_state = agent.get_initial_state(args.num_envs, device)

    # ── 训练（异常直接抛出，finally 安全关闭环境） ─────
    final_state = None
    try:
        final_state = run_training_with_resume(
            args, agent, envs, optimizer, device, writer,
            reward_normalizer, next_obs, next_done, next_rnn_state,
            start_global_step, start_update,
        )
    finally:
        safe_close(envs)
        try:
            writer.close()
        except Exception:
            pass

    # ── 正常结束：保存最终模型 ─────────────────────────
    if final_state is not None and args.save_model_path is not None:
        save_dict = {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        _gew.save_pt_model(
            args.save_model_path, save_dict, args,
            reward_normalizer=reward_normalizer,
            extra=final_state,
        )
        print(f"[Done] 最终模型已保存到 {args.save_model_path}")


if __name__ == "__main__":
    main()
