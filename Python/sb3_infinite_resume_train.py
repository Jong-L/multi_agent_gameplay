"""
SB3 无限训练脚本 — 自动续训、定期保存、持续 LR 退火
==================================================

配置方式: 直接修改下方 Config 数据类的默认值后运行
  python Python/sb3_infinite_resume_train.py
"""
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


@dataclass
class Config:
    """SB3 无限训练配置 — 修改默认值即可调整参数。"""

    # ---- 环境 ----
    env_path: str = "godot-game/build/game.exe"
    """Godot 可执行文件路径。"""
    seed: int = 0
    """随机种子。"""
    n_parallel: int = 1
    """并行环境数量。"""
    viz: bool = False
    """显示游戏窗口。"""
    speedup: int = 10
    """物理引擎加速倍数。"""
    gamma: float = 0.99
    """折扣因子。"""
    reward_norm: bool = True
    """启用奖励归一化。"""

    # ---- 训练循环 ----
    run_name: str = "ppo_infinite_run"
    """运行名称 (用于日志和保存路径)。"""
    log_dir: str = "logs/sb3_infinite"
    """TensorBoard 日志目录。"""
    save_dir: str = "savedmodels/infinite"
    """模型保存根目录。"""
    chunk_timesteps: int = 100_000
    """每个 learn() 块的时间步数 (脚本无限循环直到中断)。"""
    checkpoint_freq: int = 50_000
    """检查点保存频率 (env timesteps, ≤0 禁用)。"""

    # ---- LR 调度 ----
    lr_initial: float = 3e-4
    """初始学习率。"""
    lr_decay_steps: int = 20_000_000
    """LR 线性衰减的总步数。"""
    lr_final_ratio: float = 0.1
    """最终 LR = lr_initial * lr_final_ratio。"""

    # ---- 恢复 ----
    resume_model_path: Optional[str] = None
    """指定恢复训练的模型路径 (None 则自动检测)。"""


class GlobalLinearLrCallback(BaseCallback):
    """
    Continuous LR schedule based on global model.num_timesteps.
    This keeps LR progress continuous across chunked training and resume.
    """

    def __init__(self, initial_lr: float, decay_steps: int, final_ratio: float, verbose: int = 0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.decay_steps = max(1, int(decay_steps))
        self.final_lr = initial_lr * final_ratio

    def _current_lr(self) -> float:
        global_steps = float(self.model.num_timesteps)
        if global_steps >= self.decay_steps:
            return self.final_lr
        frac = 1.0 - (global_steps / self.decay_steps)
        return self.final_lr + (self.initial_lr - self.final_lr) * frac

    def _apply_lr(self):
        lr = self._current_lr()
        update_learning_rate(self.model.policy.optimizer, lr)
        if self.verbose > 1:
            print(f"[LR] steps={self.model.num_timesteps}, lr={lr:.8f}")

    def _on_training_start(self) -> None:
        self._apply_lr()

    def _on_step(self) -> bool:
        self._apply_lr()
        return True


def build_env(args):
    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        show_window=args.viz,
        seed=args.seed,
        n_parallel=args.n_parallel,
        speedup=args.speedup,
        gamma=args.gamma,
    )
    env = VecMonitor(env)
    return env


def resolve_paths(args):
    run_root = pathlib.Path(args.save_dir) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    latest_model_zip = run_root / "latest_model.zip"
    latest_vecnorm = run_root / "latest_model.vecnormalize.pkl"
    state_json = run_root / "train_state.json"
    checkpoint_dir = run_root / "checkpoints"
    return run_root, latest_model_zip, latest_vecnorm, state_json, checkpoint_dir


def save_runtime_state(path: pathlib.Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_runtime_state(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_all(model: PPO, env, model_path: pathlib.Path, vecnorm_path: pathlib.Path, state_path: pathlib.Path, args):
    model.save(model_path)
    if args.reward_norm and isinstance(env, VecNormalize):
        env.save(vecnorm_path)
    state = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_timesteps": int(model.num_timesteps),
        "chunk_timesteps": int(args.chunk_timesteps),
        "lr_initial": float(args.lr_initial),
        "lr_decay_steps": int(args.lr_decay_steps),
        "lr_final_ratio": float(args.lr_final_ratio),
    }
    save_runtime_state(state_path, state)
    print(f"[SAVE] model={model_path}, steps={model.num_timesteps}")


def main():
    args = Config()
    run_root, latest_model_zip, latest_vecnorm, state_json, checkpoint_dir = resolve_paths(args)
    os.makedirs(args.log_dir, exist_ok=True)

    env = build_env(args)

    resume_zip = pathlib.Path(args.resume_model_path) if args.resume_model_path else None
    if resume_zip is None and latest_model_zip.exists():
        resume_zip = latest_model_zip

    if resume_zip is not None and resume_zip.exists():
        if args.reward_norm:
            resume_vecnorm = resume_zip.with_suffix(".vecnormalize.pkl")
            if resume_vecnorm.exists():
                env = VecNormalize.load(resume_vecnorm, env)
                print(f"[RESUME] loaded vecnormalize: {resume_vecnorm}")
            elif latest_vecnorm.exists():
                env = VecNormalize.load(latest_vecnorm, env)
                print(f"[RESUME] loaded vecnormalize fallback: {latest_vecnorm}")
            else:
                env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                print("[RESUME] vecnormalize file not found, using fresh statistics.")
        model = PPO.load(resume_zip, env=env, tensorboard_log=args.log_dir, device="auto")
        model.set_env(env)
        print(f"[RESUME] loaded model: {resume_zip}")
    else:
        if args.reward_norm:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model = PPO(
            "MultiInputPolicy",
            env,
            ent_coef=0.0001,
            verbose=1,
            n_steps=32,
            tensorboard_log=args.log_dir,
            learning_rate=args.lr_initial,
            gamma=args.gamma,
            device="auto",
        )
        print("[START] training from scratch.")

    last_state = load_runtime_state(state_json)
    if last_state is not None:
        print(f"[STATE] previous recorded steps={last_state.get('num_timesteps')}")
    print(f"[RUN] run_root={run_root}")
    print(f"[RUN] tensorboard_log={args.log_dir}, tb_log_name={args.run_name}")

    lr_callback = GlobalLinearLrCallback(
        initial_lr=args.lr_initial,
        decay_steps=args.lr_decay_steps,
        final_ratio=args.lr_final_ratio,
        verbose=0,
    )

    callbacks = [lr_callback]
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // env.num_envs),
            save_path=str(checkpoint_dir),
            name_prefix=args.run_name,
        )
        callbacks.append(checkpoint_cb)
        print(f"[CKPT] every {args.checkpoint_freq} steps -> {checkpoint_dir}")

    callback = CallbackList(callbacks)

    try:
        chunk_index = 0
        while True:
            chunk_index += 1
            before_steps = model.num_timesteps
            print(f"[LEARN] chunk={chunk_index}, from_steps={before_steps}, +{args.chunk_timesteps}")
            model.learn(
                total_timesteps=args.chunk_timesteps,
                callback=callback,
                tb_log_name=args.run_name,
                reset_num_timesteps=False,
            )
            save_all(model, env, latest_model_zip, latest_vecnorm, state_json, args)
    except (KeyboardInterrupt, ConnectionError, ConnectionResetError) as e:
        print(f"[STOP] interrupted ({type(e).__name__}), saving latest state...")
        save_all(model, env, latest_model_zip, latest_vecnorm, state_json, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
