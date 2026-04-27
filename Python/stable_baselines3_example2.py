import argparse
from math import fabs
import os
import pathlib
import time
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# To download the env source and binary:
# 1.  gdrl.env_from_hub -r edbeeching/godot_rl_BallChase
# 2.  chmod +x examples/godot_rl_BallChase/bin/BallChase.x86_64

if can_import("ray"):
    print("WARNING, stable baselines and ray[rllib] are not compatible")

parser = argparse.ArgumentParser(allow_abbrev=False)#全匹配
parser.add_argument(
    "--env_path",
    default="godot-game/build2/game.exe",
    # default=None,
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument(
    "--experiment_dir",
    default="logs/sb3",
    type=str,
    help="The name of the experiment directory, in which the tensorboard logs and checkpoints (if enabled) are "
    "getting stored.",
)
parser.add_argument(
    "--experiment_name",
    default="experiment",
    type=str,
    help="The name of the experiment, which will be displayed in tensorboard and "
    "for checkpoint directory and name (if enabled).",
)
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
parser.add_argument(
    "--resume_model_path",
    default=None,
    type=str,
    help="The path to a model file previously saved using --save_model_path or a checkpoint saved using "
    "--save_checkpoints_frequency. Use this to resume training or infer from a saved model.",
)
parser.add_argument(
    "--save_model_path",
    # default="savedmodels/normal_reward_test",
    default=None,
    type=str,
    help="The path to use for saving the trained sb3 model after training is complete. Saved model can be used later "
    "to resume training. Extension will be set to .zip",
)
parser.add_argument(
    "--save_checkpoint_frequency",
    default=None,
    type=int,
    help=(
        "If set, will save checkpoints every 'frequency' environment steps. "
        "Requires a unique --experiment_name or --experiment_dir for each run. "
        "Does not need --save_model_path to be set. "
    ),
)
parser.add_argument(
    "--onnx_export_path",
    default=None,
    type=str,
    help="If included, will export onnx file after training to the path specified.",
)
parser.add_argument(
    "--timesteps",
    default=800_000,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000. If resuming from a saved model, "
    "it will continue training for this amount of steps from the saved state without counting previously trained "
    "steps",
)
parser.add_argument(
    "--inference",
    default=False,
    help="Instead of training, it will run inference on a loaded model for --timesteps steps. "
    "Requires --resume_model_path to be set.",
)
parser.add_argument(
    "--linear_lr_schedule",
    default=False,
    help="Use a linear LR schedule for training. If set, learning rate will decrease until it reaches 0 at "
    "--timesteps"
    "value. Note: On resuming training, the schedule will reset. If disabled, constant LR will be used.",
)
parser.add_argument(
    "--viz",
    help="If set true, the simulation will be displayed in a window during training. Otherwise "
    "training will run without rendering the simulation. This setting does not apply to in-editor training.",
    default=False,
)
parser.add_argument("--speedup", default=10, type=int, help="Whether to speed up the physics in the env")
parser.add_argument(
    "--n_parallel",
    default=5,
    type=int,
    help="How many instances of the environment executable to " "launch - requires --env_path to be set if > 1.",
)
parser.add_argument(
    "--reward_norm",
    help="If set, apply VecNormalize to normalize rewards (and optionally observations).",
    default=True,
)
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
args, extras = parser.parse_known_args()

def handle_onnx_export():
    # Enforce the extension of onnx and zip when saving model to avoid potential conflicts in case of same name
    # and extension used for both
    if args.onnx_export_path is not None:
        path_onnx = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
        print("Exporting onnx to: " + os.path.abspath(path_onnx))
        export_model_as_onnx(model, str(path_onnx))


def handle_model_save():
    if args.save_model_path is not None:
        zip_save_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
        print("Saving model to: " + os.path.abspath(zip_save_path))
        model.save(zip_save_path)
        # 如果启用了 VecNormalize，同时保存归一化统计数据
        if args.reward_norm:
            vecnorm_path = pathlib.Path(args.save_model_path).with_suffix(".vecnormalize.pkl")
            print("Saving VecNormalize stats to: " + os.path.abspath(vecnorm_path))
            env.save(vecnorm_path)


def close_env():
    try:
        print("closing env")
        env.close()
    except Exception as e:
        print("Exception while closing env: ", e)


def cleanup():
    handle_onnx_export()
    handle_model_save()
    close_env()


class DebugStepCallback(BaseCallback):
    """每秒打印一次step的详细信息，包括done状态"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_print_time = time.time()
        self.step_count = 0
        
    def _on_step(self) -> bool:
        current_time = time.time()
        self.step_count += 1
        
        # 每秒打印一次
        if current_time - self.last_print_time >= 1.0:
            # 获取infos信息
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            rewards = self.locals.get("rewards", [])
            
            print(f"\n{'='*60}")
            print(f"[Debug] Step: {self.step_count}, Time: {time.strftime('%H:%M:%S')}")
            print(f"  Number of envs: {len(dones)}")
            
            for i, (done, reward) in enumerate(zip(dones, rewards)):
                print(f"  Env {i}: done={done}, reward={reward:.4f}")
                
                # 检查是否有episode完成的信息
                if "episode" in infos[i]:
                    ep_info = infos[i]["episode"]
                    print(f"    -> Episode completed! Total reward: {ep_info['r']:.2f}, Length: {ep_info['l']}")
            
            print(f"{'='*60}\n")
            self.last_print_time = current_time
        
        return True


class VecNormalizeCheckpointCallback(BaseCallback):
    """在保存检查点时同时保存VecNormalize统计数据"""
    def __init__(self, checkpoint_callback, save_vecnormalize=True, verbose=0):
        super().__init__(verbose)
        self.checkpoint_callback = checkpoint_callback
        self.save_vecnormalize = save_vecnormalize
        
    def _on_step(self) -> bool:
        continue_training = self.checkpoint_callback.on_step()
        
        # 如果checkpoint被触发且需要保存VecNormalize
        if self.save_vecnormalize and hasattr(self.checkpoint_callback, 'last_save_path'):
            if self.checkpoint_callback.last_save_path is not None:
                # 构造VecNormalize统计文件路径
                checkpoint_path = pathlib.Path(self.checkpoint_callback.last_save_path)
                vecnorm_path = checkpoint_path.with_suffix('.vecnormalize.pkl')
                
                try:
                    # 找到env中的VecNormalize层并保存
                    vec_normalize_env = self._find_vec_normalize(self.training_env)
                    if vec_normalize_env is not None:
                        vec_normalize_env.save(vecnorm_path)
                        if self.verbose > 0:
                            print(f"Saved VecNormalize stats to: {vecnorm_path}")
                except Exception as e:
                    print(f"Warning: Failed to save VecNormalize stats: {e}")
        
        return continue_training
    
    def _find_vec_normalize(self, env):
        """递归查找环境中的VecNormalize实例"""
        if isinstance(env, VecNormalize):
            return env
        elif hasattr(env, 'venv'):
            return self._find_vec_normalize(env.venv)
        elif hasattr(env, 'env'):
            return self._find_vec_normalize(env.env)
        return None

path_checkpoint = os.path.join(args.experiment_dir, args.experiment_name + "_checkpoints")
abs_path_checkpoint = os.path.abspath(path_checkpoint)

# Prevent overwriting existing checkpoints when starting a new experiment if checkpoint saving is enabled
if args.save_checkpoint_frequency is not None and os.path.isdir(path_checkpoint):
    raise RuntimeError(
        abs_path_checkpoint + " folder already exists. "
        "Use a different --experiment_dir, or --experiment_name,"
        "or if previous checkpoints are not needed anymore, "
        "remove the folder containing the checkpoints. "
    )

if args.inference and args.resume_model_path is None:
    raise parser.error("Using --inference requires --resume_model_path to be set.")

if args.env_path is None and args.viz:
    print("Info: Using --viz without --env_path set has no effect, in-editor training will always render.")

env = StableBaselinesGodotEnv(
    env_path=args.env_path, show_window=args.viz, seed=args.seed, n_parallel=args.n_parallel, speedup=args.speedup, gamma=args.gamma
)
env = VecMonitor(env)

# ============ 奖励归一化（可选） ============
# VecNormalize 用 running mean/std 对奖励做 z-score 归一化，稳定训练。
# 用法：
#   --reward_norm          仅归一化奖励
#   --reward_norm --obs_norm  同时归一化奖励和观测
# 不传这两个 flag 则不启用，完全不影响现有行为。
if args.reward_norm:
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
# ============================================


# LR schedule code snippet from:
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if args.resume_model_path is None:
    learning_rate = 0.0003 if not args.linear_lr_schedule else linear_schedule(0.0003)
    model: PPO = PPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.0001,
        verbose=2,
        n_steps=32,
        tensorboard_log=args.experiment_dir,
        learning_rate=learning_rate,
        gamma=args.gamma,
    )
else:
    path_zip = pathlib.Path(args.resume_model_path)
    print("Loading model: " + os.path.abspath(path_zip))
    model = PPO.load(path_zip, env=env, tensorboard_log=args.experiment_dir)
    # 如果启用了 VecNormalize，尝试恢复归一化统计数据
    if args.reward_norm:
        vecnorm_path = path_zip.with_suffix(".vecnormalize.pkl")
        if vecnorm_path.exists():
            print("Loading VecNormalize stats from: " + os.path.abspath(vecnorm_path))
            env = VecNormalize.load(vecnorm_path, env)
        else:
            print(f"WARNING: --reward_norm is set but {vecnorm_path} not found. "
                  "VecNormalize will start with fresh statistics.")

if args.inference:
    obs = env.reset()
    for i in range(args.timesteps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
else:
    learn_arguments = dict(total_timesteps=args.timesteps, tb_log_name=args.experiment_name)
    
    # 创建调试回调
    debug_callback = DebugStepCallback()
    
    if args.save_checkpoint_frequency:
        print("Checkpoint saving enabled. Checkpoints will be saved to: " + abs_path_checkpoint)
        checkpoint_callback = CheckpointCallback(
            save_freq=(args.save_checkpoint_frequency // env.num_envs),
            save_path=path_checkpoint,
            name_prefix=args.experiment_name,
        )
        
        # 如果启用了VecNormalize，使用自定义回调同时保存统计数据
        if args.reward_norm:
            from stable_baselines3.common.callbacks import CallbackList
            vecnorm_checkpoint_callback = VecNormalizeCheckpointCallback(
                checkpoint_callback, 
                save_vecnormalize=True,
                verbose=1
            )
            learn_arguments["callback"] = vecnorm_checkpoint_callback
            print("[Checkpoint] VecNormalize stats will be saved with each checkpoint")
        else:
            # learn_arguments["callback"] = CallbackList([checkpoint_callback, debug_callback])
            learn_arguments["callback"] = checkpoint_callback
    else:
        # learn_arguments["callback"] = debug_callback
        pass
        
    try:
        training_start_time = time.time()
        model.learn(**learn_arguments)
    except (KeyboardInterrupt, ConnectionError, ConnectionResetError):
        print(
            """Training interrupted by user or a ConnectionError. Will save if --save_model_path was
            used and/or export if --onnx_export_path was used."""
        )
    finally:
        cleanup()
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        hours = training_duration // 3600
        minutes = (training_duration % 3600) // 60
        print(f"Total training time: {hours}h {minutes}m")
