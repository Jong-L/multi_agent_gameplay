import argparse
import os
import pathlib

from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def main():
    parser = argparse.ArgumentParser(description="Replay a trained model in the Godot environment.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="savedmodels/valid-mask-model.zip",
        help="Path to the saved .zip model file."
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Path to the Godot executable. If not provided, it will try to connect to an open editor."
    )
    parser.add_argument(
        "--speedup",
        type=int,
        default=1,
        help="Speed up the physics simulation (e.g., 4 for 4x speed)."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=True,
        help="Show the game window during replay."
    )
    parser.add_argument(
        "--reward_norm",
        action="store_true",
        default=True,
        help="Use VecNormalize for reward normalization (should match training config)."
    )
    parser.add_argument(
        "--obs_norm",
        action="store_true",
        default=False,
        help="Also normalize observations via VecNormalize."
    )
    
    args = parser.parse_args()

    # Resolve paths
    model_path = pathlib.Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading model from: {model_path}")
    
    # Initialize Environment
    print("Initializing Godot environment...")
    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        show_window=args.viz,
        speedup=args.speedup,
        n_parallel=1
    )
    env = VecMonitor(env)
    # Apply VecNormalize if enabled (must match training configuration)
    if args.reward_norm:
        env = VecNormalize(env, norm_obs=args.obs_norm, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
        # Try to load VecNormalize statistics if available
        vecnorm_path = model_path.with_suffix(".vecnormalize.pkl")
        if vecnorm_path.exists():
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
        else:
            print(f"WARNING: VecNormalize stats file not found at {vecnorm_path}")
            print("Using fresh normalization statistics (may affect performance)")

    # Load Model
    print("Loading PPO model...")
    model = PPO.load(model_path, env=env)

    # Replay Loop
    print("Starting replay... Press Ctrl+C to stop.")
    obs = env.reset()
    try:
        while True:
            # Predict action deterministically for best performance display
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Optional: Print info
            # print(f"Reward: {rewards}, Done: {dones}")
            
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
