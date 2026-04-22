import argparse
import os
import pathlib

from stable_baselines3 import PPO
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

def main():
    parser = argparse.ArgumentParser(description="Replay a trained model in the Godot environment.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="savedmodel_no_time_reward.zip",
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
