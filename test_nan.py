import sys
sys.path.insert(0, 'Python/training')
import numpy as np
import torch
from godot_env_wrapper import init_training_setup, layer_init

args = type('Args', (), {
    'env_path': 'curriculum_envs/s4-enemy-only/build/game.exe',
    'config_path': 'curriculum_envs/s4-enemy-only/configs/game_config.tres',
    'n_parallel': 1,
    'seed': 1,
    'show_window': False,
    'speedup': 16,
    'exp_name': 'test',
    'experiment_dir': 'logs/test',
    'save_model_path': None,
})()

writer, device, envs, seg, run_name = init_training_setup(args)
print(f"Obs dim: {envs.single_observation_space.shape}")
print(f"Seg: self={seg.self_dim}, player={seg.player_dim}, ball={seg.ball_dim}, enemy={seg.enemy_dim}, map={seg.map_dim}")
print(f"Total: {seg.self_dim + seg.player_dim + seg.ball_dim + seg.enemy_dim + seg.map_dim}")

# Test one reset
obs_array, _ = envs.reset(seed=1)
obs = torch.tensor(np.array(obs_array, dtype=np.float32), device=device)
print(f"Obs shape: {obs.shape}")
print(f"Obs has NaN: {torch.isnan(obs).any().item()}")
print(f"Obs min/max: {obs.min().item():.4f} / {obs.max().item():.4f}")

from custom_ppo import PPOAgent, NetworkType
agent = PPOAgent(int(envs.single_action_space.n), seg, args).to(device)
print(f"Params: {agent.num_params():,}")

# Test forward
with torch.no_grad():
    features, _ = agent._forward_features(obs)
    print(f"Features has NaN: {torch.isnan(features).any().item()}")
    logits = agent.actor(features)
    print(f"Logits: {logits}")
    print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
    val = agent.critic(features)
    print(f"Value: {val}")
    print(f"Value has NaN: {torch.isnan(val).any().item()}")

envs.close()
