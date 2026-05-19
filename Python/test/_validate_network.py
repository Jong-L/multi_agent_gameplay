"""
临时验证脚本: 检查 CustomSegmentedModel 的 GRU_MLP 模式是否与 RLlib API 兼容。
此脚本仅用于验证，不执行实际训练。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from gymnasium.spaces import Box

from godot_env_wrapper import ObsSegmentDims
from rllib_custom_network import CustomSegmentedModel, build_rllib_model_config, NetworkType

# --- 1. 构建模拟的 YAML 配置 ---
mock_exp = {
    "config": {
        "model": NetworkType.GRU_MLP.value,
        "num_envs_per_env_runner": 4,
        "env_config": {},
    },
    "model_config": {
        "gru_mlp": {
            "max_seq_len": 64,
            "gru_hidden": 128,
            "gru_num_layers": 1,
            "gru_input_layernorm": False,
            "ball_hiddens": [64],
            "trunk_hiddens": [128, 64],
        },
    },
    "obs_seg_config_path": r"D:\schoolTour\softwares\multi-agent-gameplay\godot-game\configs\game_config.tres",
}
mock_exp["config"]["model"] = build_rllib_model_config(mock_exp)
model_cfg = mock_exp["config"]["model"]

# --- 2. 获取观测空间 ---
seg_dims = ObsSegmentDims.from_config(mock_exp["obs_seg_config_path"])
total_dim = seg_dims.total
print(f"观测分段维度: SELF={seg_dims.self_dim}, PLAYER={seg_dims.player_dim}, "
      f"BALL={seg_dims.ball_dim}, ENEMY={seg_dims.enemy_dim}, MAP={seg_dims.map_dim}, total={total_dim}")

obs_space = Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
import gymnasium as gym
act_space = gym.spaces.Discrete(6)

# --- 3. 实例化模型 ---
model = CustomSegmentedModel(
    obs_space=obs_space,
    action_space=act_space,
    num_outputs=6,
    model_config=model_cfg,
    name="test_model",
)
print(f"\n模型类型: {model._network_type}")
print(f"特征维度: {model._features_dim}")
print(f"GRU hidden: {model._gru.hidden_size}, num_layers: {model._gru.num_layers}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

# --- 4. 测试 get_initial_state ---
init_state = model.get_initial_state()
print(f"\nget_initial_state(): len={len(init_state)}, shape={init_state[0].shape}")

# --- 5. 测试 rollout 模式 (单步, T=1) ---
print("\n--- Rollout 模式 (T=1) ---")
B = 4  # 4个agent
obs_np = np.random.randn(B, total_dim).astype(np.float32)
state_in = [init_state[0].unsqueeze(0).expand(B, -1).contiguous()]  # 模拟 RLlib 广播
seq_lens = torch.ones(B, dtype=torch.long)

input_dict = {"obs_flat": torch.from_numpy(obs_np)}
logits, state_out = model.forward(input_dict, state_in, seq_lens)
v = model.value_function()
print(f"obs_flat shape: {obs_np.shape}")
print(f"logits shape: {logits.shape} (期望: (4, 6))")
print(f"state_out[0] shape: {state_out[0].shape} (期望: (4, 128))")
print(f"value shape: {v.shape} (期望: (4,))")
assert logits.shape == (B, 6), f"Rollout logits shape mismatch: {logits.shape}"
assert state_out[0].shape == (B, model._gru.num_layers * model._gru.hidden_size), \
    f"Rollout state shape mismatch: {state_out[0].shape}"
assert v.shape == (B,), f"Rollout value shape mismatch: {v.shape}"
print("✅ Rollout 模式通过")

# --- 6. 测试训练模式 (序列批处理, T>1) ---
print("\n--- 训练模式 (B=4, T=32) ---")
B_train, T = 4, 32
obs_np = np.random.randn(B_train * T, total_dim).astype(np.float32)
# RLlib 提供每序列的有效长度 (模拟不均匀截断)
seq_lens_train = torch.tensor([32, 28, 20, 15], dtype=torch.long)
state_train = [init_state[0].unsqueeze(0).expand(B_train, -1).contiguous()]

input_dict_train = {"obs_flat": torch.from_numpy(obs_np)}
logits_train, state_out_train = model.forward(input_dict_train, state_train, seq_lens_train)
v_train = model.value_function()
print(f"obs_flat shape: {obs_np.shape}")
print(f"logits shape: {logits_train.shape} (期望: ({B_train*T}, 6))")
print(f"state_out[0] shape: {state_out_train[0].shape} (期望: ({B_train}, 128))")
print(f"value shape: {v_train.shape} (期望: ({B_train*T},))")
assert logits_train.shape == (B_train * T, 6), f"Train logits shape mismatch: {logits_train.shape}"
assert state_out_train[0].shape == (B_train, model._gru.num_layers * model._gru.hidden_size), \
    f"Train state shape mismatch: {state_out_train[0].shape}"
assert v_train.shape == (B_train * T,), f"Train value shape mismatch: {v_train.shape}"
print("✅ 训练模式通过")

# --- 7. 测试 seq_lens=None 必须报错 (已移除回退逻辑) ---
print("\n--- seq_lens=None 必须报错 ---")
try:
    model.forward(input_dict_train, state_train, None)
    assert False, "seq_lens=None should raise ValueError!"
except ValueError as e:
    assert "seq_lens" in str(e), f"Unexpected error: {e}"
    print(f"seq_lens=None → 正确抛出 ValueError: {e}")
print("✅ seq_lens=None 正确报错")

# --- 8. 测试 SEGMENTED_MLP 模式 ---
print("\n--- SEGMENTED_MLP 模式 ---")
mock_exp["config"]["model"] = NetworkType.SEGMENTED_MLP.value
mock_exp["model_config"] = {
    "segmented_mlp": {
        "self_hiddens": [48],
        "player_hiddens": [64],
        "ball_hiddens": [64],
        "enemy_hiddens": [64],
        "map_hiddens": [64],
        "trunk_hiddens": [128, 64],
    },
}
model_cfg_mlp = build_rllib_model_config(mock_exp)
model_mlp = CustomSegmentedModel(
    obs_space=obs_space, action_space=act_space,
    num_outputs=6, model_config=model_cfg_mlp, name="test_mlp",
)
init_state_mlp = model_mlp.get_initial_state()
assert init_state_mlp == [], f"MLP should return empty initial state"
logits_mlp, state_mlp = model_mlp.forward(input_dict, init_state_mlp, None)
assert logits_mlp.shape == (B, 6), f"MLP logits: {logits_mlp.shape}"
assert state_mlp == [], f"MLP state should be empty"
print(f"SEGMENTED_MLP 参数量: {sum(p.numel() for p in model_mlp.parameters()):,}")
print("✅ SEGMENTED_MLP 模式通过")

# --- 9. 检查 state 一致性: 连续两步的隐藏态不应完全相同 ---
print("\n--- State 连续性检查 ---")
obs1 = torch.randn(1, total_dim)
obs2 = torch.randn(1, total_dim)  # 不同的观测
state0 = [init_state[0].unsqueeze(0).contiguous()]
_, s1 = model.forward({"obs_flat": obs1}, state0, torch.ones(1, dtype=torch.long))
_, s2 = model.forward({"obs_flat": obs2}, s1, torch.ones(1, dtype=torch.long))
state_diff = (s1[0] - s2[0]).abs().max().item()
print(f"不同输入连续两步后隐藏态最大差异: {state_diff:.6f}")
assert state_diff > 1e-8, "连续不同输入应产生不同的隐藏态"
print("✅ State 连续性检查通过")

# --- 10. 全局总结 ---
print("\n" + "=" * 60)
print("🎉 所有验证通过: CustomSegmentedModel GRU_MLP 模式与 RLlib API 兼容")
print("=" * 60)
