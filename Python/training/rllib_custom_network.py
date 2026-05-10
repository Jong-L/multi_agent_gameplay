# RLlib 可配置训练脚本 for GodotRL.
# 支持通过 model.custom_model 切换不同网络架构 (默认分段模型 SegmentedModel)。
# 需要 rllib_config.yaml 在同一目录, 或通过 --config_file 参数指定路径。

import argparse
import os
import pathlib

import numpy as np
import ray
import yaml
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv

import torch
import torch.nn as nn

# 观测维度常量 — 与 godot-game/scripts/scene_scripts/vision_sensor.gd 保持同步
_SEG_SELF = 6          # self_state
_SEG_PLAYER = 27       # nearby_players: 3 × 9
_SEG_BALL = 32         # nearby_balls: 8 × 4
_SEG_ENEMY = 45        # nearby_enemies: 5 × 9
_SEG_MAP = 36         # map_state: 36 条射线 

# 子网络隐层维度
_SEG_SELF_HIDDEN = 16
_SEG_PLAYER_HIDDEN = 64
_SEG_BALL_HIDDEN = 64
_SEG_ENEMY_HIDDEN = 64
_SEG_MAP_HIDDEN = 64

_FUSED_DIM = _SEG_SELF_HIDDEN + _SEG_PLAYER_HIDDEN + _SEG_BALL_HIDDEN + _SEG_ENEMY_HIDDEN + _SEG_MAP_HIDDEN  # = 212
_TRUNK_HIDDEN_1 = 128
_TRUNK_HIDDEN_2 = 64


def _ortho_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """正交权重初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# 自定义分段模型 (RLlib CustomModel)
class SegmentedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 5 段独立子网络
        self.self_net = nn.Sequential(_ortho_init(nn.Linear(_SEG_SELF, _SEG_SELF_HIDDEN)), nn.ReLU())
        self.player_net = nn.Sequential(_ortho_init(nn.Linear(_SEG_PLAYER, _SEG_PLAYER_HIDDEN)), nn.ReLU())
        self.ball_net = nn.Sequential(_ortho_init(nn.Linear(_SEG_BALL, _SEG_BALL_HIDDEN)), nn.ReLU())
        self.enemy_net = nn.Sequential(_ortho_init(nn.Linear(_SEG_ENEMY, _SEG_ENEMY_HIDDEN)), nn.ReLU())
        self.map_net = nn.Sequential(_ortho_init(nn.Linear(_SEG_MAP, _SEG_MAP_HIDDEN)), nn.ReLU())

        # 共享躯干
        self.trunk = nn.Sequential(
            _ortho_init(nn.Linear(_FUSED_DIM, _TRUNK_HIDDEN_1)), nn.ReLU(),
            _ortho_init(nn.Linear(_TRUNK_HIDDEN_1, _TRUNK_HIDDEN_2)), nn.ReLU(),
        )

        # Actor 头
        self.actor = _ortho_init(nn.Linear(_TRUNK_HIDDEN_2, num_outputs), std=0.01)

        # Critic 头
        self.critic = _ortho_init(nn.Linear(_TRUNK_HIDDEN_2, 1), std=1.0)

        # 注册价值函数分支给 RLlib
        self._value_branch = self.critic

    def _forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        i = 0
        s = obs[:, i: i + _SEG_SELF];     i += _SEG_SELF
        p = obs[:, i: i + _SEG_PLAYER];   i += _SEG_PLAYER
        b = obs[:, i: i + _SEG_BALL];     i += _SEG_BALL
        e = obs[:, i: i + _SEG_ENEMY];    i += _SEG_ENEMY
        m = obs[:, i: i + _SEG_MAP]

        fused = torch.cat([
            self.self_net(s),
            self.player_net(p),
            self.ball_net(b),
            self.enemy_net(e),
            self.map_net(m),
        ], dim=1)

        return self.trunk(fused)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """RLlib 前向: 返回 (action_logits, state_outs)"""
        obs = input_dict["obs_flat"].float()       # (batch, obs_dim)
        features = self._forward_features(obs)
        logits = self.actor(features)              # (batch, num_outputs)
        # 将当前特征缓存, 供 value_function() 使用
        self._features = features
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """RLlib 调用此方法获取 V(s)"""
        return self.critic(self._features).squeeze(1)


# 模型注册 (RLlib 通过 model.custom_model 字符串查找)
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("segmented_model", SegmentedModel)


def _convert_obs(obs):
    """将 Godot 返回的 list 观测转换为 numpy float32 array"""
    if isinstance(obs, dict):
        # Godot 返回的多智能体观测结构为 {agent_id: {"obs": [..values..]}}
        result = {}
        for k, v in obs.items():
            if isinstance(v, dict) and "obs" in v:
                arr = np.array(v["obs"], dtype=np.float32)
                result[k] = {"obs": arr}
            else:
                result[k] = np.array(v, dtype=np.float32)
        return result
    return np.array(obs, dtype=np.float32)


def _wrap_pettingzoo_obs(env):
    """包装 PettingZoo env，确保观测为 float32 numpy array"""
    original_reset = env.reset
    original_step = env.step

    def reset(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        return _convert_obs(obs), info

    def step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        return _convert_obs(obs), reward, terminated, truncated, info

    env.reset = reset
    env.step = step
    return env


# 入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_file", default="Python\\training\\rllib_config.yaml", type=str, help="The yaml config file")
    parser.add_argument("--restore", default=None, type=str, help="the location of a checkpoint to restore from")
    parser.add_argument(
        "--experiment_dir",
        default="logs/rllib",
        type=str,
        help="The name of the the experiment directory, used to store logs.",
    )
    args, extras = parser.parse_known_args()

    # Get config from file
    with open(args.config_file, encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    is_multiagent = exp["env_is_multiagent"]
    show_window = exp["config"]["env_config"]["show_window"]

    # Register env
    env_name = "godot"
    env_wrapper = None

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_env_runner"] + env_config.vector_index
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        if is_multiagent:
            pz_env = GDRLPettingZooEnv(
                config=env_config,
                port=port,
                seed=seed,
                show_window=env_config.get("show_window", False),
            )
            pz_env = _wrap_pettingzoo_obs(pz_env)
            return ParallelPettingZooEnv(pz_env)
        else:
            return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)

    policy_names = None
    num_envs = None
    tmp_env = None

    if is_multiagent:  # Make temp env to get info needed for multi-agent training config
        print("Starting a temporary multi-agent env to get the policy names")
        tmp_env = GDRLPettingZooEnv(config=exp["config"]["env_config"], show_window=False)
        policy_names = tmp_env.agent_policy_names
        print("Policy names for each Agent (AIController) set in the Godot Environment", policy_names)
    else:  # Make temp env to get info needed for setting num_workers training config
        print("Starting a temporary env to get the number of envs and auto-set the num_envs_per_worker config value")
        tmp_env = GodotEnv(env_path=exp["config"]["env_config"]["env_path"], show_window=False)
        num_envs = tmp_env.num_envs

    tmp_env.close()

    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
        return policy_names[agent_id]

    ray.init(_temp_dir=os.path.abspath(args.experiment_dir))

    if is_multiagent:
        exp["config"]["multiagent"] = {
            "policies": {policy_name: PolicySpec() for policy_name in policy_names},
            "policy_mapping_fn": policy_mapping_fn,
        }
    else:
        exp["config"]["num_envs_per_env_runner"] = num_envs

    #使用自定义分段模型替代默认 fcnet
    # 自定义模型时 fcnet_hiddens 会被忽略，子网络维度硬编码在 SegmentedModel 中,vf_share_layers 也被忽略
    exp["config"]["model"] = {
        "custom_model": "segmented_model",
    }

    tuner = None
    if not args.restore:
        tuner = tune.Tuner(
            trainable=exp["algorithm"],
            param_space=exp["config"],
            run_config=train.RunConfig(
                storage_path=os.path.abspath(args.experiment_dir),
                stop=exp["stop"],
                checkpoint_config=train.CheckpointConfig(checkpoint_frequency=exp["checkpoint_frequency"]),
            ),
        )
    else:
        tuner = tune.Tuner.restore(
            trainable=exp["algorithm"],
            path=args.restore,
            resume_unfinished=True,
        )
    result = tuner.fit()

    # Onnx export after training if a checkpoint was saved
    checkpoint = result.get_best_result().checkpoint

    if checkpoint:
        result_path = result.get_best_result().path
        ppo = Algorithm.from_checkpoint(checkpoint)
        if is_multiagent:
            for policy_name in set(policy_names):
                ppo.get_policy(policy_name).export_model(f"{result_path}/onnx_export/{policy_name}_onnx", onnx=12)
                print(
                    f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/{policy_name}_onnx').resolve()}"
                )
        else:
            ppo.get_policy().export_model(f"{result_path}/onnx_export/single_agent_policy_onnx", onnx=12)
            print(
                f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/single_agent_policy_onnx').resolve()}"
            )
