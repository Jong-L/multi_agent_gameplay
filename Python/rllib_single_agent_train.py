"""
RLlib 单智能体吃球训练脚本
仅训练 player3 (learning_policy)，其余玩家 Godot 端强制 IDLE (idle_policy)

使用前请在 Godot 编辑器中修改 configs/game_config.tres：
  training_player_id = 3
  reset_on_wall = true
  wall_reset_threshold = 2

使用方法：
  python Python/rllib_single_agent_train.py
  python Python/rllib_single_agent_train.py --restore logs/rllib/single_agent_ball_chase/checkpoint_xxx
"""

import argparse
import os
import pathlib
from collections import OrderedDict

import numpy as np
import ray
import yaml
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from gymnasium import spaces

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv


class FlattenDictObsPZEnv:
    """
    PettingZoo 兼容的 wrapper，将 GDRLPettingZooEnv 的 Dict 观测展平为 flat Box。
    放在 ParallelPettingZooEnv 外层之前，使 RLlib 看到的观测从头就是 Box。
    """

    def __init__(self, env: GDRLPettingZooEnv):
        self._env = env
        self.agents = list(env.agents)
        self.possible_agents = list(env.possible_agents)
        self.agent_policy_names = env.agent_policy_names
        self.agent_name_mapping = env.agent_name_mapping

        # 计算展平后的观测空间
        self.observation_spaces = {}
        for agent in self.agents:
            orig_space = env.observation_space(agent)
            if isinstance(orig_space, spaces.Dict):
                ordered = OrderedDict(sorted(orig_space.spaces.items()))
                low = np.concatenate([s.low.ravel() for s in ordered.values()], axis=0)
                high = np.concatenate([s.high.ravel() for s in ordered.values()], axis=0)
                self.observation_spaces[agent] = spaces.Box(
                    low=low.astype(np.float32),
                    high=high.astype(np.float32),
                    dtype=np.float32,
                )
            else:
                self.observation_spaces[agent] = orig_space

        self.action_spaces = env.action_spaces

    def _flat(self, obs_dict):
        if not isinstance(obs_dict, dict):
            return obs_dict
        ordered = sorted(obs_dict.keys())
        return np.concatenate(
            [np.asarray(obs_dict[k], dtype=np.float32).ravel() for k in ordered],
            axis=0,
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs_dict, infos = self._env.reset(seed=seed, options=options)
        return {a: self._flat(obs_dict[a]) for a in obs_dict}, infos

    def step(self, actions):
        obs_dict, rewards, terms, truncs, infos = self._env.step(actions)
        return (
            {a: self._flat(obs_dict[a]) for a in obs_dict},
            rewards,
            terms,
            truncs,
            infos,
        )

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_file",
        default="rllib_single_agent_config.yaml",
        type=str,
        help="The yaml config file",
    )

    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="The location of a checkpoint to restore from",
    )
    parser.add_argument(
        "--experiment_dir",
        default="logs/rllib",
        type=str,
        help="The experiment directory for logs and checkpoints",
    )
    parser.add_argument(
        "--training_player_id",
        default=3,
        type=int,
        help="Which player (0-3) to train. Others will be forced IDLE by Godot.",
    )
    args, extras = parser.parse_known_args()

    # 加载 YAML 配置
    with open(args.config_file, encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    is_multiagent = exp["env_is_multiagent"]

    # 注册环境
    env_name = "godot"
    learning_policy_name = "learning_policy"
    idle_policy_name = "idle_policy"
    training_player_id = args.training_player_id

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_env_runner"] + env_config.vector_index
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        # 链路：GDRLPettingZooEnv → FlattenDictObsPZEnv (展平 obs) → ParallelPettingZooEnv (RLlib 适配)
        raw_env = GDRLPettingZooEnv(config=env_config, port=port, seed=seed)
        flat_env = FlattenDictObsPZEnv(raw_env)
        return ParallelPettingZooEnv(flat_env)

    tune.register_env(env_name, env_creator)

    # 启动临时环境获取 policy_names（从 Godot 端读取）
    print(f"Starting a temporary multi-agent env to get the policy names...")
    print(f"Expected: 3 agents with '{idle_policy_name}', 1 agent with '{learning_policy_name}'")
    tmp_env = GDRLPettingZooEnv(config=exp["config"]["env_config"], show_window=False)
    policy_names = tmp_env.agent_policy_names
    print(f"Got policy names from Godot: {policy_names}")
    tmp_env.close()

    # 验证 Godot 端配置是否正确
    expected_names = [idle_policy_name] * 4
    expected_names[training_player_id] = learning_policy_name
    if policy_names != expected_names:
        print(f"\nWARNING: Policy names mismatch!")
        print(f"  Expected: {expected_names}")
        print(f"  Got:      {policy_names}")
        print(f"  Please make sure game_config.tres has training_player_id = {training_player_id}")
        print(f"  Continuing anyway...\n")

    # 策略映射函数（新 API stack 签名：agent_id + episode，无 worker）
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return policy_names[agent_id]

    ray.init(_temp_dir=os.path.abspath(args.experiment_dir))

    # 配置多智能体策略
    exp["config"]["multiagent"] = {
        "policies": {
            learning_policy_name: PolicySpec(),
            idle_policy_name: PolicySpec(),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": [learning_policy_name],
    }

    # 添加停止条件
    if "stop" in exp:
        exp["config"]["stop"] = exp["stop"]

    # 去除不兼容的配置项
    if "verbose" in exp["config"]:
        del exp["config"]["verbose"]
    if "enable_rl_module_and_learner" in exp["config"]:
        del exp["config"]["enable_rl_module_and_learner"]
    if "enable_env_runner_and_connector_v2" in exp["config"]:
        del exp["config"]["enable_env_runner_and_connector_v2"]

    print(f"\n{'='*60}")
    print(f"Single-Agent Ball Chase Training")
    print(f"  Training player: {training_player_id}")
    print(f"  Learning policy: {learning_policy_name}")
    print(f"  Idle policy (frozen): {idle_policy_name}")
    print(f"  Policies to train: {exp['config']['multiagent']['policies_to_train']}")
    print(f"  Obs: Dict → flat Box")
    print(f"{'='*60}\n")

    tuner = None
    if not args.restore:
        tuner = tune.Tuner(
            trainable=exp["algorithm"],
            param_space=exp["config"],
            run_config=tune.RunConfig(
                storage_path=os.path.abspath(args.experiment_dir),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max",
                    checkpoint_frequency=10,
                ),
            ),
        )
    else:
        tuner = tune.Tuner.restore(
            trainable=exp["algorithm"],
            path=args.restore,
            resume_unfinished=True,
        )

    result = tuner.fit()

    # 导出 ONNX
    checkpoint = result.get_best_result().checkpoint
    if checkpoint:
        result_path = result.get_best_result().path
        ppo = Algorithm.from_checkpoint(checkpoint)
        ppo.get_policy(learning_policy_name).export_model(
            f"{result_path}/onnx_export/{learning_policy_name}_onnx", onnx=12
        )
        print(f"Saved ONNX policy to: {pathlib.Path(f'{result_path}/onnx_export/{learning_policy_name}_onnx').resolve()}")
