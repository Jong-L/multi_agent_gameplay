# RLlib 标准示例 for single and multi-agent training for GodotRL with onnx export.
# 使用 rllib_config.yaml (fcnet_hiddens: [256, 256] 默认模型)。
# 需要 rllib_config.yaml 在同一目录, 或通过 --config_file 参数指定路径。
#
# 如需自定义网络架构, 请使用 rllib_custom_network.py。

import argparse
import os
import pathlib

import ray
import yaml
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_file", default="Python\\training\\rllib_config.yaml", type=str, help="The yaml config file")
    parser.add_argument("--restore", default="saved-models/checkpoints/checkpoint_v1", type=str, help="the location of a checkpoint to restore from")
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

    # Register env
    env_name = "godot"

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_env_runner"] + env_config.vector_index
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        if is_multiagent:
            return ParallelPettingZooEnv(
                GDRLPettingZooEnv(
                    config=env_config,
                    port=port,
                    seed=seed,
                    show_window=env_config.get("show_window", False),
                )
            )
        else:
            return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)

    policy_names = None
    num_envs = None
    tmp_env = None

    if is_multiagent:
        print("Starting a temporary multi-agent env to get the policy names")
        tmp_env = GDRLPettingZooEnv(config=exp["config"]["env_config"], show_window=False)
        policy_names = tmp_env.agent_policy_names
        print("Policy names for each Agent (AIController) set in the Godot Environment", policy_names)
    else:
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

    tuner = None
    if not args.restore or not os.path.exists(args.restore):
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
