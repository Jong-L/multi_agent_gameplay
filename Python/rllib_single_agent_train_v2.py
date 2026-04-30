"""
RLlib 单智能体吃球训练脚本 (旧 API stack + NumPy 2.x monkey-patch)
仅训练 player3 (learning_policy)，其余玩家 Godot 端强制 IDLE (idle_policy)

与 v1 的区别：
- v1: 展平 Dict obs → 新 API stack → 兼容 NumPy 2.x（但新 stack 多 agent 兼容性差）
- v2: 保留 Dict obs → 旧 API stack → monkey-patch 修复 NumPy 2.x 不兼容问题

使用前请在 Godot 编辑器中修改 configs/game_config.tres：
  training_player_id = 3
  reset_on_wall = true
  wall_reset_threshold = 2

使用方法：
  python Python/rllib_single_agent_train_v2.py
  python Python/rllib_single_agent_train_v2.py --restore logs/rllib/single_agent_v2/checkpoint_xxx
"""

import argparse
import os
import pathlib

import numpy as np
import ray
import yaml

# ── Monkey-patch: 修复 RLlib preprocessors 中 NumPy 2.x 不兼容的 copy=False ──
# 根源：ray/rllib/models/preprocessors.py 第 219 行使用了 np.array(obs, copy=False)
# NumPy 2.0+ 在无法避免拷贝时直接抛 ValueError，而 1.x 会静默拷贝
# 修复方法：用 np.asarray() 替换，语义等价且兼容所有 NumPy 版本

def _patch_rllib_preprocessors():
    """Monkey-patch: 修复 RLlib preprocessors 中 np.array(..., copy=False) 对 NumPy 2.x 的不兼容。"""
    import ray.rllib.models.preprocessors as preproc_mod

    for cls_name in ("NoPreprocessor", "MultiBinaryPreprocessor"):
        cls = getattr(preproc_mod, cls_name, None)
        if cls is None:
            continue
        _orig = cls.write

        def _patched_write(self, observation, array, offset):
            array[offset : offset + self._size] = np.asarray(observation).ravel()

        cls.write = _patched_write
        print(f"[patch] RLlib {cls_name}.write: np.array(copy=False) → np.asarray()")


_patch_rllib_preprocessors()

# ── 正常导入 ──
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv

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
        return ParallelPettingZooEnv(
            GDRLPettingZooEnv(config=env_config, port=port, seed=seed)
        )

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

    # 策略映射函数（旧 API stack 签名）
    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
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

    # 强制使用旧 API stack（支持 Dict obs + policy_mapping_fn 旧签名）
    # monkey-patch 已修复 NumPy 2.x 兼容性
    exp["config"]["enable_rl_module_and_learner"] = False
    exp["config"]["enable_env_runner_and_connector_v2"] = False

    print(f"\n{'='*60}")
    print(f"Single-Agent Ball Chase Training (v2: old stack + NumPy patch)")
    print(f"  Training player: {training_player_id}")
    print(f"  Learning policy: {learning_policy_name}")
    print(f"  Idle policy (frozen): {idle_policy_name}")
    print(f"  Policies to train: {exp['config']['multiagent']['policies_to_train']}")
    print(f"  API stack: old (Policy-based), NumPy {np.__version__}")
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
