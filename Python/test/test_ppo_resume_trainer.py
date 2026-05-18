import pathlib
import tempfile
import unittest
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from godot_env_wrapper import ObsSegmentDims
from ppo_resume_trainer import (
    Args,
    NetworkType,
    RolloutData,
    _build_train_state,
    _count_completed_episodes,
    _save_checkpoint,
    create_agent_and_optimizer,
    init_observation_state,
    load_checkpoint_if_requested,
    run_training_with_resume,
)


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class FakeVecEnv:
    def __init__(self, num_envs=2, obs_dim=3, episode_period=3):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Discrete(2)
        self.episode_period = episode_period
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return self._obs(), {}

    def step(self, actions):
        self.step_count += 1
        obs = self._obs()
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        if self.step_count % self.episode_period == 0:
            terminated[0] = True
        return obs, rewards, terminated, truncated, {}

    def _obs(self):
        base = np.arange(self.num_envs * 3, dtype=np.float32).reshape(self.num_envs, 3)
        return base + float(self.step_count)


def tiny_args(save_model_path=None):
    args = Args()
    args.network_type = NetworkType.MLP
    args.reward_norm = False
    args.cuda = False
    args.num_envs = 2
    args.num_steps = 4
    args.batch_size = args.num_envs * args.num_steps
    args.num_minibatches = 2
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.update_epochs = 1
    args.total_timesteps = args.batch_size
    args.save_every_n_episodes = 1
    args.max_checkpoints = 2
    args.save_model_path = save_model_path
    args.mlp_hiddens = (8,)
    args.anneal_lr = False
    return args


class PpoResumeTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = torch.device("cpu")
        self.seg = ObsSegmentDims(
            self_dim=1,
            player_dim=1,
            ball_dim=1,
            enemy_dim=0,
            map_dim=0,
        )

    def test_count_completed_episodes_counts_done_rows_once(self):
        rollout = RolloutData(
            obs=torch.zeros((4, 3, 3)),
            actions=torch.zeros((4, 3), dtype=torch.long),
            logprobs=torch.zeros((4, 3)),
            rewards=torch.zeros((4, 3)),
            dones=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            values=torch.zeros((4, 3)),
            next_obs=torch.zeros((3, 3)),
            next_done=torch.tensor([0.0, 0.0, 1.0]),
        )
        self.assertEqual(_count_completed_episodes(rollout), 3)

    def test_checkpoint_round_trip_restores_training_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(pathlib.Path(tmpdir) / "ppo_ckpt.pt")
            args = tiny_args(save_model_path=ckpt_path)
            args.reward_norm = True
            agent, optimizer, reward_normalizer = create_agent_and_optimizer(
                2, self.seg, args, self.device
            )
            reward_normalizer.update_array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

            loss = sum(param.sum() for param in agent.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            saved_state = _build_train_state(
                global_step=12,
                update=3,
                episode_count=5,
                optimizer=optimizer,
                reward_normalizer=reward_normalizer,
                episode_returns=deque([1.5, 2.5], maxlen=20),
            )
            _save_checkpoint(ckpt_path, agent, optimizer, args, reward_normalizer, saved_state)

            restored_agent, restored_optimizer, restored_reward_normalizer = (
                create_agent_and_optimizer(2, self.seg, args, self.device)
            )
            global_step, start_update, episode_count = load_checkpoint_if_requested(
                ckpt_path,
                True,
                restored_agent,
                restored_optimizer,
                restored_reward_normalizer,
                self.device,
            )

            self.assertEqual((global_step, start_update, episode_count), (12, 4, 5))
            self.assertEqual(
                reward_normalizer.state_dict(),
                restored_reward_normalizer.state_dict(),
            )
            self.assertTrue(restored_optimizer.state_dict()["state"])
            for key, value in agent.state_dict().items():
                self.assertTrue(torch.equal(value, restored_agent.state_dict()[key]))

    def test_training_smoke_saves_episode_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = str(pathlib.Path(tmpdir) / "ppo_smoke")
            args = tiny_args(save_model_path=base_path)
            env = FakeVecEnv(num_envs=args.num_envs, episode_period=2)
            agent, optimizer, reward_normalizer = create_agent_and_optimizer(
                2, self.seg, args, self.device
            )
            next_obs, next_done, next_rnn_state = init_observation_state(
                args, env, agent, None, None, None, self.device
            )

            final_state = run_training_with_resume(
                args,
                agent,
                env,
                optimizer,
                self.device,
                DummyWriter(),
                reward_normalizer,
                next_obs,
                next_done,
                next_rnn_state,
                start_global_step=0,
                start_update=1,
                start_episode_count=0,
            )

            self.assertEqual(final_state["global_step"], args.num_steps)
            self.assertEqual(final_state["update"], 1)
            self.assertGreaterEqual(final_state["episode_count"], 1)
            checkpoints = list(pathlib.Path(tmpdir).glob("ppo_smoke_episode*.pt"))
            self.assertEqual(len(checkpoints), 1)


if __name__ == "__main__":
    unittest.main()
