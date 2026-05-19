"""
测试 IPPO 并行环境训练的正确性。
独立于 Godot 环境，使用模拟数据进行验证。
"""

import sys
from pathlib import Path

# 确保能 import 项目模块
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Mock godot_rl（避免需要安装 Godot 运行时）
import types
_mock_godot_rl = types.ModuleType("godot_rl")
_mock_godot_rl_wrappers = types.ModuleType("godot_rl.wrappers")
_mock_godot_rl_clean_rl = types.ModuleType("godot_rl.wrappers.clean_rl_wrapper")
_mock_godot_rl_clean_rl.CleanRLGodotEnv = type("CleanRLGodotEnv", (), {})
_mock_godot_rl.wrappers = _mock_godot_rl_wrappers
_mock_godot_rl_wrappers.clean_rl_wrapper = _mock_godot_rl_clean_rl
sys.modules["godot_rl"] = _mock_godot_rl
sys.modules["godot_rl.wrappers"] = _mock_godot_rl_wrappers
sys.modules["godot_rl.wrappers.clean_rl_wrapper"] = _mock_godot_rl_clean_rl

import numpy as np
import torch
import torch.optim as optim
from collections import deque
from typing import Optional

from custom_ippo import (
    IPPOAgent,
    collect_parallel_rollout_ippo,
    compute_gae,
    compute_actor_loss,
    compute_critic_loss,
    evaluate_recurrent_sequences,
    train_agent_update,
)
from custom_ppo_dataclass import AgentConfig, IppoArgs, RolloutData, NetworkType
from godot_env_wrapper import ObsSegmentDims, RewardNormalizer


# ─── 模拟环境 ───────────────────────────────────────────────
class MockDiscreteEnv:
    """模拟 GodotDiscreteEnvWrapper 的最小接口。"""

    def __init__(self, n_parallel: int, obs_dim: int, n_actions: int):
        self.num_envs = n_parallel
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._step = 0

        from gymnasium import spaces
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(n_actions)

    def reset(self, seed=None):
        obs = np.random.randn(self.num_envs, self._obs_dim).astype(np.float32) * 0.1
        return obs, {}

    def step(self, actions):
        self._step += 1
        obs = np.random.randn(self.num_envs, self._obs_dim).astype(np.float32) * 0.1
        rewards = np.random.randn(self.num_envs).astype(np.float32) * 0.1
        # 模拟偶尔的 episode 结束
        dones = np.zeros(self.num_envs, dtype=bool)
        if self._step % 50 == 0:
            dones[np.random.choice(self.num_envs, size=max(1, self.num_envs // 4))] = True
        return obs, rewards, dones, dones, {}

    def close(self):
        pass


# ─── 测试用 ObsSegmentDims ──────────────────────────────────
def make_test_seg(obs_dim: int = 45) -> ObsSegmentDims:
    """构造简单的观测分段（MLP 模式下仅 total 有用，但 GRU 需要分段）。"""
    return ObsSegmentDims(
        self_dim=8 + 7,   # 15
        player_dim=10,
        ball_dim=8,
        enemy_dim=5,
        map_dim=7,
    )  # total = 45


# ─── 辅助函数 ───────────────────────────────────────────────
def make_test_agents(
    n_agents: int, n_actions: int, seg: ObsSegmentDims,
    network_type: NetworkType = NetworkType.MLP,
    train_mask: list[bool] | None = None,
) -> tuple[list[IPPOAgent], list[AgentConfig]]:
    """创建测试用的 agent 和配置。"""
    if train_mask is None:
        train_mask = [True] * n_agents
    agents = []
    configs = []
    for i in range(n_agents):
        cfg = AgentConfig(
            agent_id=i,
            train=train_mask[i],
            network_type=network_type,
            mlp_hiddens=(64, 32),
            gru_hidden=32,
            gru_num_layers=1,
            learning_rate=3e-4,
            reward_norm=False,
        )
        agent = IPPOAgent(n_actions, seg, cfg)
        configs.append(cfg)
        agents.append(agent)
    return agents, configs


def check_shape(name: str, tensor, expected_shape: tuple) -> bool:
    """检查张量形状，打印错误。"""
    if tensor is None:
        if expected_shape is None:
            return True
        print(f"  ✗ {name}: expected shape {expected_shape}, got None")
        return False
    actual = tuple(tensor.shape)
    ok = actual == expected_shape
    if not ok:
        print(f"  ✗ {name}: expected {expected_shape}, got {actual}")
    return ok


def check_close(name: str, a, b, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    """检查两个张量是否接近。"""
    a_t = torch.as_tensor(a).float()
    b_t = torch.as_tensor(b).float()
    ok = torch.allclose(a_t, b_t, rtol=rtol, atol=atol)
    if not ok:
        diff = (a_t - b_t).abs().max().item()
        print(f"  ✗ {name}: not close, max diff = {diff:.6f}")
    return ok


def test_gae_single_env():
    """测试 GAE 在单环境（0 维 env）下的计算。"""
    print("\n=== Test: GAE (single env) ===")
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    next_value = torch.tensor(0.5)
    next_done = torch.tensor(0.0)

    adv, returns = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)

    # 用最简单的公式验证第一项
    # delta_4 = r_4 + gamma * V_5 - V_4
    delta_4 = 1.0 + 0.99 * 0.5 - 0.5
    expected_adv_4 = delta_4  # gae_lambda^0 * delta_4
    assert check_close("adv[4]", adv[4], expected_adv_4)

    print("  ✓ GAE single-env passed")


def test_gae_parallel_env():
    """测试 GAE 在并行环境下的计算。"""
    print("\n=== Test: GAE (parallel envs) ===")
    # 2 个并行环境, 3 个时间步
    rewards = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])  # (3, 2)
    values = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])  # (3, 2)
    dones = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])  # (3, 2)
    next_value = torch.tensor([0.5, 0.5])  # (2,)
    next_done = torch.tensor([0.0, 0.0])  # (2,)

    adv, returns = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)

    assert check_shape("adv", adv, (3, 2))
    assert check_shape("returns", returns, (3, 2))

    # 手动验证 env=0, step=2: reward=0, nextvalue=0.5, value=0.5
    delta = 0.0 + 0.99 * 0.5 - 0.5
    assert check_close("adv[2,0]", adv[2, 0], delta)
    print("  ✓ GAE parallel-env passed")


def test_gae_with_dones():
    """测试带 episode 终止的 GAE。"""
    print("\n=== Test: GAE with dones ===")
    rewards = torch.tensor([
        [1.0, 1.0],
        [0.0, 0.0],
    ])  # (2, 2)
    values = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    dones = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],  # env=1 在第 1 步终止
    ])
    next_value = torch.tensor([0.5, 0.0])  # env=1 的 next_value 应该是新 episode 的 value
    next_done = torch.tensor([0.0, 1.0])

    adv, returns = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)

    assert check_shape("adv", adv, (2, 2))
    # env=0: 正常; env=1: 由于 done，delta 只到 done 步
    # 不做具体数值验证，仅检查 shape 和不含 NaN
    assert not adv.isnan().any(), "adv contains NaN"
    assert not returns.isnan().any(), "returns contains NaN"
    print("  ✓ GAE with dones passed")


def test_parallel_rollout_mlp():
    """测试 MLP agent 的并行 rollout。"""
    print("\n=== Test: Parallel rollout (MLP) ===")
    n_agents = 2
    n_game_envs = 3
    n_actions = 6
    obs_dim = 45
    num_steps = 8
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.MLP)
    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    # 初始状态
    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents]
    reward_normalizers = [None] * n_agents

    rollouts, gs, rnn_states_out, new_eps = collect_parallel_rollout_ippo(
        agent_configs, agents, env, num_steps, device,
        next_obs, next_done, 0,
        episode_returns, accum_rewards, reward_normalizers,
        rnn_states, step_increment=n_game_envs,
    )

    assert len(rollouts) == n_agents

    for i in range(n_agents):
        r = rollouts[i]
        assert check_shape(f"agent_{i}.obs", r.obs, (num_steps, n_game_envs, obs_dim))
        assert check_shape(f"agent_{i}.actions", r.actions, (num_steps, n_game_envs))
        assert check_shape(f"agent_{i}.logprobs", r.logprobs, (num_steps, n_game_envs))
        assert check_shape(f"agent_{i}.rewards", r.rewards, (num_steps, n_game_envs))
        assert check_shape(f"agent_{i}.dones", r.dones, (num_steps, n_game_envs))
        assert check_shape(f"agent_{i}.values", r.values, (num_steps, n_game_envs))
        assert check_shape(f"agent_{i}.next_obs", r.next_obs, (n_game_envs, obs_dim))
        assert check_shape(f"agent_{i}.next_done", r.next_done, (n_game_envs,))
        assert check_shape(f"agent_{i}.next_value", r.next_value, (n_game_envs,))
        assert r.rnn_states is None

        # actions 应该在有效范围内
        assert r.actions.min() >= 0 and r.actions.max() < n_actions

    # 验证 next_obs / next_done 可以重建回 flat 格式
    next_obs_rebuilt = torch.stack([r.next_obs for r in rollouts], dim=1).reshape(env.num_envs, -1)
    next_done_rebuilt = torch.stack([r.next_done for r in rollouts], dim=1).reshape(env.num_envs)
    assert check_shape("rebuilt next_obs", next_obs_rebuilt, (env.num_envs, obs_dim))
    assert check_shape("rebuilt next_done", next_done_rebuilt, (env.num_envs,))

    print("  ✓ MLP parallel rollout passed")


def test_parallel_rollout_gru():
    """测试 GRU agent 的并行 rollout。"""
    print("\n=== Test: Parallel rollout (GRU) ===")
    n_agents = 2
    n_game_envs = 2
    n_actions = 6
    obs_dim = 45
    num_steps = 8
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.GRU_MLP)
    for agent in agents:
        assert agent.is_recurrent

    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents]
    reward_normalizers = [None] * n_agents

    rollouts, gs, rnn_states_out, new_eps = collect_parallel_rollout_ippo(
        agent_configs, agents, env, num_steps, device,
        next_obs, next_done, 0,
        episode_returns, accum_rewards, reward_normalizers,
        rnn_states, step_increment=n_game_envs,
    )

    for i in range(n_agents):
        r = rollouts[i]
        state_size = agents[i].recurrent_state_size
        assert check_shape(f"agent_{i}.rnn_states", r.rnn_states, (num_steps, n_game_envs, state_size))
        assert check_shape(f"agent_{i}.next_rnn_state", r.next_rnn_state, (n_game_envs, state_size))
        assert r.rnn_states is not None and not r.rnn_states.isnan().any()

    print("  ✓ GRU parallel rollout passed")


def test_parallel_rollout_mixed_train():
    """测试混合可训练/不可训练 agent。"""
    print("\n=== Test: Parallel rollout (mixed train/non-train) ===")
    n_agents = 3
    n_game_envs = 2
    n_actions = 5
    obs_dim = 45
    num_steps = 4
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(
        n_agents, n_actions, seg, NetworkType.MLP,
        train_mask=[True, False, True],
    )
    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents]
    reward_normalizers = [None] * n_agents

    rollouts, gs, rnn_states_out, new_eps = collect_parallel_rollout_ippo(
        agent_configs, agents, env, num_steps, device,
        next_obs, next_done, 0,
        episode_returns, accum_rewards, reward_normalizers,
        rnn_states, step_increment=n_game_envs,
    )

    # 非训练 agent 的 logprobs 和 values 应为 0（默认值）
    assert torch.all(rollouts[1].logprobs == 0.0), "non-train agent logprobs should be 0"
    assert torch.all(rollouts[1].values == 0.0), "non-train agent values should be 0"
    assert rollouts[1].next_value is None, "non-train agent next_value should be None"

    print("  ✓ Mixed train/non-train rollout passed")


def test_recurrent_training():
    """测试 GRU agent 的完整训练更新。"""
    print("\n=== Test: GRU training update ===")
    n_agents = 1
    n_game_envs = 2
    n_actions = 5
    obs_dim = 45
    num_steps = 16
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.GRU_MLP)
    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents]
    reward_normalizers = [None] * n_agents

    rollouts, gs, rnn_states_out, new_eps = collect_parallel_rollout_ippo(
        agent_configs, agents, env, num_steps, device,
        next_obs, next_done, 0,
        episode_returns, accum_rewards, reward_normalizers,
        rnn_states, step_increment=n_game_envs,
    )

    # 训练更新
    args = IppoArgs()
    args.num_agents = n_agents
    args.num_envs = env.num_envs
    args.num_game_envs = n_game_envs
    args.batch_size = env.num_envs * num_steps
    args.num_minibatches = 2
    args.minibatch_size = args.batch_size // 2
    args.recurrent_seq_len = 8
    args.update_epochs = 1
    args.norm_adv = True
    args.clip_vloss = True
    args.target_kl = None

    optimizer = optim.Adam(agents[0].parameters(), lr=3e-4)

    metrics = train_agent_update(
        agents[0], optimizer, rollouts[0], agent_configs[0], args, device
    )

    assert "pg_loss" in metrics, "missing pg_loss"
    assert "v_loss" in metrics, "missing v_loss"
    assert "entropy" in metrics, "missing entropy"
    assert "approx_kl" in metrics, "missing approx_kl"
    assert not np.isnan(metrics["pg_loss"]), "pg_loss is NaN"
    assert not np.isnan(metrics["v_loss"]), "v_loss is NaN"

    print(f"  Metrics: pg_loss={metrics['pg_loss']:.4f}, v_loss={metrics['v_loss']:.4f}, "
          f"approx_kl={metrics['approx_kl']:.4f}, entropy={metrics['entropy']:.4f}")
    print("  ✓ GRU training update passed")


def test_mlp_training():
    """测试 MLP agent 的完整训练更新。"""
    print("\n=== Test: MLP training update ===")
    n_agents = 1
    n_game_envs = 1  # 单环境: 基本路径
    n_actions = 5
    obs_dim = 45
    num_steps = 32
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.MLP)
    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [agent.get_initial_state(n_game_envs, device) for agent in agents]
    reward_normalizers = [None] * n_agents

    rollouts, gs, rnn_states_out, new_eps = collect_parallel_rollout_ippo(
        agent_configs, agents, env, num_steps, device,
        next_obs, next_done, 0,
        episode_returns, accum_rewards, reward_normalizers,
        rnn_states, step_increment=n_game_envs,
    )

    args = IppoArgs()
    args.num_agents = n_agents
    args.num_envs = env.num_envs
    args.num_game_envs = n_game_envs
    args.batch_size = env.num_envs * num_steps
    args.num_minibatches = 2
    args.minibatch_size = args.batch_size // 2
    args.update_epochs = 1
    args.norm_adv = True
    args.clip_vloss = True
    args.target_kl = None

    optimizer = optim.Adam(agents[0].parameters(), lr=3e-4)

    metrics = train_agent_update(
        agents[0], optimizer, rollouts[0], agent_configs[0], args, device
    )

    assert not np.isnan(metrics["pg_loss"]), "pg_loss is NaN"
    assert not np.isnan(metrics["v_loss"]), "v_loss is NaN"

    print(f"  Metrics: pg_loss={metrics['pg_loss']:.4f}, v_loss={metrics['v_loss']:.4f}, "
          f"approx_kl={metrics['approx_kl']:.4f}, entropy={metrics['entropy']:.4f}")
    print("  ✓ MLP single-env training passed (baseline)")


def test_rnn_state_reset():
    """测试 GRU 隐藏态在 done 时的重置逻辑。"""
    print("\n=== Test: RNN state reset on done ===")
    n_agents = 1
    n_game_envs = 2
    n_actions = 5
    obs_dim = 45
    num_steps = 8
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.GRU_MLP)
    agent = agents[0]
    state_size = agent.recurrent_state_size

    # 手工构造数据: env=0 正常, env=1 在第 2 步 done
    obs = torch.randn(num_steps, n_game_envs, obs_dim, device=device)
    actions = torch.randint(0, n_actions, (num_steps, n_game_envs), device=device)
    logprobs = torch.zeros(num_steps, n_game_envs, device=device)
    rewards = torch.zeros(num_steps, n_game_envs, device=device)
    dones = torch.zeros(num_steps, n_game_envs, device=device)
    dones[2, 1] = 1.0  # env=1 done at step 2
    values = torch.zeros(num_steps, n_game_envs, device=device)
    rnn_states_tensor = torch.zeros(num_steps, n_game_envs, state_size, device=device)
    # 模拟: env=1 在 step 3 的 rnn_state 应被重置
    rnn_states_tensor[3, 1] = torch.zeros(state_size)  # 重置为 0

    next_obs = obs[-1]  # (n_game_envs, obs_dim)
    next_done = torch.zeros(n_game_envs, device=device)
    next_value = torch.zeros(n_game_envs, device=device)
    next_rnn_state = agent.get_initial_state(n_game_envs, device)

    rollout = RolloutData(
        obs=obs, actions=actions, logprobs=logprobs,
        rewards=rewards, dones=dones, values=values,
        next_obs=next_obs, next_done=next_done,
        next_value=next_value, rnn_states=rnn_states_tensor,
        next_rnn_state=next_rnn_state,
    )

    # 使用 evaluate_recurrent_sequences 验证
    seq_starts = np.array([0])
    seq_ends = np.array([num_steps])
    seq_envs = np.array([1])  # 只测试 env=1

    try:
        idxs, new_lp, new_ent, new_val = evaluate_recurrent_sequences(
            agent, rollout, seq_starts, seq_ends, seq_envs, device
        )
        # 不应崩溃
        assert not new_lp.isnan().any()
        print("  ✓ RNN state reset on done passed")
    except Exception as e:
        print(f"  ✗ RNN state reset failed: {e}")
        raise


def test_end_to_end_two_updates():
    """端到端测试: 两个连续 update（验证状态传递）。"""
    print("\n=== Test: End-to-end two updates ===")
    n_agents = 2
    n_game_envs = 2
    n_actions = 5
    obs_dim = 45
    num_steps = 8
    device = torch.device("cpu")

    seg = make_test_seg(obs_dim)
    agents, agent_configs = make_test_agents(n_agents, n_actions, seg, NetworkType.MLP)
    env = MockDiscreteEnv(n_agents * n_game_envs, obs_dim, n_actions)

    args = IppoArgs()
    args.num_agents = n_agents
    args.num_envs = env.num_envs
    args.num_game_envs = n_game_envs
    args.batch_size = env.num_envs * num_steps
    args.num_minibatches = 1
    args.minibatch_size = args.batch_size
    args.recurrent_seq_len = num_steps
    args.update_epochs = 1
    args.norm_adv = True
    args.clip_vloss = True
    args.target_kl = None

    optimizers = [optim.Adam(a.parameters(), lr=3e-4) for a in agents]
    reward_normalizers = [None] * n_agents

    next_obs_array, _ = env.reset()
    next_obs = torch.tensor(next_obs_array, dtype=torch.float32, device=device)
    next_done = torch.zeros(env.num_envs, device=device)

    episode_returns = [deque(maxlen=20) for _ in range(n_agents)]
    accum_rewards = np.zeros((n_agents, n_game_envs), dtype=np.float64)
    rnn_states = [a.get_initial_state(n_game_envs, device) for a in agents]

    for update_idx in range(2):
        rollouts, _, rnn_states, _ = collect_parallel_rollout_ippo(
            agent_configs, agents, env, num_steps, device,
            next_obs, next_done, update_idx * num_steps * n_game_envs,
            episode_returns, accum_rewards, reward_normalizers,
            rnn_states, step_increment=n_game_envs,
        )

        next_obs = torch.stack(
            [r.next_obs for r in rollouts], dim=1
        ).reshape(env.num_envs, -1)
        next_done = torch.stack(
            [r.next_done for r in rollouts], dim=1
        ).reshape(env.num_envs)

        for i in range(n_agents):
            if agent_configs[i].train:
                metrics = train_agent_update(
                    agents[i], optimizers[i], rollouts[i],
                    agent_configs[i], args, device,
                )
                assert not np.isnan(metrics["pg_loss"]), f"update {update_idx} agent {i}: NaN loss"

    print("  ✓ End-to-end two updates passed")


def test_reward_normalizer_parallel():
    """测试 RewardNormalizer 的批量接口。"""
    print("\n=== Test: RewardNormalizer parallel ===")
    norm = RewardNormalizer(clip=5.0)

    # 单步更新
    norm.update(1.0)
    norm.update(-1.0)

    # 批量归一化
    rewards = np.array([1.0, -1.0, 5.0, -5.0], dtype=np.float32)
    normed = norm.normalize_array(rewards)

    # 应在 [-5, 5] 范围内
    assert np.all(normed >= -5.0) and np.all(normed <= 5.0), "clip failed"
    print(f"  Normalized: {normed}")

    # 批量更新
    norm.update_array(np.array([2.0, 3.0, -2.0, -3.0], dtype=np.float32))
    assert norm.count > 2, "count should increase after batch update"
    print("  ✓ RewardNormalizer parallel passed")


# ─── 主入口 ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("IPPO Parallel Training Test Suite")
    print("=" * 60)

    tests = [
        ("GAE single env", test_gae_single_env),
        ("GAE parallel env", test_gae_parallel_env),
        ("GAE with dones", test_gae_with_dones),
        ("MLP parallel rollout", test_parallel_rollout_mlp),
        ("GRU parallel rollout", test_parallel_rollout_gru),
        ("Mixed train/non-train rollout", test_parallel_rollout_mixed_train),
        ("MLP training (baseline)", test_mlp_training),
        ("GRU training update", test_recurrent_training),
        ("RNN state reset", test_rnn_state_reset),
        ("End-to-end two updates", test_end_to_end_two_updates),
        ("RewardNormalizer parallel", test_reward_normalizer_parallel),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            import traceback
            print(f"\n  ✗ ERROR in {name}:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("Failures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
