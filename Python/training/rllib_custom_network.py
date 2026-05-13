

#类型定义
from __future__ import annotations

import argparse
import os
import sys
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import ray
import torch
import torch.nn as nn
import yaml
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import get_checkpoint_info

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv

# 确保 Ray worker 进程也能找到同目录的本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from godot_env_wrapper import ObsSegmentDims


class NetworkType(str, Enum):
    """支持的网络架构类型。"""
    DEFAULT = "default"
    SEGMENTED_MLP = "segmented_mlp"
    GRU_MLP = "gru_mlp"

@dataclass
class NetworkDefaults:
    """分段网络隐层维度的内置默认值。

    当 YAML model_config 中未显式配置某段的 hiddens 时使用。
    所有属性可被 YAML 配置覆盖。
    """
    self_hidden: int = 48
    player_hidden: int = 64
    ball_hidden: int = 64
    enemy_hidden: int = 64
    map_hidden: int = 64
    trunk_hiddens: tuple = (128, 64)

    @staticmethod
    def _as_tuple(value, default):
        """将 None / int / iterable 统一为 tuple。"""
        if value is None:
            value = default
        if isinstance(value, int):
            return (value,)
        return tuple(int(v) for v in value)

    def get_hiddens(self, config: dict, plural_key: str, singular_key: str,
                    default: tuple) -> tuple:
        """从 config dict 中提取隐层维度"""
        if plural_key in config:
            return self._as_tuple(config[plural_key], default)
        if singular_key in config:
            return self._as_tuple(config[singular_key], default)
        return self._as_tuple(None, default)


#网络构建块
def ortho_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """正交权重初始化 (用于 Linear 层)。"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def init_gru_weights(gru: nn.GRU):
    """GRU 权重初始化: Xavier (input) + Orthogonal (hidden) + Zero (bias)。"""
    for name, param in gru.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(param)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0.0)
    return gru


def make_mlp(input_dim: int, hidden_sizes: tuple,
            final_std: float = np.sqrt(2)) -> tuple[nn.Module, int]:
    """构建 MLP 序列: Linear + ReLU 堆叠。

    Returns:
        (nn.Sequential | nn.Identity, output_dim) — 空 hidden_sizes 时返回 Identity
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(ortho_init(nn.Linear(in_dim, hidden_size), std=final_std))
        layers.append(nn.ReLU())
        in_dim = hidden_size
    net = nn.Sequential(*layers) if layers else nn.Identity()
    return net, in_dim # 输出维度


class _TemporalGRU(nn.Module):
    """GRU 配置容器 + 初始状态工厂 + packed sequence 前向。

    forward() 接收 PackedSequence (由 pack_padded_sequence 生成),
    返回 (packed_output, h_n)。

      - get_initial_state() → [zeros(L*H)]  (1D, 无 batch 维)
      - forward() 的 state 分量为 (B, L*H) 2D 张量
      - 内部通过 view(B, L, H).transpose(0, 1) / transpose+reshape 与 nn.GRU 的 (L, B, H) 约定互转
      - L=1 时 L*H = H，行为与单层完全一致
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = init_gru_weights(
            nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        )
        self.output_dim = hidden_size

    def get_initial_state(self) -> list:
        """RLlib: 返回 1D 初始隐藏态 (L*H,), 无 batch 维。

        所有层展平为单一张量, RLlib 广播时自动扩展为 (B, L*H)。
        L=1 时等价于 (H,), 与单层路径完全一致。
        """
        weight = next(self.parameters())
        return [weight.new_zeros(self.num_layers * self.hidden_size)]

    def forward(self, packed_input, h0):
        """将 PackedSequence 送入 GRU，返回 (packed_output, h_n)。

        Args:
            packed_input: PackedSequence (由 pack_padded_sequence 生成)
            h0: (L, B, H) 初始隐藏态

        Returns:
            (packed_output, h_n):
              - packed_output: PackedSequence
              - h_n: (L, B, H) 最终隐藏态
        """
        return self.gru(packed_input, h0)


def parse_network_type(value) -> NetworkType:
    """将字符串或枚举值解析为 NetworkType。"""
    if isinstance(value, NetworkType):
        return value
    text = str(value or NetworkType.SEGMENTED_MLP.value).lower()
    for nt in NetworkType:
        if text in (nt.value, nt.name.lower()):
            return nt
    raise ValueError(
        f"Unsupported network_type={value!r}. "
        f"Use one of {[t.value for t in NetworkType]}."
    )


#网络构建块
class SegmentedObsHelper:
    """观测分段辅助工具: 封装 ObsSegmentDims, 提供统一的分段/拼接接口。
    """
    def __init__(self, seg_dims: ObsSegmentDims):
        self._d = seg_dims

    @property
    def dims(self) -> ObsSegmentDims:
        return self._d

    def total_dim(self) -> int:
        d = self._d
        return d.self_dim + d.player_dim + d.ball_dim + d.enemy_dim + d.map_dim

    def split(self, obs: torch.Tensor) -> tuple:
        """将扁平观测 (batch, total_dim) 拆解为 (self, player, ball, enemy, map) 五段。
        每段保留 batch 维, 返回 5 个 (batch, seg_dim) 张量。
        """
        d = self._d
        i = 0
        s = obs[:, i: i + d.self_dim];   i += d.self_dim
        p = obs[:, i: i + d.player_dim]; i += d.player_dim
        b = obs[:, i: i + d.ball_dim];   i += d.ball_dim
        e = obs[:, i: i + d.enemy_dim];  i += d.enemy_dim
        m = obs[:, i: i + d.map_dim]
        return s, p, b, e, m


class CustomSegmentedModel(TorchModelV2, nn.Module):
    """RLlib 自定义分段/GRU-MLP 模型。

    支持两种网络架构:
      - SEGMENTED_MLP: 5 段独立 MLP (self/player/ball/enemy/map) -> concat -> trunk
      - GRU_MLP:     时序段 (self/player/enemy/map) -> GRU + BALL MLP -> concat -> trunk

    通过 custom_model_config 注入:
      - network_type: "segmented_mlp" | "gru_mlp"
      - obs_seg_dims: ObsSegmentDims 实例 (观测分段维度)
      - 各段的 hiddens 配置
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,model_config, name)
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {}) or {}
        self._network_type = parse_network_type(custom.get("network_type", NetworkType.SEGMENTED_MLP.value))
        self._defaults = NetworkDefaults()

        # 观测分段辅助 — 通过 custom_model_config 注入
        seg_dims = custom.get("obs_seg_dims")
        if seg_dims is None:
            raise ValueError(
                "obs_seg_dims must be provided in custom_model_config. "
                "Use build_rllib_model_config() to construct the model config."
            )
        self._obs = SegmentedObsHelper(seg_dims)
        flat_obs_shape = getattr(obs_space, "shape", None)
        if flat_obs_shape is not None and len(flat_obs_shape) > 0:
            flat_obs_dim = int(np.prod(flat_obs_shape))
            if flat_obs_dim != self._obs.total_dim():
                raise ValueError(
                    f"Observation dim mismatch: obs_space has {flat_obs_dim}, "
                    f"but obs_seg_dims total is {self._obs.total_dim()}."
                )

        # 按网络类型构建特征提取网络
        self._build_network(custom)

        # Actor / Critic 头 — 共享 trunk 输出
        self._actor = ortho_init(
            nn.Linear(self._features_dim, num_outputs), std=0.01
        )
        self._critic = ortho_init(
            nn.Linear(self._features_dim, 1), std=1.0
        )

        # RLlib 要求声明的属性
        self._features: Optional[torch.Tensor] = None

    # 网络构建
    def _build_network(self, custom: dict):
        """工厂方法: 按 network_type 分发到对应的构建方法。"""
        if self._network_type == NetworkType.SEGMENTED_MLP:
            self._build_segmented_mlp(custom)
        elif self._network_type == NetworkType.GRU_MLP:
            self._build_gru_mlp(custom)
        else:
            raise ValueError(f"Unsupported network_type={self._network_type}")

    def _build_segmented_mlp(self, config: dict):
        """5 段独立 MLP 子网络 -> concat -> trunk -> actor/critic 头。"""
        d = self._defaults
        seg = self._obs.dims

        self.self_net, self_out   = make_mlp(seg.self_dim,   d.get_hiddens(config, "self_hiddens",   "self_hidden",   (d.self_hidden,)))
        self.player_net, plr_out  = make_mlp(seg.player_dim, d.get_hiddens(config, "player_hiddens", "player_hidden", (d.player_hidden,)))
        self.ball_net, ball_out   = make_mlp(seg.ball_dim,   d.get_hiddens(config, "ball_hiddens",   "ball_hidden",   (d.ball_hidden,)))
        self.enemy_net, enemy_out = make_mlp(seg.enemy_dim,  d.get_hiddens(config, "enemy_hiddens",  "enemy_hidden",  (d.enemy_hidden,)))
        self.map_net, map_out     = make_mlp(seg.map_dim,    d.get_hiddens(config, "map_hiddens",    "map_hidden",    (d.map_hidden,)))

        fused_dim = self_out + plr_out + ball_out + enemy_out + map_out
        trunk_hiddens = d.get_hiddens(config, "trunk_hiddens", "trunk_hidden", d.trunk_hiddens)
        self.trunk, self._features_dim = make_mlp(fused_dim, trunk_hiddens)

    def _build_gru_mlp(self, config: dict):
        """GRU-MLP 混合网络。

        时序段 (SELF/PLAYER/ENEMY/MAP) -> 原始特征concat -> LayerNorm -> GRU
        非时序段 (BALL) -> 独立 MLP -> concat -> trunk -> actor/critic

        观测槽位顺序固定, 这使得隐藏态不错位。
        """
        d = self._defaults
        seg = self._obs.dims

        gru_hidden = config.get("gru_hidden", 128)
        gru_layers = config.get("gru_num_layers", 1)
        use_layernorm = config.get("gru_input_layernorm", True)

        # GRU 输入
        gru_input_dim = seg.self_dim + seg.player_dim + seg.enemy_dim + seg.map_dim
        self._gru = _TemporalGRU(gru_input_dim, gru_hidden, num_layers=gru_layers)
        self._gru_ln = nn.LayerNorm(gru_input_dim) if use_layernorm else nn.Identity()

        # Ball 独立 MLP
        ball_hiddens = d.get_hiddens(config, "ball_hiddens", "ball_hidden", (d.ball_hidden,))
        self.ball_net, ball_out_dim = make_mlp(seg.ball_dim, ball_hiddens)

        fused_dim = self._gru.output_dim + ball_out_dim
        trunk_hiddens = d.get_hiddens(config, "trunk_hiddens", "trunk_hidden", d.trunk_hiddens)
        self.trunk, self._features_dim = make_mlp(fused_dim, trunk_hiddens)

    # RLlib 接口
    @override(TorchModelV2)
    def get_initial_state(self):
        """RLlib: 返回不含 batch 维的初始隐藏态列表。

        RLlib 约定: 每个元素为 1D 张量 (H,), RLlib 广播时自动扩展为 (B, H)。
        非 GRU 模型返回空列表。
        """
        if self._network_type == NetworkType.GRU_MLP:
            return self._gru.get_initial_state()  # [zeros(H)]
        return []

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """RLlib 前向: 返回 (action_logits, state_outs)。

        GRU_MLP 使用 add_time_dimension 恢复时域, 遵循 RLlib 标准状态约定:
          - state 各分量为 (B, H) 2D 张量
          - 输出 logits 为 (B*T, num_outputs) 扁平格式
        """
        obs = input_dict["obs_flat"].float()
        if self._network_type == NetworkType.GRU_MLP:
            return self._forward_gru_mlp(obs, state, seq_lens)
        else:
            return self._forward_segmented_mlp(obs, state)

    @override(TorchModelV2)
    def value_function(self):
        """RLlib 调用此方法获取 V(s), 返回 1D (batch,) 张量。"""
        return self._critic(self._features).squeeze(-1)

    # 前向传播
    def _forward_segmented_mlp(self, obs: torch.Tensor, state):
        """SEGMENTED_MLP 前向: 5 段独立 MLP -> concat -> trunk。"""
        s, p, b, e, m = self._obs.split(obs)
        fused = torch.cat([
            self.self_net(s),
            self.player_net(p),
            self.ball_net(b),
            self.enemy_net(e),
            self.map_net(m),
        ], dim=1)
        self._features = self.trunk(fused)
        logits = self._actor(self._features)
        return logits, state

    def _forward_gru_mlp(self, obs: torch.Tensor, state, seq_lens):
        """GRU_MLP 前向: 使用 add_time_dimension 恢复时域后处理。

        统一处理 rollout (T=1) 和 training (T>1) 两种模式,
        不再区分单步/序列批处理分支。

        RLlib 状态约定:
          - state[0]: (B, L*H) 2D 张量 (num_layers 展平)
          - 返回的 state_out[0]: (B, L*H) 2D 张量

        内部通过 view(B, L, H).transpose(0, 1) 与 nn.GRU 的 (L, B, H) 约定互转。

        Args:
            obs: 扁平观测 (B*T, total_dim)
            state: [h_prev], h_prev shape (B, L*H)
            seq_lens: (B,) 每条序列的有效长度

        Returns:
            (logits, [h_new]):
              - logits: (B*T, num_outputs) 扁平格式
              - h_new:  (B, L*H) — 每序列最终隐藏态 (所有层展平)
        """
        h_prev = state[0]  # (B, L*H) — RLlib 标准 2D 格式
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        gru = self._gru
        L, H = gru.num_layers, gru.hidden_size

        if seq_lens is None:
            raise ValueError(
                "seq_lens must be provided for GRU_MLP model. "
                "Ensure max_seq_len is set in model config."
            )

        # RLlib 扁平观测 → 恢复时域 (B, T, obs_dim)
        time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            obs, seq_lens=seq_lens, framework="torch", time_major=time_major,
        )
        # time_major 时 transpose 为 batch-major 以便后续统一处理
        if time_major:
            inputs = inputs.transpose(0, 1)
        B, T = inputs.size(0), inputs.size(1)

        # 分段拆解
        obs_flat = inputs.reshape(B * T, -1)
        s, p, b, e, m = self._obs.split(obs_flat)

        # 时序段 concat -> LayerNorm -> (B, T, gru_input_dim)
        gru_input = torch.cat([
            s.reshape(B, T, -1),
            p.reshape(B, T, -1),
            e.reshape(B, T, -1),
            m.reshape(B, T, -1),
        ], dim=-1)
        gru_input = self._gru_ln(gru_input)

        # RLlib (B, L*H) -> PyTorch GRU (L, B, H)
        # 必须先 view(B, L, H) 再 transpose — 直接 view(L, B, H) 会打乱 batch/layer 语义
        h0 = h_prev.view(B, L, H).transpose(0, 1).contiguous()

        # 处理 seq_lens: 转为 CPU long tensor并 clamp
        if isinstance(seq_lens, torch.Tensor):
            lengths = seq_lens.detach().to(dtype=torch.long, device="cpu")
        else:
            lengths = torch.as_tensor(seq_lens, dtype=torch.long)
        lengths = lengths.clamp(min=1, max=T)

        # Packed GRU forward — 正确处理变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, h_new = self._gru(packed, h0)
        gru_feats, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T,
        )

        # Ball MLP (非时序段, 逐帧独立处理)
        ball_feats = self.ball_net(b.reshape(B * T, -1)).reshape(B, T, -1)

        # 融合 + trunk + heads
        fused = torch.cat([gru_feats, ball_feats], dim=-1)  # (B, T, fused_dim)
        self._features = self.trunk(fused.reshape(B * T, -1))  # (B*T, features_dim)
        logits = self._actor(self._features)  # (B*T, num_outputs)

        # PyTorch GRU (L, B, H) → RLlib (B, L*H)
        h_new = h_new.transpose(0, 1).reshape(B, L * H)
        return logits, [h_new]


# 模型注册
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("custom_segmented_model", CustomSegmentedModel)


#配置桥接
def build_rllib_model_config(exp: dict) -> dict:
    """从 YAML exp 配置构建 RLlib model config dict。

    读取 exp["config"]["model"] (字符串) 作为网络类型,
    exp["model_config"][type] 作为对应的结构参数,
    返回 RLlib 兼容的 model config dict。
    """
    network_type = str(exp["config"]["model"]).strip().lower()
    type_params = exp.get("model_config", {}).get(network_type, {}) #结构参数

    if network_type == NetworkType.DEFAULT.value:
        # RLlib 内置网络: 直接传递 model 参数
        return dict(type_params)

    # 加载观测分段维度
    tres_path = exp.get("obs_seg_config_path", r"D:\schoolTour\softwares\multi-agent-gameplay\godot-game\configs\game_config.tres")
    obs_seg_dims = ObsSegmentDims.from_config(tres_path)

    # 自定义网络: 注入 custom_model + custom_model_config
    model = {
        "custom_model": "custom_segmented_model",
        "custom_model_config": {
            "network_type": network_type,
            "obs_seg_dims": obs_seg_dims,
            **dict(type_params),
        },
    }
    if network_type == NetworkType.GRU_MLP.value:
        model["max_seq_len"] = int(type_params.get("max_seq_len", 32))
    return model


#环境工具
def convert_godot_obs(obs):
    """将 Godot 返回的 list/dict 观测转换为 numpy float32 array。

    支持两种格式:
      - 多智能体: {agent_id: {"obs": [..values..]}}
      - 单智能体: [..values..]
    """
    if isinstance(obs, dict):
        result = {}
        for k, v in obs.items():
            if isinstance(v, dict) and "obs" in v:#多智能体
                arr = np.array(v["obs"], dtype=np.float32)
                result[k] = {"obs": arr}
            else:
                result[k] = np.array(v, dtype=np.float32)
        return result
    return np.array(obs, dtype=np.float32)

def wrap_pettingzoo_obs(env):
    """包装 PettingZoo env, 确保观测为 float32 numpy array。

    通过猴子补丁替换 reset/step 方法, 在返回值上应用 convert_godot_obs。
    """
    original_reset = env.reset
    original_step = env.step

    def reset(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        return convert_godot_obs(obs), info

    def step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        return convert_godot_obs(obs), reward, terminated, truncated, info

    env.reset = reset
    env.step = step

    return env

def _parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_file", default="D:\\schoolTour\\softwares\\multi-agent-gameplay\\Python\\training\\rllib_s1_config.yaml",
        type=str, help="The yaml config file",
    )
    parser.add_argument(
        "--restore", default=None, type=str,
        help="Deprecated alias for --resume_experiment.",
    )
    parser.add_argument(
        "--resume_experiment", default=None, type=str,
        help="Tune experiment directory for crash recovery, e.g. logs/rllib/PPO_....",
    )
    parser.add_argument(
        "--resume_checkpoint", default=None, type=str,
        help=(
            "RLlib checkpoint to initialize a new curriculum stage. "
            "If an experiment/trial directory is provided, the latest checkpoint_* "
            "under it will be used."
        ),
    )
    parser.add_argument(
        "--experiment_dir", default="logs/rllib", type=str,
        help="The name of the experiment directory, used to store logs.",
    )
    args, _extras = parser.parse_known_args()
    return args


class RLLibTrainingPipeline:
    """RLlib 训练流水线
    """
    def __init__(
        self,
        config_path: str,
        experiment_dir: str,
        resume_experiment: Optional[str] = None,
        resume_checkpoint: Optional[str] = None,
        restore: Optional[str] = None,
    ):
        self._config_path = config_path
        self._experiment_dir = experiment_dir
        if restore and resume_experiment:
            raise ValueError("Use only one of --restore or --resume_experiment.")
        if restore:
            print("[Resume] --restore is deprecated; treating it as --resume_experiment.")
            resume_experiment = restore
        if resume_experiment and resume_checkpoint:
            raise ValueError("Use only one of --resume_experiment or --resume_checkpoint.")

        self._resume_experiment = resume_experiment
        self._resume_checkpoint = resume_checkpoint

        # 执行过程中填充
        self._exp: Optional[dict] = None # 配置
        self._is_multiagent: bool = False
        self._policy_names: Optional[list] = None
        self._num_envs: Optional[int] = None

    @staticmethod
    def _as_abs_path(path: str) -> pathlib.Path:
        p = pathlib.Path(path).expanduser()
        if not p.is_absolute():
            p = pathlib.Path.cwd() / p
        return p.resolve()

    @staticmethod
    def _looks_like_tune_experiment(path: pathlib.Path) -> bool:
        return (
            (path / "tuner.pkl").exists()
            or any(path.glob("experiment_state-*.json"))
        )

    @staticmethod
    def _looks_like_rllib_checkpoint(path: pathlib.Path) -> bool:
        if path.name.startswith("checkpoint_"):
            return True
        return any(
            (path / marker).exists()
            for marker in (
                "algorithm_state.pkl",
                "rllib_checkpoint.json",
                "checkpoint.pkl",
                "metadata.json",
            )
        )

    def _resolve_experiment_path(self, path: str) -> str:
        exp_path = self._as_abs_path(path)
        if not exp_path.exists():
            raise FileNotFoundError(f"Resume experiment path does not exist: {exp_path}")
        if not exp_path.is_dir() or not self._looks_like_tune_experiment(exp_path):
            raise ValueError(
                "--resume_experiment must point to a Tune experiment directory "
                f"(the folder containing tuner.pkl): {exp_path}"
            )
        return str(exp_path)

    def _resolve_checkpoint_path(self, path: str) -> str:
        checkpoint_path = self._as_abs_path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint path does not exist: {checkpoint_path}")
        if checkpoint_path.is_file():
            return str(checkpoint_path)
        if self._looks_like_rllib_checkpoint(checkpoint_path):
            return str(checkpoint_path)

        candidates = [
            p for p in checkpoint_path.rglob("checkpoint_*")
            if p.is_dir() and self._looks_like_rllib_checkpoint(p)
        ]
        if not candidates:
            raise ValueError(
                "--resume_checkpoint must point to an RLlib checkpoint directory "
                "or a folder containing checkpoint_* directories: "
                f"{checkpoint_path}"
            )

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[Resume] Using latest checkpoint under {checkpoint_path}: {latest}")
        return str(latest)

    @staticmethod
    def _make_checkpoint_config(exp: dict) -> train.CheckpointConfig:
        num_to_keep = exp.get("num_checkpoints_to_keep")
        if num_to_keep is not None:
            num_to_keep = int(num_to_keep)
        return train.CheckpointConfig(
            checkpoint_frequency=int(exp.get("checkpoint_frequency", 0)),
            checkpoint_at_end=bool(exp.get("checkpoint_at_end", True)),
            num_to_keep=num_to_keep, #  设置保留的检查点数量
        )

    @staticmethod
    def _load_policy_weights(checkpoint_path: str, policy_mapping_fn=None) -> dict:
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        checkpoint_state = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info=checkpoint_info,
            policy_mapping_fn=policy_mapping_fn,
        )
        worker_state = checkpoint_state.get("worker") or {}
        policy_states = worker_state.get("policy_states") or {}
        weights = { #  提取每个策略对应的权重
            policy_id: policy_state["weights"]
            for policy_id, policy_state in policy_states.items()
            if "weights" in policy_state
        }
        if not weights:
            raise ValueError(
                "No policy weights found in RLlib checkpoint: "
                f"{checkpoint_path}"
            )
        return weights

    @staticmethod
    def _make_restore_callbacks(
        checkpoint_path: str,
        base_callbacks=None,
        policy_mapping_fn=None,
    ):
        base_callbacks = base_callbacks or DefaultCallbacks #  如果没有提供基础回调类，则使用默认的DefaultCallbacks
        if not isinstance(base_callbacks, type):
            raise TypeError(
                "RLlib callbacks must be a callback class when using --resume_checkpoint."
            )

        class RestoreFromCheckpointCallbacks(base_callbacks):
            def on_algorithm_init(self, *, algorithm, **kwargs):
                super().on_algorithm_init(algorithm=algorithm, **kwargs)
                checkpoint_weights = RLLibTrainingPipeline._load_policy_weights(
                    checkpoint_path,
                    policy_mapping_fn=policy_mapping_fn,
                )
                weights_to_load = {}
                for policy_id, weights in checkpoint_weights.items():
                    if algorithm.get_policy(policy_id) is None:
                        print(
                            f"[Resume] Skipping checkpoint policy {policy_id!r}; "
                            "it is not present in the current stage."
                        )
                        continue
                    weights_to_load[policy_id] = weights

                if not weights_to_load:
                    raise ValueError(
                        "None of the checkpoint policies exist in the current "
                        "stage config."
                    )

                print(
                    "[Resume] Loading policy weights from "
                    f"{checkpoint_path}: {sorted(weights_to_load)}"
                )
                algorithm.set_weights(weights_to_load)
                if hasattr(algorithm, "env_runner_group"):
                    algorithm.env_runner_group.sync_weights(
                        policies=list(weights_to_load),
                        from_worker_or_learner_group=(
                            algorithm.env_runner_group.local_env_runner
                        ),
                    )

        return RestoreFromCheckpointCallbacks

    #加载配置
    def _load_config(self) -> dict:
        """加载 YAML 配置, 构建 RLlib model config, 返回完整 exp dict。"""
        with open(self._config_path, encoding="utf-8") as f:
            exp = yaml.safe_load(f)

        self._is_multiagent = exp["env_is_multiagent"]

        # 桥接: YAML model 字符串 -> RLlib model config dict
        exp["config"]["model"] = build_rllib_model_config(exp)

        return exp

    #创建环境
    def _create_env_creator(self) -> Callable:
        """创建 Ray 注册用的 env_creator 工厂函数。

        Returns:
            env_creator(env_config) -> Ray env instance
        """
        exp = self._exp
        is_multiagent = self._is_multiagent

        def env_creator(env_config):
            index = env_config.worker_index* exp["config"]["num_envs_per_env_runner"]+ env_config.vector_index
            port = index + GodotEnv.DEFAULT_PORT
            seed = index
            if is_multiagent:
                pz_env = GDRLPettingZooEnv(
                    config=env_config, port=port, seed=seed,
                    show_window=env_config.get("show_window", False),
                )
                pz_env = wrap_pettingzoo_obs(pz_env)
                return ParallelPettingZooEnv(pz_env)
            else:
                return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

        return env_creator

    def _make_temp_env(self) -> None:
        """创建临时环境以获取 policy_names (多智能体) 或 num_envs (单智能体)。"""
        env_config = self._exp["config"]["env_config"]

        if self._is_multiagent:
            print("Starting a temporary multi-agent env to get the policy names")
            tmp_env = GDRLPettingZooEnv(config=env_config, show_window=False)
            self._policy_names = tmp_env.agent_policy_names
            print("Policy names for each Agent (AIController) set in the "
                  "Godot Environment", self._policy_names)
        else:
            print("Starting a temporary env to get the number of envs and "
                  "auto-set the num_envs_per_worker config value")
            tmp_env = GodotEnv(env_path=env_config["env_path"], show_window=False)
            self._num_envs = tmp_env.num_envs

        tmp_env.close()

    #配置多智能体 / 单智能体
    def _configure_agent_mode(self) -> None:
        """根据 is_multiagent 注入 policies 或 num_envs_per_env_runner。"""
        if self._is_multiagent:
            def policy_mapping_fn(agent_id: int, _episode=None, _worker=None, **_kwargs) -> str:
                return self._policy_names[agent_id]

            self._exp["config"]["multiagent"] = {
                "policies": {
                    pn: PolicySpec() for pn in self._policy_names
                },
                "policy_mapping_fn": policy_mapping_fn,
            }
        else:
            self._exp["config"]["num_envs_per_env_runner"] = self._num_envs

    #训练
    def _run_training(self) -> tune.ResultGrid:
        """执行 RLlib 训练 (新建或恢复)。"""
        exp = self._exp

        if self._resume_experiment:#恢复单次训练
            experiment_path = self._resolve_experiment_path(self._resume_experiment)
            tuner = tune.Tuner.restore(
                trainable=exp["algorithm"],
                path=experiment_path,
                resume_unfinished=True,
            )
        else:
            if self._resume_checkpoint:#用已有模型开始训练
                checkpoint_path = self._resolve_checkpoint_path(self._resume_checkpoint)
                base_callbacks = exp["config"].get("callbacks")
                policy_mapping_fn = exp["config"].get("multiagent", {}).get("policy_mapping_fn")
                exp["config"]["callbacks"] = self._make_restore_callbacks(
                    checkpoint_path,
                    base_callbacks,
                    policy_mapping_fn=policy_mapping_fn,
                )

            tuner = tune.Tuner(
                trainable=exp["algorithm"],
                param_space=exp["config"],
                run_config=train.RunConfig(
                    storage_path=os.path.abspath(self._experiment_dir),
                    stop=exp["stop"],
                    checkpoint_config=self._make_checkpoint_config(exp),
                ),
            )

        return tuner.fit()

    #ONNX 导出
    @staticmethod
    def _export_onnx(result: tune.ResultGrid, is_multiagent: bool,policy_names: Optional[list]) -> None:
        """从最优 checkpoint 导出 ONNX 模型。"""
        checkpoint = result.get_best_result().checkpoint
        if not checkpoint:
            return

        result_path = result.get_best_result().path
        algo = Algorithm.from_checkpoint(checkpoint)

        if is_multiagent:
            for policy_name in set(policy_names):
                export_path = f"{result_path}/onnx_export/{policy_name}_onnx"
                algo.get_policy(policy_name).export_model(export_path, onnx=12)
                print(f"Saving onnx policy to "
                    f"{pathlib.Path(export_path).resolve()}")
        else:
            export_path = f"{result_path}/onnx_export/single_agent_policy_onnx"
            algo.get_policy().export_model(export_path, onnx=12)
            print(f"Saving onnx policy to "
                f"{pathlib.Path(export_path).resolve()}")

    # 主入口
    def execute(self) -> None:
        """完整训练流水线"""
        # 加载 YAML 配置
        self._exp = self._load_config()

        #注册环境
        env_name = "godot"
        tune.register_env(env_name, self._create_env_creator())

        # 临时环境获取元数据
        self._make_temp_env()

        # 多/单智能体配置注入
        self._configure_agent_mode()

        # 初始化 Ray (注入 PYTHONPATH 使 worker 进程能找到本地模块)
        training_dir = os.path.dirname(os.path.abspath(__file__))
        ray.init(
            _temp_dir=os.path.abspath(self._experiment_dir),
            runtime_env={"env_vars": {"PYTHONPATH": training_dir}},
        )

        # 训练
        result = self._run_training()

        # 导出 ONNX
        self._export_onnx(result, self._is_multiagent, self._policy_names)


#入口

if __name__ == "__main__":
    args = _parse_args()
    pipeline = RLLibTrainingPipeline(
        config_path=args.config_file,
        experiment_dir=args.experiment_dir,
        resume_experiment=args.resume_experiment,
        resume_checkpoint=args.resume_checkpoint,
        restore=args.restore,
    )
    pipeline.execute()
