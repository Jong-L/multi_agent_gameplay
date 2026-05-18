from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from custom_ppo_dataclass import NetworkType
from godot_env_wrapper import ObsSegmentDims, layer_init


def as_hidden_tuple(value, default: tuple) -> tuple:
    """Normalize None/int/iterable hidden-size config into a tuple[int, ...]."""
    if value is None:
        value = default
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


def make_mlp(input_dim: int, hidden_sizes: tuple) -> tuple[nn.Module, int]:
    """Build Linear+ReLU layers and return both module and output size."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(layer_init(nn.Linear(in_dim, hidden_size)))
        layers.append(nn.ReLU())
        in_dim = hidden_size
    return (nn.Sequential(*layers) if layers else nn.Identity(), in_dim)


def init_gru_weights(gru: nn.GRU) -> nn.GRU:
    """Use stable GRU initialization matching rllib_custom_network.py."""
    for name, param in gru.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(param)
        elif "bias" in name:
            torch.nn.init.constant_(param, 0.0)
    return gru


class SegmentedObsHelper:
    """Small helper for the shared observation layout."""

    def __init__(self, seg: ObsSegmentDims):
        self.seg = seg

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        i = 0
        s = obs[:, i: i + self.seg.self_dim]
        i += self.seg.self_dim
        p = obs[:, i: i + self.seg.player_dim]
        i += self.seg.player_dim
        b = obs[:, i: i + self.seg.ball_dim]
        i += self.seg.ball_dim
        e = obs[:, i: i + self.seg.enemy_dim]
        i += self.seg.enemy_dim
        m = obs[:, i: i + self.seg.map_dim]
        return s, p, b, e, m


class FlatMlpEncoder(nn.Module):
    def __init__(self, obs_dim: int, mlp_hiddens: tuple):
        super().__init__()
        self.trunk, self.output_dim = make_mlp(obs_dim, mlp_hiddens)
        self.recurrent_state_size = 0

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.trunk(obs), None


class SegmentedMlpEncoder(nn.Module):
    def __init__(
        self,
        obs_helper: SegmentedObsHelper,
        self_hiddens: tuple,
        player_hiddens: tuple,
        ball_hiddens: tuple,
        enemy_hiddens: tuple,
        map_hiddens: tuple,
        trunk_hiddens: tuple,
    ):
        super().__init__()
        seg = obs_helper.seg
        self.obs = obs_helper
        self.self_net, self_out = make_mlp(seg.self_dim, self_hiddens)
        self.player_net, player_out = make_mlp(seg.player_dim, player_hiddens)
        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        self.enemy_net, enemy_out = make_mlp(seg.enemy_dim, enemy_hiddens)
        self.map_net, map_out = make_mlp(seg.map_dim, map_hiddens)

        fused_dim = self_out + player_out + ball_out + enemy_out + map_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)
        self.recurrent_state_size = 0

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        s, p, b, e, m = self.obs.split(obs)
        fused = torch.cat([
            self.self_net(s),
            self.player_net(p),
            self.ball_net(b),
            self.enemy_net(e),
            self.map_net(m),
        ], dim=1)
        return self.trunk(fused), None


class GruMlpEncoder(nn.Module):
    """GRU-MLP encoder mirroring rllib_custom_network.py."""

    def __init__(
        self,
        obs_helper: SegmentedObsHelper,
        ball_hiddens: tuple,
        trunk_hiddens: tuple,
        gru_hidden: int,
        gru_num_layers: int,
        gru_input_layernorm: bool,
    ):
        super().__init__()
        seg = obs_helper.seg
        self.obs = obs_helper
        self.gru_hidden = int(gru_hidden)
        self.gru_num_layers = int(gru_num_layers)
        self.recurrent_state_size = self.gru_hidden * self.gru_num_layers

        gru_input_dim = seg.self_dim + seg.player_dim + seg.enemy_dim + seg.map_dim
        self.gru_ln = nn.LayerNorm(gru_input_dim) if gru_input_layernorm else nn.Identity()
        self.gru = init_gru_weights(
            nn.GRU(
                input_size=gru_input_dim,
                hidden_size=self.gru_hidden,
                num_layers=self.gru_num_layers,
                batch_first=True,
            )
        )

        self.ball_net, ball_out = make_mlp(seg.ball_dim, ball_hiddens)
        fused_dim = self.gru_hidden + ball_out
        self.trunk, self.output_dim = make_mlp(fused_dim, trunk_hiddens)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.recurrent_state_size, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        if rnn_state is None:
            rnn_state = self.initial_state(batch_size, obs.device)
        if rnn_state.dim() == 1:
            rnn_state = rnn_state.unsqueeze(0)

        s, p, b, e, m = self.obs.split(obs)
        gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(1)
        gru_input = self.gru_ln(gru_input)

        h0 = rnn_state.view(batch_size, self.gru_num_layers, self.gru_hidden)
        h0 = h0.transpose(0, 1).contiguous()
        gru_out, h_new = self.gru(gru_input, h0)

        ball_features = self.ball_net(b)
        fused = torch.cat([gru_out[:, -1, :], ball_features], dim=1)
        features = self.trunk(fused)
        h_new = h_new.transpose(0, 1).reshape(batch_size, self.recurrent_state_size)
        return features, h_new

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete single-environment sequence through the GRU."""
        s, p, b, e, m = self.obs.split(obs_seq)

        gru_input = torch.cat([s, p, e, m], dim=1).unsqueeze(0)
        gru_input = self.gru_ln(gru_input)

        h0 = rnn_state.view(1, self.gru_num_layers, self.gru_hidden)
        h0 = h0.transpose(0, 1).contiguous()
        gru_out, h_new = self.gru(gru_input, h0)

        ball_features = self.ball_net(b)
        fused = torch.cat([gru_out.squeeze(0), ball_features], dim=1)
        features = self.trunk(fused)
        h_new = h_new.transpose(0, 1).reshape(1, self.recurrent_state_size)
        return features, h_new


class DiscreteActorCriticAgent(nn.Module):
    """Policy/value module with pluggable MLP, segmented MLP, or GRU-MLP encoder."""

    def __init__(
        self,
        n_actions: int,
        seg: ObsSegmentDims,
        cfg,
    ):
        super().__init__()

        self.network_type = cfg.network_type
        self.seg = seg
        obs_helper = SegmentedObsHelper(seg)

        self_hiddens = as_hidden_tuple(cfg.self_hidden, (cfg.self_hidden,))
        player_hiddens = as_hidden_tuple(cfg.player_hidden, (cfg.player_hidden,))
        ball_hiddens = as_hidden_tuple(cfg.ball_hidden, (cfg.ball_hidden,))
        enemy_hiddens = as_hidden_tuple(cfg.enemy_hidden, (cfg.enemy_hidden,))
        map_hiddens = as_hidden_tuple(cfg.map_hidden, (cfg.map_hidden,))
        trunk_hiddens = as_hidden_tuple(cfg.trunk_hiddens, cfg.trunk_hiddens)
        mlp_hiddens = as_hidden_tuple(cfg.mlp_hiddens, cfg.mlp_hiddens)

        if self.network_type == NetworkType.SEGMENTED_MLP:
            self.encoder = SegmentedMlpEncoder(
                obs_helper,
                self_hiddens,
                player_hiddens,
                ball_hiddens,
                enemy_hiddens,
                map_hiddens,
                trunk_hiddens,
            )
        elif self.network_type == NetworkType.MLP:
            self.encoder = FlatMlpEncoder(seg.total, mlp_hiddens)
        elif self.network_type == NetworkType.GRU_MLP:
            self.encoder = GruMlpEncoder(
                obs_helper,
                ball_hiddens=ball_hiddens,
                trunk_hiddens=trunk_hiddens,
                gru_hidden=cfg.gru_hidden,
                gru_num_layers=cfg.gru_num_layers,
                gru_input_layernorm=cfg.gru_input_layernorm,
            )
        else:
            raise ValueError(f"Unsupported network_type={self.network_type}")

        self.actor = layer_init(nn.Linear(self.encoder.output_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(self.encoder.output_dim, 1), std=1.0)

    @property
    def is_recurrent(self) -> bool:
        return self.encoder.recurrent_state_size > 0

    @property
    def recurrent_state_size(self) -> int:
        return self.encoder.recurrent_state_size

    def get_initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.is_recurrent:
            return None
        return self.encoder.initial_state(batch_size, device)

    def _forward_features(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.encoder(obs, rnn_state) if self.is_recurrent else self.encoder(obs)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_value(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features, _ = self._forward_features(obs, rnn_state)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        rnn_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        features, next_rnn_state = self._forward_features(obs, rnn_state)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        result = (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(features),
        )
        if return_state:
            return (*result, next_rnn_state)
        return result

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one recurrent sequence and return logprob/entropy/value/state."""
        if not self.is_recurrent:
            raise ValueError("evaluate_sequence requires a recurrent encoder.")

        features_seq, next_state = self.encoder.forward_sequence(obs_seq, rnn_state)
        logits_seq = self.actor(features_seq)
        probs = Categorical(logits=logits_seq)
        logprobs = probs.log_prob(action_seq)
        entropies = probs.entropy()
        values = self.critic(features_seq).squeeze(-1)
        return logprobs, entropies, values, next_state
