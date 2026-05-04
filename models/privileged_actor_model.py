import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


def _build_mlp(in_dim: int, hidden_dims, out_dim: int, activation: str) -> nn.Sequential:
    act_cls = nn.ReLU if activation == "relu" else nn.Tanh
    layers = []
    prev = int(in_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(act_cls())
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class PrivilegedActorModel(TorchModelV2, nn.Module):
    """
    Stage-1 privileged learning model.
    - privileged_encoder(privileged) -> latent
    - actor/value consume concat([obs, latent]).
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {})
        obs_dim = int(custom.get("obs_dim", 336))
        privileged_dim = int(custom.get("privileged_dim", 25))
        self.latent_dim = int(custom.get("latent_dim", 1))
        encoder_hiddens = custom.get("encoder_hiddens", [64, 64])
        actor_hiddens = custom.get("actor_hiddens", [256, 256])
        value_hiddens = custom.get("value_hiddens", [256, 256])
        activation = str(custom.get("activation", "relu")).lower()

        fused_dim = obs_dim + self.latent_dim
        self.privileged_encoder = _build_mlp(
            privileged_dim, encoder_hiddens, self.latent_dim, activation
        )
        self.actor_backbone = _build_mlp(
            fused_dim, actor_hiddens, num_outputs, activation
        )
        self.value_backbone = _build_mlp(
            fused_dim, value_hiddens, 1, activation
        )
        self._last_value = None
        self.last_latent_mean = 0.0
        self.last_latent_std = 0.0

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        obs_vec = obs["obs"].float()
        privileged_vec = obs["privileged"].float()
        latent = self.privileged_encoder(privileged_vec)
        fused = torch.cat([obs_vec, latent], dim=-1)
        logits = self.actor_backbone(fused)
        self._last_value = self.value_backbone(fused).squeeze(-1)

        with torch.no_grad():
            self.last_latent_mean = float(torch.mean(latent).cpu().item())
            self.last_latent_std = float(torch.std(latent).cpu().item())

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value

    def export_split_state_dict(self):
        return {
            "privileged_encoder": self.privileged_encoder.state_dict(),
            "actor_backbone": self.actor_backbone.state_dict(),
            "value_backbone": self.value_backbone.state_dict(),
            "latent_dim": int(self.latent_dim),
        }
