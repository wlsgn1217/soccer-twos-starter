import os
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from models._mlp import _build_mlp
from models.history_encoder import HistoryEncoder


class GRUStudentPrivilegedCriticModel(TorchModelV2, nn.Module):
    """
    Asymmetric actor-critic for student selfplay fine-tuning.

    Actor:  GRU history encoder (obs+prev_action -> latent) + MLP backbone (latent+obs -> logits)
            Initialized from DAgger student (history encoder) and JH_0_AGENT (actor backbone).

    Critic: Privileged encoder (privileged_vec -> latent) + MLP value backbone (latent+obs -> value)
            Initialized from teacher privileged checkpoint.

    Observation space must be Dict({"obs": Box(obs_dim,), "privileged": Box(privileged_dim,)}).
    RLlib flattens this alphabetically: [obs_dim, privileged_dim] concatenated.

    RNN state: single GRU hidden vector [B, gru_hidden_size].
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {}) or {}
        self.obs_dim = int(cfg.get("obs_dim", 336))
        self.privileged_dim = int(cfg.get("privileged_dim", 25))
        self.latent_dim = int(cfg.get("latent_dim", 16))
        self.gru_hidden_size = int(cfg.get("gru_hidden_size", 256))
        self.action_embed_dim = int(cfg.get("action_embed_dim", 8))
        self.action_dim = int(cfg.get("action_dim", num_outputs))
        output_hiddens = list(cfg.get("output_hiddens", [64]))
        encoder_hiddens = list(cfg.get("encoder_hiddens", [64, 64]))
        actor_hiddens = list(cfg.get("actor_hiddens", [256, 256]))
        value_hiddens = list(cfg.get("value_hiddens", [256, 256]))
        activation = str(cfg.get("activation", "relu")).lower()
        freeze_actor = bool(cfg.get("freeze_actor_backbone", True))
        freeze_value = bool(cfg.get("freeze_value_network", False))

        # ---- Actor: GRU history encoder ----
        self.history_encoder = HistoryEncoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_size=self.gru_hidden_size,
            action_embed_dim=self.action_embed_dim,
            output_hiddens=output_hiddens,
            activation=activation,
        )

        # ---- Actor: MLP backbone ----
        self.actor_backbone = _build_mlp(
            self.obs_dim + self.latent_dim, actor_hiddens, num_outputs, activation
        )

        # ---- Critic: privileged encoder + value backbone ----
        self.privileged_encoder = _build_mlp(
            self.privileged_dim, encoder_hiddens, self.latent_dim, activation
        )
        self.value_backbone = _build_mlp(
            self.obs_dim + self.latent_dim, value_hiddens, 1, activation
        )

        self._value_out = None

        # ---- Load weights ----
        actor_weights_path = cfg.get("actor_weights_path", "")
        history_encoder_path = cfg.get("history_encoder_path", "")
        privileged_encoder_weights_path = cfg.get("privileged_encoder_weights_path", "")
        value_weights_path = cfg.get("value_weights_path", "")

        if actor_weights_path and os.path.isfile(actor_weights_path):
            self.actor_backbone.load_state_dict(
                torch.load(actor_weights_path, map_location="cpu")
            )
            print(f"[GRUStudentModel] Loaded actor backbone from {actor_weights_path}")

        if history_encoder_path and os.path.isfile(history_encoder_path):
            self.history_encoder.load_state_dict(
                torch.load(history_encoder_path, map_location="cpu")
            )
            print(f"[GRUStudentModel] Loaded history encoder from {history_encoder_path}")

        if privileged_encoder_weights_path and os.path.isfile(privileged_encoder_weights_path):
            self.privileged_encoder.load_state_dict(
                torch.load(privileged_encoder_weights_path, map_location="cpu")
            )
            print(f"[GRUStudentModel] Loaded privileged encoder from {privileged_encoder_weights_path}")

        if value_weights_path and os.path.isfile(value_weights_path):
            self.value_backbone.load_state_dict(
                torch.load(value_weights_path, map_location="cpu")
            )
            print(f"[GRUStudentModel] Loaded value backbone from {value_weights_path}")

        # ---- Freeze weights ----
        if freeze_actor:
            for p in self.actor_backbone.parameters():
                p.requires_grad_(False)
        if freeze_value:
            for p in self.privileged_encoder.parameters():
                p.requires_grad_(False)
            for p in self.value_backbone.parameters():
                p.requires_grad_(False)

    def get_initial_state(self):
        return [torch.zeros(self.gru_hidden_size)]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        input_dict["obs_flat"]: [B*T, obs_dim + privileged_dim]
        input_dict["prev_actions"]: [B*T] int64  (may be absent in GAE bootstrap calls)
        state[0]: [B, gru_hidden_size]            (may be [] in dummy/bootstrap calls)
        seq_lens: [B] lengths of each sequence in the batch
        """
        obs_flat = input_dict["obs_flat"].float()   # [B*T, D]

        # prev_actions is absent in PPO's GAE bootstrap dummy call; default to zeros (null tokens).
        prev_actions_raw = input_dict.get("prev_actions")
        if prev_actions_raw is None:
            prev_actions = torch.zeros(obs_flat.shape[0], dtype=torch.long, device=obs_flat.device)
        else:
            prev_actions = prev_actions_raw.long()

        B = len(seq_lens)
        T = obs_flat.shape[0] // B

        obs_flat_t = obs_flat.reshape(B, T, -1)          # [B, T, D]
        prev_actions_t = prev_actions.reshape(B, T)       # [B, T]

        obs_t = obs_flat_t[:, :, :self.obs_dim]
        privileged_t = obs_flat_t[:, :, self.obs_dim: self.obs_dim + self.privileged_dim]

        # GRU forward.  Shift prev_actions by +1: index 0 = null/start-of-episode token.
        # state may be an empty list during bootstrap calls; fall back to zero hidden state.
        if state and len(state) > 0:
            h = state[0].unsqueeze(0)                     # [1, B, H]
        else:
            h = torch.zeros(1, B, self.gru_hidden_size, device=obs_flat.device)
        act_idx = (prev_actions_t + 1).clamp(0, self.action_dim)  # [B, T]
        act_emb = self.history_encoder.action_embed(act_idx)       # [B, T, E]
        gru_input = torch.cat([obs_t, act_emb], dim=-1)            # [B, T, obs+E]
        gru_out, new_h = self.history_encoder.gru(gru_input, h)    # [B,T,H], [1,B,H]
        latent_t = self.history_encoder.output_head(gru_out)        # [B, T, latent_dim]

        # Actor
        fused_actor = torch.cat([obs_t, latent_t], dim=-1)          # [B, T, obs+latent]
        logits_t = self.actor_backbone(fused_actor)                  # [B, T, num_outputs]

        # Critic (asymmetric: uses privileged features unavailable to actor at deployment)
        priv_latent_t = self.privileged_encoder(privileged_t)        # [B, T, latent_dim]
        fused_value = torch.cat([obs_t, priv_latent_t], dim=-1)      # [B, T, obs+latent]
        self._value_out = self.value_backbone(fused_value).squeeze(-1)  # [B, T]

        logits_flat = logits_t.reshape(-1, self.num_outputs)
        return logits_flat, [new_h.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out.reshape(-1)
