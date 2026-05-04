import torch
import torch.nn as nn

from models._mlp import _build_mlp


class HistoryEncoder(nn.Module):
    """
    GRU encoder: maps a rolling window of (obs_t, prev_action_t) pairs to a
    latent vector approximating the privileged-information latent from Stage 1.

    Action indexing convention:
      0   = null / start-of-episode token (no previous action)
      1.. = actual action index + 1

    Training forward (batch):
      obs_seq:  [B, T, obs_dim]
      act_seq:  [B, T]  int64, values in {0 .. action_dim}
      -> latent [B, latent_dim]  (from the GRU's final hidden state)

    Inference step (one step at a time, preserving hidden state across a rollout):
      obs_t:    [1, obs_dim]
      prev_act: [1]  int64  (0 at episode start)
      hidden:   [1, 1, hidden_size]
      -> latent [1, latent_dim], new_hidden [1, 1, hidden_size]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_size: int = 128,
        action_embed_dim: int = 8,
        output_hiddens=(64,),
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # +1 so index 0 is the unambiguous "null / start-of-episode" token.
        self.action_embed = nn.Embedding(action_dim + 1, action_embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=obs_dim + action_embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_head = _build_mlp(
            hidden_size, list(output_hiddens), latent_dim, activation
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Batched training forward.  Hidden is zero-initialized if None (truncated BPTT).
        Returns latent computed from the last step's GRU output.
        """
        act_emb = self.action_embed(act_seq)                   # [B, T, E]
        x = torch.cat([obs_seq, act_emb], dim=-1)              # [B, T, obs+E]
        out, _ = self.gru(x, hidden)                           # [B, T, H]
        return self.output_head(out[:, -1, :])                 # [B, latent_dim]

    def step(
        self,
        obs_t: torch.Tensor,
        prev_act: torch.Tensor,
        hidden: torch.Tensor,
    ):
        """
        Single-step inference, updating the hidden state in-place.

        obs_t:    [1, obs_dim]
        prev_act: [1]  int64
        hidden:   [1, 1, hidden_size]
        Returns:  latent [1, latent_dim], new_hidden [1, 1, hidden_size]
        """
        act_emb = self.action_embed(prev_act.unsqueeze(1))     # [1, 1, E]
        x = torch.cat([obs_t.unsqueeze(1), act_emb], dim=-1)  # [1, 1, obs+E]
        out, new_hidden = self.gru(x, hidden)                  # out [1,1,H]
        return self.output_head(out[:, 0, :]), new_hidden      # ([1,L], [1,1,H])

    def init_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
