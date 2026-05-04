"""
Stage-2 DAgger training: history-encoder student policy.

The Stage-1 teacher uses a privileged MLP encoder (ground-truth state -> 16-d latent).
The Stage-2 student replaces that encoder with a GRU that reads a rolling window of
(obs_t, prev_action_t) pairs.  The actor backbone is frozen; only the history encoder
is trained.

Loss (per batch window):
  total = alpha * MSE(history_latent, privileged_latent.detach())
        + (1 - alpha) * KL(student_log_probs, teacher_probs.detach())

DAgger beta schedule: at iteration i, execute the teacher with probability beta_i
(linearly annealed from beta_start -> beta_end).  Labels are always the teacher's
full action distribution, not just the argmax.

Usage:
  python train_dagger_student.py \\
      --stage1-split-dir ./ray_results/.../split_stage1 \\
      --out-dir ./dagger_student_run1
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import soccer_twos
from custom_env_wrapper import _normalize_info
from dagger_utils import compute_beta, to_torch_device
from models.history_encoder import HistoryEncoder
from models._mlp import _build_mlp
from privileged_features import PRIVILEGED_OBS_DIM, build_privileged_vector
from train_against_baseline import BlueTeamVsBaselineWrapper, DEFAULT_BASELINE_MODULE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StudentConfig:
    stage1_split_dir: str
    out_dir: str
    baseline_module: str
    window_len: int
    dagger_iters: int
    episodes_per_iter: int
    max_steps_per_episode: int
    student_epochs: int
    batch_size: int
    learning_rate: float
    beta_start: float
    beta_end: float
    max_episodes_in_buffer: int
    seed: int
    device: str
    obs_dim: int
    gru_hidden_size: int
    action_embed_dim: int
    output_hiddens: Tuple[int, ...]
    activation: str
    alpha: float
    worker_id: int
    base_port: int


def parse_args() -> StudentConfig:
    p = argparse.ArgumentParser(description="Stage-2 DAgger: train history-encoder student.")
    p.add_argument("--stage1-split-dir", type=str, required=True,
                   help="Directory containing actor_weights.pt and privileged_encoder_weights.pt.")
    p.add_argument("--out-dir", type=str, default="./dagger_student_artifacts")
    p.add_argument("--baseline-module", type=str, default=DEFAULT_BASELINE_MODULE,
                   help="Opponent baseline module name.")
    p.add_argument("--window-len", type=int, default=20,
                   help="BPTT window length T (history steps fed to GRU each update).")
    p.add_argument("--dagger-iters", type=int, default=20)
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--max-steps-per-episode", type=int, default=2000)
    p.add_argument("--student-epochs", type=int, default=10,
                   help="Gradient steps per DAgger iteration (one sampled batch each).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--beta-start", type=float, default=1.0)
    p.add_argument("--beta-end", type=float, default=0.1)
    p.add_argument("--max-episodes-in-buffer", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--obs-dim", type=int, default=336)
    p.add_argument("--gru-hidden-size", type=int, default=128)
    p.add_argument("--action-embed-dim", type=int, default=8)
    p.add_argument("--output-hiddens", type=int, nargs="+", default=[64])
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on latent MSE loss; (1-alpha) on action KL loss.")
    p.add_argument("--worker-id", type=int, default=0,
                   help="ML-Agents worker ID. Use a unique value per parallel run.")
    p.add_argument("--base-port", type=int, default=5005,
                   help="ML-Agents base port. Each run should use a different port.")
    args = p.parse_args()
    return StudentConfig(
        stage1_split_dir=args.stage1_split_dir,
        out_dir=args.out_dir,
        baseline_module=args.baseline_module,
        window_len=args.window_len,
        dagger_iters=args.dagger_iters,
        episodes_per_iter=args.episodes_per_iter,
        max_steps_per_episode=args.max_steps_per_episode,
        student_epochs=args.student_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        max_episodes_in_buffer=args.max_episodes_in_buffer,
        seed=args.seed,
        device=args.device,
        obs_dim=args.obs_dim,
        gru_hidden_size=args.gru_hidden_size,
        action_embed_dim=args.action_embed_dim,
        output_hiddens=tuple(args.output_hiddens),
        activation=args.activation,
        alpha=args.alpha,
        worker_id=args.worker_id,
        base_port=args.base_port,
    )


# ---------------------------------------------------------------------------
# Load Stage-1 components from split dir
# ---------------------------------------------------------------------------

def _infer_mlp_arch(state_dict: dict) -> Tuple[int, List[int], int]:
    """Return (in_dim, hidden_dims, out_dim) from a _build_mlp state dict."""
    weight_keys = sorted(
        [k for k in state_dict if k.endswith(".weight")],
        key=lambda k: int(k.split(".")[0]),
    )
    in_dim = int(state_dict[weight_keys[0]].shape[1])
    out_dim = int(state_dict[weight_keys[-1]].shape[0])
    hidden_dims = [int(state_dict[k].shape[0]) for k in weight_keys[:-1]]
    return in_dim, hidden_dims, out_dim


def load_stage1_components(split_dir: str, activation: str):
    """
    Load and freeze the actor backbone and privileged encoder from Stage 1.
    Returns (actor_backbone, privileged_encoder, action_dim, priv_dim, latent_dim).
    Architecture is inferred from the saved state dicts — no CLI flags required.
    """
    actor_path = os.path.join(split_dir, "actor_weights.pt")
    encoder_path = os.path.join(split_dir, "privileged_encoder_weights.pt")
    for path in (actor_path, encoder_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing Stage-1 weight file: {path}")

    actor_state = torch.load(actor_path, map_location="cpu")
    encoder_state = torch.load(encoder_path, map_location="cpu")

    actor_in, actor_hidden, action_dim = _infer_mlp_arch(actor_state)
    priv_dim, encoder_hidden, latent_dim = _infer_mlp_arch(encoder_state)
    obs_dim = actor_in - latent_dim

    print(f"[Stage1] Inferred: obs_dim={obs_dim}, priv_dim={priv_dim}, "
          f"latent_dim={latent_dim}, action_dim={action_dim}")
    print(f"[Stage1] actor_hidden={actor_hidden}, encoder_hidden={encoder_hidden}")

    actor_backbone = _build_mlp(actor_in, actor_hidden, action_dim, activation)
    actor_backbone.load_state_dict(actor_state)
    for p in actor_backbone.parameters():
        p.requires_grad = False

    privileged_encoder = _build_mlp(priv_dim, encoder_hidden, latent_dim, activation)
    privileged_encoder.load_state_dict(encoder_state)
    for p in privileged_encoder.parameters():
        p.requires_grad = False

    return actor_backbone, privileged_encoder, action_dim, priv_dim, latent_dim


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """
    Stores completed episodes as aligned arrays.  Samples fixed-length windows
    for BPTT training, always within a single episode (no cross-boundary windows).

    Per-episode storage:
      obs:       [T, obs_dim]  float32
      prev_acts: [T]           int64, 0 = null/start-of-episode token
      privileged:[T, priv_dim] float32
    """

    def __init__(self, window_len: int, max_episodes: int = 500):
        self.window_len = window_len
        self.max_episodes = max_episodes
        self._episodes: List[dict] = []

    def add_episode(
        self,
        obs: np.ndarray,        # [T, obs_dim]
        prev_acts: np.ndarray,  # [T]
        privileged: np.ndarray, # [T, priv_dim]
    ) -> None:
        if len(obs) < self.window_len:
            return
        self._episodes.append({
            "obs": obs.astype(np.float32),
            "prev_acts": prev_acts.astype(np.int64),
            "privileged": privileged.astype(np.float32),
        })
        if len(self._episodes) > self.max_episodes:
            del self._episodes[0]

    def total_steps(self) -> int:
        return sum(len(ep["obs"]) for ep in self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def sample_windows(self, batch_size: int) -> Optional[dict]:
        """
        Sample batch_size windows of length window_len.
        obs_windows:    [B, T, obs_dim]
        act_windows:    [B, T]           prev_act indices
        current_obs:    [B, obs_dim]     obs at the last window step
        current_priv:   [B, priv_dim]    privileged at the last window step
        """
        eligible = [ep for ep in self._episodes if len(ep["obs"]) >= self.window_len]
        if not eligible:
            return None

        obs_wins, act_wins, cur_obs_list, cur_priv_list = [], [], [], []
        T = self.window_len
        for _ in range(batch_size):
            ep = random.choice(eligible)
            T_ep = len(ep["obs"])
            end = random.randint(T - 1, T_ep - 1)
            start = end - T + 1
            obs_wins.append(ep["obs"][start : end + 1])
            act_wins.append(ep["prev_acts"][start : end + 1])
            cur_obs_list.append(ep["obs"][end])
            cur_priv_list.append(ep["privileged"][end])

        return {
            "obs_windows": np.stack(obs_wins, axis=0),       # [B, T, obs_dim]
            "act_windows": np.stack(act_wins, axis=0),       # [B, T]
            "current_obs": np.stack(cur_obs_list, axis=0),   # [B, obs_dim]
            "current_priv": np.stack(cur_priv_list, axis=0), # [B, priv_dim]
        }


# ---------------------------------------------------------------------------
# Student model (history encoder + frozen Stage-1 components)
# ---------------------------------------------------------------------------

class HistoryStudentModel(nn.Module):
    """
    history_encoder  — trainable
    actor_backbone   — frozen (Stage 1)
    privileged_encoder — frozen (Stage 1), used only to generate training targets
    """

    def __init__(
        self,
        history_encoder: HistoryEncoder,
        actor_backbone: nn.Module,
        privileged_encoder: nn.Module,
    ):
        super().__init__()
        self.history_encoder = history_encoder
        self.actor_backbone = actor_backbone
        self.privileged_encoder = privileged_encoder

    def forward(
        self,
        obs_windows: torch.Tensor,   # [B, T, obs_dim]
        act_windows: torch.Tensor,   # [B, T]
        current_obs: torch.Tensor,   # [B, obs_dim]
        current_priv: torch.Tensor,  # [B, priv_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_latent = self.history_encoder(obs_windows, act_windows)  # [B, latent_dim]
        with torch.no_grad():
            privileged_latent = self.privileged_encoder(current_priv)    # [B, latent_dim]
            teacher_logits = self.actor_backbone(
                torch.cat([current_obs, privileged_latent], dim=-1)
            )
        student_logits = self.actor_backbone(
            torch.cat([current_obs, history_latent], dim=-1)
        )
        return history_latent, privileged_latent, student_logits, teacher_logits


def compute_losses(
    history_latent: torch.Tensor,
    privileged_latent: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_loss = F.mse_loss(history_latent, privileged_latent)
    # KL(teacher || student): minimise when student matches teacher's distribution.
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    action_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    total = alpha * latent_loss + (1.0 - alpha) * action_loss
    return total, latent_loss, action_loss


# ---------------------------------------------------------------------------
# Privileged feature extraction from env info
# ---------------------------------------------------------------------------

def _extract_privileged(info) -> np.ndarray:
    info_norm = _normalize_info(info)
    if info_norm is None:
        return np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
    try:
        return build_privileged_vector(info_norm, 0)
    except Exception:
        return np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)


def _flatten_obs(obs, obs_dim: int) -> np.ndarray:
    if isinstance(obs, dict):
        if 0 in obs:
            obs = obs[0]
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
    vec = np.asarray(obs, dtype=np.float32).reshape(-1)
    if vec.shape[0] > obs_dim:
        vec = vec[:obs_dim]
    elif vec.shape[0] < obs_dim:
        padded = np.zeros(obs_dim, dtype=np.float32)
        padded[: vec.shape[0]] = vec
        vec = padded
    return vec


# ---------------------------------------------------------------------------
# Episode collection (2v2 game vs baseline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _teacher_action(
    model: HistoryStudentModel,
    obs_vec: np.ndarray,
    priv_vec: np.ndarray,
    device: torch.device,
) -> int:
    obs_t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)
    priv_t = torch.as_tensor(priv_vec, dtype=torch.float32).unsqueeze(0).to(device)
    priv_latent = model.privileged_encoder(priv_t)
    logits = model.actor_backbone(torch.cat([obs_t, priv_latent], dim=-1))
    return int(logits.argmax(dim=-1).item())


def collect_episode(
    env,
    model: HistoryStudentModel,
    beta: float,
    max_steps: int,
    obs_dim: int,
    device: torch.device,
) -> list:
    """
    Collect one 2v2 episode against the baseline using the beta-mixture of
    teacher and student for both training agents (0 and 1).

    Returns a list of two trajectory dicts (one per agent):
      obs_seq:      [T, obs_dim]
      prev_act_seq: [T]          (0 = start-of-episode null token)
      priv_seq:     [T, priv_dim]
      ep_reward, ep_steps
    """
    obs_dict = env.reset()
    obs_vecs = {a: _flatten_obs(obs_dict[a], obs_dim) for a in (0, 1)}

    obs_lists = {0: [], 1: []}
    prev_act_lists = {0: [], 1: []}
    priv_lists = {0: [], 1: []}
    hiddens = {a: model.history_encoder.init_hidden(1, device) for a in (0, 1)}
    prev_acts = {0: 0, 1: 0}
    priv_vecs = {a: np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32) for a in (0, 1)}

    done = False
    ep_reward = 0.0
    step_i = 0
    model.eval()

    while not done and step_i < max_steps:
        action_dict = {}
        new_hiddens = {}

        for a in (0, 1):
            obs_t = torch.as_tensor(obs_vecs[a], dtype=torch.float32).unsqueeze(0).to(device)
            prev_act_tensor = torch.tensor([prev_acts[a]], dtype=torch.long, device=device)

            # Student: GRU step → frozen actor → action
            history_latent, new_hidden = model.history_encoder.step(obs_t, prev_act_tensor, hiddens[a])
            student_logits = model.actor_backbone(torch.cat([obs_t, history_latent], dim=-1))
            student_action = int(student_logits.argmax(dim=-1).item())

            # Teacher: privileged encoder → frozen actor → action
            teacher_action = _teacher_action(model, obs_vecs[a], priv_vecs[a], device)

            exec_action = teacher_action if np.random.rand() < beta else student_action

            obs_lists[a].append(obs_vecs[a].copy())
            prev_act_lists[a].append(prev_acts[a])
            priv_lists[a].append(priv_vecs[a].copy())

            action_dict[a] = exec_action
            new_hiddens[a] = new_hidden

        next_obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

        for a in (0, 1):
            priv_vecs[a] = _extract_privileged(info_dict[a])
            hiddens[a] = new_hiddens[a]
            obs_vecs[a] = _flatten_obs(next_obs_dict[a], obs_dim)
            prev_acts[a] = action_dict[a] + 1
            ep_reward += float(rew_dict.get(a, 0.0))

        done = bool(done_dict.get("__all__", False))
        step_i += 1

    return [
        {
            "obs_seq": np.stack(obs_lists[a]),
            "prev_act_seq": np.array(prev_act_lists[a], dtype=np.int64),
            "priv_seq": np.stack(priv_lists[a]),
            "ep_reward": ep_reward / 2.0,
            "ep_steps": step_i,
        }
        for a in (0, 1)
        if len(obs_lists[a]) > 0
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: HistoryStudentModel,
    buffer: TrajectoryBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    alpha: float,
) -> dict:
    model.history_encoder.train()
    total_sum = latent_sum = action_sum = 0.0
    n = 0

    for _ in range(n_epochs):
        batch = buffer.sample_windows(batch_size)
        if batch is None:
            break

        obs_w = torch.as_tensor(batch["obs_windows"], dtype=torch.float32).to(device)
        act_w = torch.as_tensor(batch["act_windows"], dtype=torch.long).to(device)
        cur_obs = torch.as_tensor(batch["current_obs"], dtype=torch.float32).to(device)
        cur_priv = torch.as_tensor(batch["current_priv"], dtype=torch.float32).to(device)

        h_lat, p_lat, s_logits, t_logits = model(obs_w, act_w, cur_obs, cur_priv)
        total_loss, latent_loss, action_loss = compute_losses(
            h_lat, p_lat, s_logits, t_logits, alpha
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.history_encoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_sum += total_loss.item()
        latent_sum += latent_loss.item()
        action_sum += action_loss.item()
        n += 1

    denom = max(1, n)
    return {"total": total_sum / denom, "latent": latent_sum / denom, "action": action_sum / denom}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: HistoryStudentModel, out_dir: str, iteration: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"history_encoder_iter_{iteration:03d}.pt")
    torch.save(model.history_encoder.state_dict(), path)


def save_final(model: HistoryStudentModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        model.history_encoder.state_dict(),
        os.path.join(out_dir, "history_encoder_final.pt"),
    )
    # Also save config needed to reconstruct at inference time.
    gru = model.history_encoder.gru
    embed = model.history_encoder.action_embed
    torch.save(
        {
            "obs_dim": int(gru.input_size - embed.embedding_dim),
            "action_dim": int(embed.num_embeddings - 1),
            "latent_dim": int(model.history_encoder.output_head[-1].out_features),
            "hidden_size": int(gru.hidden_size),
            "action_embed_dim": int(embed.embedding_dim),
        },
        os.path.join(out_dir, "history_encoder_meta.pt"),
    )
    print(f"[save] history encoder → {out_dir}/history_encoder_final.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = parse_args()
    _set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = to_torch_device(cfg.device)

    # Build 2v2 environment against the baseline (same setting as train_against_baseline.py)
    base_env = soccer_twos.make(
        variation=soccer_twos.EnvType.multiagent_player,
        multiagent=True,
        flatten_branched=True,
        worker_id=cfg.worker_id,
        base_port=cfg.base_port,
    )
    env = BlueTeamVsBaselineWrapper(
        base_env,
        baseline_module=cfg.baseline_module,
        obs_mode="privileged_dict",
        expected_obs_dim=cfg.obs_dim,
    )

    # Load Stage-1 components; infer dims from saved state dicts.
    actor_backbone, privileged_encoder, action_dim, priv_dim, latent_dim = (
        load_stage1_components(cfg.stage1_split_dir, cfg.activation)
    )

    history_encoder = HistoryEncoder(
        obs_dim=cfg.obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_size=cfg.gru_hidden_size,
        action_embed_dim=cfg.action_embed_dim,
        output_hiddens=cfg.output_hiddens,
        activation=cfg.activation,
    )

    model = HistoryStudentModel(
        history_encoder=history_encoder,
        actor_backbone=actor_backbone,
        privileged_encoder=privileged_encoder,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.history_encoder.parameters(), lr=cfg.learning_rate
    )

    buffer = TrajectoryBuffer(
        window_len=cfg.window_len,
        max_episodes=cfg.max_episodes_in_buffer,
    )

    trainable = sum(p.numel() for p in model.history_encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,}  /  total: {total:,}")
    print(f"[init] window_len={cfg.window_len}  gru_hidden={cfg.gru_hidden_size}  "
          f"alpha={cfg.alpha}  device={device}")

    for it in range(cfg.dagger_iters):
        beta = compute_beta(it, cfg.beta_start, cfg.beta_end, cfg.dagger_iters)

        # --- collect ---
        ep_rewards, ep_steps_list = [], []
        for _ in range(cfg.episodes_per_iter):
            agent_episodes = collect_episode(
                env=env,
                model=model,
                beta=beta,
                max_steps=cfg.max_steps_per_episode,
                obs_dim=cfg.obs_dim,
                device=device,
            )
            for ep in agent_episodes:
                buffer.add_episode(ep["obs_seq"], ep["prev_act_seq"], ep["priv_seq"])
                ep_rewards.append(ep["ep_reward"])
                ep_steps_list.append(ep["ep_steps"])

        # --- train ---
        losses = train_epoch(
            model=model,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            batch_size=cfg.batch_size,
            n_epochs=cfg.student_epochs,
            alpha=cfg.alpha,
        )

        print(
            f"[iter {it:03d}]  beta={beta:.3f}  "
            f"buf_eps={len(buffer)}  buf_steps={buffer.total_steps():,}  "
            f"ep_rew={np.mean(ep_rewards):.3f}  "
            f"loss={losses['total']:.5f}  "
            f"latent={losses['latent']:.5f}  action={losses['action']:.5f}"
        )

        save_checkpoint(model, cfg.out_dir, it)

    save_final(model, cfg.out_dir)
    env.close()
    print(f"[done] Artifacts saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
