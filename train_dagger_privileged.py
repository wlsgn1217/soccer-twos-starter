import argparse
import importlib.util
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import soccer_twos
from custom_env_wrapper import _normalize_info
from dagger_utils import (
    AggregatedDataset,
    call_expert_single_action,
    compute_beta,
    create_expert_agent,
    flatten_multidiscrete_action,
    to_torch_device,
)
from models.privileged_actor_model import _build_mlp
from privileged_features import PRIVILEGED_OBS_DIM, build_privileged_vector
from privileged_curriculum_sampling import tasks
from curriculum_sampling import sample_players_states, sample_pos_vel


@dataclass
class DAggerConfig:
    expert_module: str
    out_dir: str
    dagger_iters: int
    episodes_per_iter: int
    max_steps_per_episode: int
    student_epochs: int
    batch_size: int
    learning_rate: float
    beta_start: float
    beta_end: float
    max_dataset_size: int
    seed: int
    device: str
    obs_dim: int
    privileged_dim: int
    latent_dim: int
    encoder_hiddens: Tuple[int, ...]
    actor_hiddens: Tuple[int, ...]
    activation: str
    label_smoothing: float
    randomize_initial_state: bool
    pack_rllib_checkpoint: bool
    rllib_template_checkpoint: Optional[str]
    init_split_dir: Optional[str]


class PrivilegedStudent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        privileged_dim: int,
        action_dim: int,
        latent_dim: int,
        encoder_hiddens,
        actor_hiddens,
        activation: str,
    ):
        super().__init__()
        self.privileged_encoder = _build_mlp(privileged_dim, encoder_hiddens, latent_dim, activation)
        self.actor_backbone = _build_mlp(obs_dim + latent_dim, actor_hiddens, action_dim, activation)

    def forward(self, obs_vec: torch.Tensor, privileged_vec: torch.Tensor) -> torch.Tensor:
        latent = self.privileged_encoder(privileged_vec)
        fused = torch.cat([obs_vec, latent], dim=-1)
        return self.actor_backbone(fused)

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, privileged_vec: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        priv_t = torch.as_tensor(privileged_vec, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_t, priv_t)
        return int(torch.argmax(logits, dim=-1).item())


def parse_args() -> DAggerConfig:
    parser = argparse.ArgumentParser(description="Train privileged student with DAgger.")
    parser.add_argument("--expert-module", type=str, default="ceia_baseline_agent")
    parser.add_argument("--out-dir", type=str, default="./dagger_artifacts")
    parser.add_argument("--dagger-iters", type=int, default=10)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=2000)
    parser.add_argument("--student-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=0.05)
    parser.add_argument("--max-dataset-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--obs-dim", type=int, default=336)
    parser.add_argument("--privileged-dim", type=int, default=PRIVILEGED_OBS_DIM)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--encoder-hiddens", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--actor-hiddens", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--randomize-initial-state",
        action="store_true",
        default=True,
        help="Randomize initial ball/player state every episode.",
    )
    parser.add_argument(
        "--no-randomize-initial-state",
        dest="randomize_initial_state",
        action="store_false",
        help="Disable randomized initialization.",
    )
    parser.add_argument(
        "--pack-rllib-checkpoint",
        action="store_true",
        help="After training, pack split_stage1 weights into an RLlib checkpoint under split_stage1 (for previleged_agent CHECKPOINT_PATH).",
    )
    parser.add_argument(
        "--rllib-template-checkpoint",
        type=str,
        default=None,
        help="Optional RLlib checkpoint file for packing shell; default matches previleged_agent. Only if --pack-rllib-checkpoint.",
    )
    parser.add_argument(
        "--init-split-dir",
        type=str,
        default=None,
        help="Directory with actor_weights.pt and privileged_encoder_weights.pt from a prior run (split_stage1). Warm-starts the student before continuing DAgger.",
    )
    args = parser.parse_args()
    return DAggerConfig(
        expert_module=args.expert_module,
        out_dir=args.out_dir,
        dagger_iters=args.dagger_iters,
        episodes_per_iter=args.episodes_per_iter,
        max_steps_per_episode=args.max_steps_per_episode,
        student_epochs=args.student_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        max_dataset_size=args.max_dataset_size,
        seed=args.seed,
        device=args.device,
        obs_dim=args.obs_dim,
        privileged_dim=args.privileged_dim,
        latent_dim=args.latent_dim,
        encoder_hiddens=tuple(args.encoder_hiddens),
        actor_hiddens=tuple(args.actor_hiddens),
        activation=args.activation,
        label_smoothing=args.label_smoothing,
        randomize_initial_state=bool(args.randomize_initial_state),
        pack_rllib_checkpoint=bool(args.pack_rllib_checkpoint),
        rllib_template_checkpoint=args.rllib_template_checkpoint,
        init_split_dir=args.init_split_dir,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_player_obs(obs, player_id: int = 0):
    if isinstance(obs, dict):
        # Wrapped privileged-dict mode case.
        if "obs" in obs and "privileged" in obs:
            return obs["obs"]
        # Prefer player id 0 for single-player team_vs_policy mode.
        if 0 in obs:
            obs = obs[0]
        elif "0" in obs:
            obs = obs["0"]
        else:
            obs = next(iter(obs.values()))
        if isinstance(obs, dict) and "obs" in obs:
            return obs["obs"]
    return obs


def _flatten_obs(obs) -> np.ndarray:
    obs = _select_player_obs(obs)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _prepare_expert_obs(obs, obs_dim: int, player_id: int = 0):
    """
    Normalize observation into expert-compatible dict[player_id -> raw_obs_vec].
    """
    raw = _select_player_obs(obs, player_id=player_id)
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    # Baseline expert checkpoint expects 336-dim per-player observation.
    if vec.shape[0] > obs_dim:
        vec = vec[:obs_dim]
    elif vec.shape[0] < obs_dim:
        padded = np.zeros(obs_dim, dtype=np.float32)
        padded[: vec.shape[0]] = vec
        vec = padded
    return {player_id: vec}


def _resolve_action_dim(action_space: gym.Space) -> Tuple[int, np.ndarray]:
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action_space.n), None
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        return int(np.prod(nvec)), nvec
    raise ValueError(f"Unsupported action space for DAgger: {action_space}")


def _extract_privileged(info, player_id: int = 0) -> np.ndarray:
    info_norm = _normalize_info(info)
    if info_norm is None:
        return np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
    return build_privileged_vector(info_norm, player_id)


def _sample_reset_state() -> Tuple[dict, dict]:
    """
    Sample randomized initial state using the most diverse curriculum task.
    """
    task = tasks[-1]
    ball_state = sample_pos_vel(task["ranges"]["ball"])
    players_states = sample_players_states(task, ball_state["position"])
    return ball_state, players_states


def _randomize_env_reset(env: gym.Env, enabled: bool) -> None:
    if not enabled:
        return
    if not hasattr(env, "env_channel"):
        return
    ball_state, players_states = _sample_reset_state()
    env.env_channel.set_parameters(ball_state=ball_state, players_states=players_states)


def collect_iteration_data(
    env: gym.Env,
    expert,
    student: PrivilegedStudent,
    beta: float,
    episodes: int,
    max_steps: int,
    nvec: np.ndarray,
    randomize_initial_state: bool,
    obs_dim: int,
    action_dim: int,
) -> Dict[str, np.ndarray]:
    obs_rows, privileged_rows, action_rows = [], [], []
    student.eval()

    for _ in range(episodes):
        _randomize_env_reset(env, enabled=randomize_initial_state)
        obs = env.reset()
        done = False
        step_i = 0
        privileged = np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
        while not done and step_i < max_steps:
            obs_raw = _select_player_obs(obs)
            obs_vec = np.asarray(obs_raw, dtype=np.float32).reshape(-1)
            use_expert = np.random.rand() < beta

            # Most provided experts expect dict[player_id -> raw_obs_vec].
            expert_obs = _prepare_expert_obs(obs_raw, obs_dim=obs_dim, player_id=0)
            expert_action = call_expert_single_action(expert, expert_obs, player_id=0)
            expert_label = _to_expert_label(expert_action, nvec=nvec, action_dim=action_dim)

            if use_expert:
                # In flattened envs (Discrete), always execute flattened index.
                # In branched envs (MultiDiscrete), decode index back to branches.
                env_action = (
                    int(expert_label)
                    if nvec is None
                    else _unflatten_action(int(expert_label), nvec)
                )
            else:
                student_action_idx = student.act(obs_vec, privileged)
                env_action = student_action_idx if nvec is None else _unflatten_action(student_action_idx, nvec)

            next_obs, _, done, info = env.step(env_action)
            privileged = _extract_privileged(info)

            obs_rows.append(obs_vec)
            privileged_rows.append(privileged.copy())
            action_rows.append(expert_label)

            obs = next_obs
            step_i += 1

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "privileged": np.asarray(privileged_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.int64),
    }


def _unflatten_action(action_idx: int, nvec: np.ndarray) -> np.ndarray:
    out = np.zeros_like(nvec)
    v = int(action_idx)
    for i in range(len(nvec) - 1, -1, -1):
        out[i] = v % int(nvec[i])
        v //= int(nvec[i])
    return out


def _infer_nvec_for_discrete(action: np.ndarray, action_dim: int):
    """
    Infer branch sizes from a branched expert action for flattened Discrete envs.
    Example: action_dim=27 and len(action)=3 -> nvec=[3,3,3].
    """
    k = int(action.shape[0])
    if k <= 0:
        return None
    min_base = max(int(np.max(action)) + 1, 2)
    for base in range(min_base, 11):
        if (base ** k) == int(action_dim):
            return np.asarray([base] * k, dtype=np.int64)
    return None


def _to_expert_label(expert_action, nvec: np.ndarray, action_dim: int) -> int:
    if isinstance(expert_action, (int, np.integer)):
        return int(expert_action)
    arr = np.asarray(expert_action)
    if arr.ndim == 0:
        return int(arr.item())
    if nvec is not None:
        return flatten_multidiscrete_action(arr, nvec)
    inferred_nvec = _infer_nvec_for_discrete(arr.reshape(-1), action_dim=action_dim)
    if inferred_nvec is None:
        raise ValueError(
            f"Could not infer nvec for expert action shape {arr.shape} and action_dim {action_dim}."
        )
    return flatten_multidiscrete_action(arr, inferred_nvec)


def train_student(
    student: PrivilegedStudent,
    dataset: AggregatedDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    label_smoothing: float,
) -> float:
    if len(dataset) == 0:
        return 0.0
    student.train()
    obs_t = torch.as_tensor(dataset.obs, dtype=torch.float32)
    priv_t = torch.as_tensor(dataset.privileged, dtype=torch.float32)
    act_t = torch.as_tensor(dataset.actions, dtype=torch.long)
    loader = DataLoader(TensorDataset(obs_t, priv_t, act_t), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    use_builtin_label_smoothing = float(label_smoothing) > 0.0
    criterion = None
    if not use_builtin_label_smoothing:
        criterion = nn.CrossEntropyLoss()
    mean_loss = 0.0
    n_batches = 0
    student.to(device)

    for _ in range(epochs):
        for batch_obs, batch_priv, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_priv = batch_priv.to(device)
            batch_act = batch_act.to(device)
            logits = student(batch_obs, batch_priv)
            if criterion is not None:
                loss = criterion(logits, batch_act)
            else:
                # Backward-compatible label smoothing for older torch versions.
                log_probs = torch.log_softmax(logits, dim=-1)
                n_classes = logits.shape[-1]
                smooth = float(label_smoothing)
                with torch.no_grad():
                    true_dist = torch.full_like(
                        log_probs, fill_value=smooth / max(1, n_classes - 1)
                    )
                    true_dist.scatter_(1, batch_act.unsqueeze(1), 1.0 - smooth)
                loss = (-true_dist * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss += float(loss.item())
            n_batches += 1
    student.to(torch.device("cpu"))
    return mean_loss / max(1, n_batches)


def load_student_from_split_dir(student: PrivilegedStudent, split_dir: str) -> None:
    enc_path = os.path.join(split_dir, "privileged_encoder_weights.pt")
    act_path = os.path.join(split_dir, "actor_weights.pt")
    for path in (enc_path, act_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing weight file for --init-split-dir: {path}")
    student.privileged_encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
    student.actor_backbone.load_state_dict(torch.load(act_path, map_location="cpu"))


def export_student(student: PrivilegedStudent, out_dir: str) -> None:
    split_dir = os.path.join(out_dir, "split_stage1")
    os.makedirs(split_dir, exist_ok=True)
    torch.save(student.actor_backbone.state_dict(), os.path.join(split_dir, "actor_weights.pt"))
    torch.save(
        student.privileged_encoder.state_dict(),
        os.path.join(split_dir, "privileged_encoder_weights.pt"),
    )
    torch.save(
        {
            "obs_dim": student.actor_backbone[0].in_features - student.privileged_encoder[-1].out_features,
            "privileged_dim": student.privileged_encoder[0].in_features,
            "latent_dim": student.privileged_encoder[-1].out_features,
            "action_dim": student.actor_backbone[-1].out_features,
        },
        os.path.join(split_dir, "student_meta.pt"),
    )


def _pack_rllib_checkpoint(split_dir: str, template_checkpoint: Optional[str]) -> str:
    """Load tools/pack_split_to_rllib_checkpoint.py without requiring tools as a package."""
    tool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "pack_split_to_rllib_checkpoint.py")
    spec = importlib.util.spec_from_file_location("pack_split_to_rllib_checkpoint", tool_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.pack_split_to_rllib_checkpoint(split_dir, template_checkpoint)


def main():
    cfg = parse_args()
    _set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = to_torch_device(cfg.device)

    env = soccer_twos.make(
        variation=soccer_twos.EnvType.team_vs_policy,
        multiagent=False,
        flatten_branched=True,
        single_player=True,
        opponent_policy=lambda *_: 0,
    )

    action_dim, nvec = _resolve_action_dim(env.action_space)
    student = PrivilegedStudent(
        obs_dim=cfg.obs_dim,
        privileged_dim=cfg.privileged_dim,
        action_dim=action_dim,
        latent_dim=cfg.latent_dim,
        encoder_hiddens=cfg.encoder_hiddens,
        actor_hiddens=cfg.actor_hiddens,
        activation=cfg.activation,
    )
    if cfg.init_split_dir:
        load_student_from_split_dir(student, cfg.init_split_dir)
        print(f"[DAgger] Loaded student weights from {cfg.init_split_dir}")
    expert = create_expert_agent(cfg.expert_module, env)
    dataset = AggregatedDataset.empty(
        obs_dim=cfg.obs_dim,
        privileged_dim=cfg.privileged_dim,
        max_size=cfg.max_dataset_size,
    )

    for it in range(cfg.dagger_iters):
        beta = compute_beta(it, cfg.beta_start, cfg.beta_end, cfg.dagger_iters)
        iteration_data = collect_iteration_data(
            env=env,
            expert=expert,
            student=student,
            beta=beta,
            episodes=cfg.episodes_per_iter,
            max_steps=cfg.max_steps_per_episode,
            nvec=nvec,
            randomize_initial_state=cfg.randomize_initial_state,
            obs_dim=cfg.obs_dim,
            action_dim=action_dim,
        )
        dataset.add(
            obs=iteration_data["obs"],
            privileged=iteration_data["privileged"],
            actions=iteration_data["actions"],
        )
        dataset.save_npz(os.path.join(cfg.out_dir, f"dataset_iter_{it:03d}.npz"))
        mean_loss = train_student(
            student=student,
            dataset=dataset,
            device=device,
            epochs=cfg.student_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            label_smoothing=cfg.label_smoothing,
        )
        print(
            f"[DAgger] iter={it:03d} beta={beta:.4f} "
            f"new_samples={len(iteration_data['actions'])} total_samples={len(dataset)} "
            f"mean_loss={mean_loss:.6f}"
        )

    export_student(student, cfg.out_dir)
    split_dir = os.path.join(cfg.out_dir, "split_stage1")
    if cfg.pack_rllib_checkpoint:
        ckpt = _pack_rllib_checkpoint(split_dir, cfg.rllib_template_checkpoint)
        print(f"Packed RLlib checkpoint for previleged_agent: {ckpt}")
        print(f"Example: export CHECKPOINT_PATH={ckpt}")
    env.close()
    print(f"Exported DAgger student artifacts to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
