import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
try:
    from soccer_twos import AgentInterface
except ModuleNotFoundError:  # Allows lightweight unit tests without env dependency.
    AgentInterface = object


def compute_beta(iteration: int, beta_start: float, beta_end: float, total_iters: int) -> float:
    """Linear beta schedule for DAgger mixture policy."""
    if total_iters <= 1:
        return float(beta_end)
    t = float(max(0, min(iteration, total_iters - 1))) / float(total_iters - 1)
    return float(beta_start + (beta_end - beta_start) * t)


@dataclass
class AggregatedDataset:
    """Simple aggregation buffer for DAgger with optional max size."""

    obs: np.ndarray
    privileged: np.ndarray
    actions: np.ndarray
    max_size: int = 0

    @classmethod
    def empty(cls, obs_dim: int, privileged_dim: int, max_size: int = 0):
        return cls(
            obs=np.zeros((0, obs_dim), dtype=np.float32),
            privileged=np.zeros((0, privileged_dim), dtype=np.float32),
            actions=np.zeros((0,), dtype=np.int64),
            max_size=int(max_size),
        )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def add(self, obs: np.ndarray, privileged: np.ndarray, actions: np.ndarray) -> None:
        if obs.shape[0] != privileged.shape[0] or obs.shape[0] != actions.shape[0]:
            raise ValueError("obs, privileged, and actions must have matching first dimension.")
        self.obs = np.concatenate([self.obs, obs.astype(np.float32)], axis=0)
        self.privileged = np.concatenate([self.privileged, privileged.astype(np.float32)], axis=0)
        self.actions = np.concatenate([self.actions, actions.astype(np.int64)], axis=0)
        self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        if self.max_size <= 0 or len(self) <= self.max_size:
            return
        keep = self.max_size
        idx = np.random.choice(len(self), size=keep, replace=False)
        idx = np.sort(idx)
        self.obs = self.obs[idx]
        self.privileged = self.privileged[idx]
        self.actions = self.actions[idx]

    def save_npz(self, path: str) -> None:
        np.savez_compressed(path, obs=self.obs, privileged=self.privileged, actions=self.actions)


def create_expert_agent(module_name: str, env: Any) -> AgentInterface:
    """
    Load an expert module and instantiate the first suitable agent class.

    Supports common naming conventions (RayAgent/TeamAgent/Agent) and generic
    `AgentInterface` subclasses.
    """
    module = importlib.import_module(module_name)
    preferred_names = ("RayAgent", "TeamAgent", "Agent", "PlayerAgent")

    for name in preferred_names:
        cls = getattr(module, name, None)
        if inspect.isclass(cls):
            return cls(env)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is AgentInterface:
            continue
        try:
            if issubclass(obj, AgentInterface):
                return obj(env)
        except TypeError:
            continue

    raise ValueError(f"Could not find an AgentInterface-compatible class in module '{module_name}'.")


def to_torch_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        # Guard against CUDA-visible GPUs that are unsupported by this torch build
        # (e.g., sm_89 device with an older wheel).
        try:
            major, minor = torch.cuda.get_device_capability(0)
            target_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            if target_arch not in supported_arches:
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device(device_str)


def flatten_multidiscrete_action(action: Any, nvec: Optional[np.ndarray]) -> int:
    """Convert action to flattened discrete index if needed."""
    if isinstance(action, (int, np.integer)):
        return int(action)
    if np.isscalar(action):
        return int(action)
    arr = np.asarray(action)
    if arr.ndim == 0:
        return int(arr.item())
    if nvec is None:
        flat = arr.reshape(-1)
        if flat.shape[0] == 1:
            return int(flat[0])
        raise ValueError("nvec is required to flatten non-scalar actions.")
    arr = arr.astype(np.int64).reshape(-1)
    if arr.shape[0] != len(nvec):
        padded = np.zeros(len(nvec), dtype=np.int64)
        copy_n = min(arr.shape[0], len(nvec))
        if copy_n > 0:
            padded[:copy_n] = arr[:copy_n]
        arr = padded
    idx = 0
    for a_i, n_i in zip(arr, nvec):
        idx = idx * int(n_i) + (int(a_i) % int(n_i))
    return int(idx)


def call_expert_single_action(expert, obs_input: Any, player_id: int = 0) -> Any:
    """
    Query expert action robustly across different AgentInterface implementations.
    """
    try:
        out = expert.act(obs_input)
    except Exception:
        # Avoid double-wrapping dict inputs like {0: obs} -> {0: {0: obs}}.
        if isinstance(obs_input, dict) and player_id in obs_input:
            out = expert.act(obs_input)
        else:
            out = expert.act({player_id: obs_input})
    if isinstance(out, dict):
        if player_id in out:
            return out[player_id]
        if str(player_id) in out:
            return out[str(player_id)]
        return next(iter(out.values()))
    return out
