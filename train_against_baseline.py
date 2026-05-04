import argparse
import gym
import os
import pickle
import random
import re
import time
from datetime import datetime
from glob import glob

import numpy as np
import ray
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.registry import get_trainable_cls
from soccer_twos import EnvType
import soccer_twos

from curriculum_sampling import sample_player, sample_pos_vel
from custom_env_wrapper import (
    CustomEnvWrapper,
    _compute_worker_id,
    create_rllib_env as create_single_policy_rllib_env,
)
from utils import create_rllib_env as utils_create_rllib_env
from dagger_utils import create_expert_agent, flatten_multidiscrete_action
from models.rllib_registration import register_privileged_actor_model
from tools.export_stage1_privileged_weights import export_split_weights


NUM_ENVS_PER_WORKER = 3
DEFAULT_BASELINE_MODULE = "ceia_baseline_agent"
DEFAULT_INIT_POLICY_CHECKPOINT_PRIVILEGED = (
    "./ray_results/PPO_curriculum/Privileged_Latent_Dim_16_Curriculum/checkpoint_000113/checkpoint-113"
)
# Matches train_ray_curriculum.py (FC PPO); resolve_checkpoint_path picks latest checkpoint-* under this tree.
DEFAULT_INIT_POLICY_CHECKPOINT_RAW = "./ray_results/PPO_curriculum/Baseline_Curriculum"
_INIT_CHECKPOINT_AUTO = "AUTO"

INIT_CHECKPOINT_PATH = None
INIT_MODEL_CONFIG = None
INIT_ENV_CONFIG = None
DEFAULT_RANDOM_RESET_PROB = 0.3
DEFAULT_RANDOM_RESET_RANGES = {
    "ball": {
        "position": {"x": [-8.0, 8.0], "y": [-4.5, 4.5]},
        "velocity": {"x": [-2.0, 2.0], "y": [-2.0, 2.0]},
    },
    "players": {
        0: {
            "position": {"x": [-14.0, -2.0], "y": [-5.0, 5.0]},
            "velocity": {"x": [-1.5, 1.5], "y": [-1.5, 1.5]},
            "rotation_y": [0.0, 360.0],
        },
        1: {
            "position": {"x": [-14.0, -2.0], "y": [-5.0, 5.0]},
            "velocity": {"x": [-1.5, 1.5], "y": [-1.5, 1.5]},
            "rotation_y": [0.0, 360.0],
        },
        2: {
            "position": {"x": [2.0, 14.0], "y": [-5.0, 5.0]},
            "velocity": {"x": [-1.5, 1.5], "y": [-1.5, 1.5]},
            "rotation_y": [0.0, 360.0],
        },
        3: {
            "position": {"x": [2.0, 14.0], "y": [-5.0, 5.0]},
            "velocity": {"x": [-1.5, 1.5], "y": [-1.5, 1.5]},
            "rotation_y": [0.0, 360.0],
        },
    },
}


def _extract_checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def resolve_checkpoint_path(path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(expanded):
        return expanded
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Checkpoint path does not exist: {expanded}")
    if not os.path.isdir(expanded):
        raise ValueError(f"Checkpoint path must be a file or directory: {expanded}")

    direct = sorted(
        glob(os.path.join(expanded, "checkpoint-*")),
        key=lambda p: (_extract_checkpoint_step(p), os.path.getmtime(p)),
    )
    direct = [p for p in direct if os.path.isfile(p)]
    if direct:
        return direct[-1]

    recursive = sorted(
        glob(os.path.join(expanded, "**", "checkpoint-*"), recursive=True),
        key=lambda p: (_extract_checkpoint_step(p), os.path.getmtime(p)),
    )
    recursive = [p for p in recursive if os.path.isfile(p)]
    if recursive:
        return recursive[-1]

    raise FileNotFoundError(
        "Could not resolve an RLlib checkpoint file under path: "
        f"{expanded}"
    )


class BlueTeamVsBaselineWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(
        self,
        env,
        baseline_module: str,
        obs_mode: str,
        expected_obs_dim: int,
        enable_reward_shaping=False,
        prox_weight=0.05,
        goal_weight=0.08,
        shaping_clip=0.05,
        d_prox_max=20.0,
        x_min=-15.0,
        x_max=15.0,
        disable_shaping_on_terminal=True,
        random_reset_prob=DEFAULT_RANDOM_RESET_PROB,
        random_reset_ranges=None,
    ):
        super().__init__(env)
        self.env = env
        self._obs_wrapper = CustomEnvWrapper(
            env,
            obs_mode=obs_mode,
            expected_obs_dim=expected_obs_dim,
            enable_reward_shaping=enable_reward_shaping,
            prox_weight=prox_weight,
            goal_weight=goal_weight,
            shaping_clip=shaping_clip,
            d_prox_max=d_prox_max,
            x_min=x_min,
            x_max=x_max,
            disable_shaping_on_terminal=disable_shaping_on_terminal,
        )
        self._baseline = create_expert_agent(baseline_module, env)
        self.observation_space = self._obs_wrapper.observation_space
        self.action_space = env.action_space
        self._last_raw_obs = None
        self._train_team = 0
        self.random_reset_prob = float(np.clip(random_reset_prob, 0.0, 1.0))
        self.random_reset_ranges = random_reset_ranges or DEFAULT_RANDOM_RESET_RANGES
        self._flatten_nvec = self._infer_flatten_nvec(env)

    def reset(self):
        self._maybe_randomize_reset()
        self._last_raw_obs = self.env.reset()
        self._train_team = int(np.random.randint(0, 2))
        self._obs_wrapper._reset_potentials()
        transformed = self._obs_wrapper._transform_obs(self._last_raw_obs, info=None)
        train_ids = self._team_agent_ids(self._train_team)
        return {
            0: transformed[train_ids[0]],
            1: transformed[train_ids[1]],
        }

    @staticmethod
    def _team_agent_ids(team_id: int):
        return (0, 1) if int(team_id) == 0 else (2, 3)

    @staticmethod
    def _infer_flatten_nvec(env):
        flattener = getattr(env, "_flattener", None)
        action_lookup = getattr(flattener, "action_lookup", None)
        if not isinstance(action_lookup, dict) or not action_lookup:
            return None
        sample = next(iter(action_lookup.values()))
        sample_arr = np.asarray(sample, dtype=np.int64).reshape(-1)
        if sample_arr.size <= 0:
            return None
        nvec = np.zeros(sample_arr.size, dtype=np.int64)
        for action in action_lookup.values():
            arr = np.asarray(action, dtype=np.int64).reshape(-1)
            if arr.size != sample_arr.size:
                continue
            nvec = np.maximum(nvec, arr)
        return nvec + 1

    def _infer_nvec_from_action(self, action):
        arr = np.asarray(action)
        if arr.ndim == 0:
            return None
        flat = arr.reshape(-1)
        k = int(flat.shape[0])
        if k <= 0 or not isinstance(self.action_space, gym.spaces.Discrete):
            return None
        action_dim = int(self.action_space.n)
        min_base = max(int(np.max(flat)) + 1, 2)
        for base in range(min_base, 11):
            if (base ** k) == action_dim:
                return np.asarray([base] * k, dtype=np.int64)
        return None

    def _coerce_env_action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            nvec = self._flatten_nvec
            if nvec is None:
                nvec = self._infer_nvec_from_action(action)
                if nvec is not None:
                    self._flatten_nvec = nvec
            return flatten_multidiscrete_action(action, nvec)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            arr = np.asarray(action, dtype=np.int64).reshape(-1)
            if arr.size == 0:
                return np.zeros(len(self.action_space.nvec), dtype=np.int64)
            if arr.size != len(self.action_space.nvec):
                padded = np.zeros(len(self.action_space.nvec), dtype=np.int64)
                padded[: min(arr.size, len(self.action_space.nvec))] = arr[
                    : min(arr.size, len(self.action_space.nvec))
                ]
                arr = padded
            return arr
        if isinstance(action, np.ndarray) and action.ndim == 0:
            return int(action.item())
        if np.isscalar(action):
            return int(action)
        return action

    @staticmethod
    def _resolve_env_channel(env):
        current = env
        for _ in range(4):
            channel = getattr(current, "env_channel", None)
            if channel is not None:
                return channel
            current = getattr(current, "env", None)
            if current is None:
                break
        return None

    def _sample_random_players(self):
        sampled = {}
        for pid, cfg in self.random_reset_ranges["players"].items():
            sampled[int(pid)] = sample_player(cfg)
        return sampled

    def _maybe_randomize_reset(self):
        if self.random_reset_prob <= 0.0:
            return
        if random.random() >= self.random_reset_prob:
            return
        env_channel = self._resolve_env_channel(self.env)
        if env_channel is None:
            return
        ball_state = sample_pos_vel(self.random_reset_ranges["ball"])
        players_states = self._sample_random_players()
        env_channel.set_parameters(
            ball_state=ball_state,
            players_states=players_states,
        )

    def step(self, action_dict):
        if self._last_raw_obs is None:
            raise RuntimeError("Environment must be reset before calling step().")

        train_ids = self._team_agent_ids(self._train_team)
        baseline_ids = self._team_agent_ids(1 - self._train_team)
        opponent_obs = {
            0: np.asarray(self._last_raw_obs[baseline_ids[0]], dtype=np.float32),
            1: np.asarray(self._last_raw_obs[baseline_ids[1]], dtype=np.float32),
        }
        opponent_actions = self._baseline.act(opponent_obs)

        env_action = {
            train_ids[0]: self._coerce_env_action(action_dict[0]),
            train_ids[1]: self._coerce_env_action(action_dict[1]),
            baseline_ids[0]: self._coerce_env_action(
                opponent_actions[0] if 0 in opponent_actions else opponent_actions["0"]
            ),
            baseline_ids[1]: self._coerce_env_action(
                opponent_actions[1] if 1 in opponent_actions else opponent_actions["1"]
            ),
        }
        raw_obs, raw_rew, raw_done, raw_info = self.env.step(env_action)
        self._last_raw_obs = raw_obs
        transformed = self._obs_wrapper._transform_obs(raw_obs, raw_info)
        shaped_rew = self._obs_wrapper._shape_reward(raw_obs, raw_rew, raw_info, raw_done)

        obs = {
            0: transformed[train_ids[0]],
            1: transformed[train_ids[1]],
        }
        rew = {
            0: shaped_rew[train_ids[0]],
            1: shaped_rew[train_ids[1]],
        }
        done = {
            0: bool(raw_done.get("__all__", False)),
            1: bool(raw_done.get("__all__", False)),
            "__all__": bool(raw_done.get("__all__", False)),
        }
        info = {
            0: dict(raw_info[train_ids[0]]),
            1: dict(raw_info[train_ids[1]]),
        }
        info[0]["reward_metrics"] = dict(
            self._obs_wrapper._last_reward_metrics.get(train_ids[0], {})
        )
        info[1]["reward_metrics"] = dict(
            self._obs_wrapper._last_reward_metrics.get(train_ids[1], {})
        )
        return obs, rew, done, info


def create_rllib_env_against_baseline(env_config: dict = None):
    cfg = (
        {k: v for k, v in env_config.items()}
        if hasattr(env_config, "items")
        else dict(env_config or {})
    )
    cfg["worker_id"] = _compute_worker_id(env_config or {}, cfg)

    obs_mode = cfg.pop("obs_mode", "privileged_dict")
    expected_obs_dim = int(cfg.pop("expected_obs_dim", 336))
    baseline_module = cfg.pop("baseline_module", DEFAULT_BASELINE_MODULE)
    enable_reward_shaping = bool(cfg.pop("enable_reward_shaping", False))
    prox_weight = float(cfg.pop("prox_weight", 0.05))
    goal_weight = float(cfg.pop("goal_weight", 0.08))
    shaping_clip = float(cfg.pop("shaping_clip", 0.05))
    d_prox_max = float(cfg.pop("d_prox_max", 20.0))
    x_min = float(cfg.pop("x_min", -15.0))
    x_max = float(cfg.pop("x_max", 15.0))
    disable_shaping_on_terminal = bool(cfg.pop("disable_shaping_on_terminal", True))
    random_reset_prob = float(cfg.pop("random_reset_prob", DEFAULT_RANDOM_RESET_PROB))
    random_reset_ranges = cfg.pop("random_reset_ranges", DEFAULT_RANDOM_RESET_RANGES)

    cfg.setdefault("variation", EnvType.multiagent_player)
    cfg.setdefault("multiagent", True)
    cfg.setdefault("flatten_branched", True)
    cfg.pop("single_player", None)
    cfg.pop("opponent_policy", None)

    base_env = soccer_twos.make(**cfg)
    return BlueTeamVsBaselineWrapper(
        base_env,
        baseline_module=baseline_module,
        obs_mode=obs_mode,
        expected_obs_dim=expected_obs_dim,
        enable_reward_shaping=enable_reward_shaping,
        prox_weight=prox_weight,
        goal_weight=goal_weight,
        shaping_clip=shaping_clip,
        d_prox_max=d_prox_max,
        x_min=x_min,
        x_max=x_max,
        disable_shaping_on_terminal=disable_shaping_on_terminal,
        random_reset_prob=random_reset_prob,
        random_reset_ranges=random_reset_ranges,
    )


class BaselineTrainCallback(DefaultCallbacks):
    _REWARD_KEYS = ("base", "prox", "goal", "shape_total", "total")
    _did_warmstart = False
    # Set from main before tune.run: must match whether this run uses privileged_actor_model.
    train_uses_privileged_model = True

    def on_episode_start(self, **kwargs):
        episode = kwargs["episode"]
        for key in self._REWARD_KEYS:
            episode.user_data[f"train_{key}"] = []

    def on_episode_step(self, **kwargs):
        episode = kwargs["episode"]
        for slot_id in (0, 1):
            last_info = episode.last_info_for(slot_id)
            if not isinstance(last_info, dict):
                continue
            metrics = last_info.get("reward_metrics")
            if not isinstance(metrics, dict):
                continue
            for key in self._REWARD_KEYS:
                value = metrics.get(key)
                if isinstance(value, (int, float, np.floating, np.integer)):
                    episode.user_data[f"train_{key}"].append(float(value))

    def on_episode_end(self, **kwargs):
        episode = kwargs["episode"]
        for key in self._REWARD_KEYS:
            train_values = episode.user_data.get(f"train_{key}", [])
            if train_values:
                episode.custom_metrics[f"train_{key}_mean"] = float(
                    np.mean(train_values)
                )

    @staticmethod
    def _find_params_path(checkpoint_path):
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        cur = checkpoint_dir
        for _ in range(5):
            candidate = os.path.join(cur, "params.pkl")
            if os.path.exists(candidate):
                return candidate
            cur = os.path.dirname(cur)
        return None

    @staticmethod
    def _read_source_configs(checkpoint_path):
        params_path = BaselineTrainCallback._find_params_path(checkpoint_path)
        if params_path is None:
            raise ValueError(
                f"Could not find params.pkl near: {checkpoint_path}. "
                "Searched checkpoint and up to 4 parent directories."
            )
        with open(params_path, "rb") as f:
            src_cfg = pickle.load(f)
        return (src_cfg.get("model", {}) or {}, src_cfg.get("env_config", {}) or {})

    @staticmethod
    def _load_checkpoint_policy_weights(checkpoint_path):
        params_path = BaselineTrainCallback._find_params_path(checkpoint_path)
        if params_path is None:
            raise ValueError(
                f"Could not find params.pkl near: {checkpoint_path}. "
                "Searched checkpoint and up to 4 parent directories."
            )

        with open(params_path, "rb") as f:
            src_cfg = pickle.load(f)

        global INIT_MODEL_CONFIG, INIT_ENV_CONFIG
        INIT_MODEL_CONFIG = src_cfg.get("model", {}) or {}
        INIT_ENV_CONFIG = src_cfg.get("env_config", {}) or {}
        uses_privileged_actor = (
            (INIT_MODEL_CONFIG.get("custom_model") == "privileged_actor_model")
        )
        src_cfg["num_workers"] = 0
        src_cfg["num_gpus"] = 0
        # Baseline curriculum FC runs use utils.create_rllib_env (no CustomEnvWrapper).
        src_cfg["env"] = (
            "SoccerWarmStart" if uses_privileged_actor else "SoccerWarmStartFC"
        )
        src_env_cfg = dict(src_cfg.get("env_config", {}) or {})
        src_env_cfg["worker_id_offset"] = int(time.time() * 1000) % 50000 + 20000
        if uses_privileged_actor:
            src_env_cfg["worker_make_retries"] = 8
            src_env_cfg["worker_retry_stride"] = 1000
        if "base_port" not in src_env_cfg:
            src_env_cfg["base_port"] = 5005 + random.randint(0, 200) * 10
        src_cfg["env_config"] = src_env_cfg

        src_trainer = get_trainable_cls("PPO")(env=src_cfg["env"], config=src_cfg)
        src_trainer.restore(checkpoint_path)
        src_policy = src_trainer.get_policy("default") or src_trainer.get_policy("default_policy") or src_trainer.get_policy()
        if src_policy is None:
            raise ValueError("Could not find default policy in source checkpoint")
        return src_policy.get_weights()

    @staticmethod
    def _checkpoint_uses_privileged_actor(checkpoint_path: str) -> bool:
        params_path = BaselineTrainCallback._find_params_path(checkpoint_path)
        if params_path is None:
            return False
        with open(params_path, "rb") as f:
            src_cfg = pickle.load(f)
        model = src_cfg.get("model") or {}
        return model.get("custom_model") == "privileged_actor_model"

    def on_train_result(self, **info):
        if INIT_CHECKPOINT_PATH and not BaselineTrainCallback._did_warmstart:
            init_priv = BaselineTrainCallback._checkpoint_uses_privileged_actor(
                INIT_CHECKPOINT_PATH
            )
            train_priv = BaselineTrainCallback.train_uses_privileged_model
            if init_priv != train_priv:
                print(
                    "---- Skipping warm-start: checkpoint policy architecture does not match "
                    "this run. Init checkpoint is "
                    f"{'privileged_actor_model' if init_priv else 'default FC net'}, "
                    f"but this training run uses "
                    f"{'privileged_actor_model' if train_priv else 'raw FC net'}. "
                    "Use --init-policy-checkpoint from a compatible run, or omit it. ----"
                )
                BaselineTrainCallback._did_warmstart = True
            else:
                print(f"---- Warm-starting from: {INIT_CHECKPOINT_PATH} ----")
                src_weights = BaselineTrainCallback._load_checkpoint_policy_weights(
                    INIT_CHECKPOINT_PATH
                )
                trainer = info["trainer"]
                trainer.get_policy("default").set_weights(src_weights)
                BaselineTrainCallback._did_warmstart = True
                print("---- Warm-start complete ----")

        default_policy = info["trainer"].get_policy("default")
        if default_policy is not None and hasattr(default_policy.model, "last_latent_mean"):
            info["result"].setdefault("custom_metrics", {})
            info["result"]["custom_metrics"]["latent_mean"] = (
                default_policy.model.last_latent_mean
            )
            info["result"]["custom_metrics"]["latent_std"] = (
                default_policy.model.last_latent_std
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a shared privileged PPO policy for agents 0 and 1 against ceia_baseline_agent."
    )
    parser.add_argument(
        "--baseline-module",
        type=str,
        default=DEFAULT_BASELINE_MODULE,
        help="Module name for the fixed opponent agent.",
    )
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume the exact same trainer state.",
    )
    parser.add_argument(
        "--init-policy-checkpoint",
        type=str,
        default=_INIT_CHECKPOINT_AUTO,
        help=(
            "Checkpoint file or directory for warm-start. Use AUTO (default): privileged curriculum "
            "checkpoint for --obs-mode privileged_dict, or ray_results/PPO_curriculum/Baseline_Curriculum "
            "for raw. Pass \"\" to disable."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="./ray_results",
        help="Root directory where per-run result folders are created.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional tag used in the per-run folder name.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=-1,
        help="soccer_twos / ML-Agents base port. Use -1 for auto.",
    )
    parser.add_argument(
        "--worker-id-offset",
        type=int,
        default=-1,
        help="Offset added to RLlib worker ids. Use -1 for auto-derived value.",
    )
    parser.add_argument(
        "--prox-weight",
        type=float,
        default=0.3,
        help="Reward shaping weight for agent-to-ball proximity.",
    )
    parser.add_argument(
        "--goal-weight",
        type=float,
        default=0.5,
        help="Reward shaping weight for ball progress toward opponent goal.",
    )
    parser.add_argument(
        "--shaping-clip",
        type=float,
        default=1.0,
        help="Clip value applied to prox+goal shaping reward.",
    )
    parser.add_argument(
        "--d-prox-max",
        type=float,
        default=8.0,
        help="Distance scale for proximity shaping.",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=5000,
        help="PPO rollout_fragment_length.",
    )
    parser.add_argument(
        "--num-sgd-iter",
        type=int,
        default=30,
        help="PPO num_sgd_iter.",
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=128,
        help="PPO sgd_minibatch_size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4000,
        help="PPO train_batch_size (must be >= sgd_minibatch_size).",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="privileged_dict",
        choices=["raw", "privileged_dict"],
        help="raw: vector observations only (no privileged features); "
        "privileged_dict: obs dict with privileged vector for privileged_actor_model.",
    )
    args = parser.parse_args()

    use_privileged_model = args.obs_mode == "privileged_dict"
    BaselineTrainCallback.train_uses_privileged_model = use_privileged_model

    ray.init()
    register_privileged_actor_model()
    tune.registry.register_env("SoccerWarmStart", create_single_policy_rllib_env)
    tune.registry.register_env("SoccerWarmStartFC", utils_create_rllib_env)
    tune.registry.register_env("SoccerAgainstBaseline", create_rllib_env_against_baseline)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_tag if args.run_tag else (
        "privileged_vs_baseline" if use_privileged_model else "raw_vs_baseline"
    )
    run_dir = os.path.join(args.results_root, f"{run_label}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    restore_checkpoint = args.restore_checkpoint.strip() or None

    init_arg = (args.init_policy_checkpoint or "").strip()
    init_used_auto_default = init_arg.upper() == _INIT_CHECKPOINT_AUTO
    if init_used_auto_default:
        init_policy_checkpoint_str = (
            DEFAULT_INIT_POLICY_CHECKPOINT_PRIVILEGED
            if use_privileged_model
            else DEFAULT_INIT_POLICY_CHECKPOINT_RAW
        )
    elif init_arg == "":
        init_policy_checkpoint_str = ""
    else:
        init_policy_checkpoint_str = init_arg

    init_policy_checkpoint = None
    if init_policy_checkpoint_str:
        try:
            init_policy_checkpoint = resolve_checkpoint_path(init_policy_checkpoint_str)
        except FileNotFoundError:
            if init_used_auto_default:
                print(
                    f"[train_against_baseline] No checkpoint found under "
                    f"{init_policy_checkpoint_str!r}; continuing without warm-start."
                )
                init_policy_checkpoint = None
            else:
                raise

    if restore_checkpoint:
        restore_checkpoint = resolve_checkpoint_path(restore_checkpoint)
    if init_policy_checkpoint and not use_privileged_model:
        if BaselineTrainCallback._checkpoint_uses_privileged_actor(
            init_policy_checkpoint
        ):
            print(
                "Ignoring --init-policy-checkpoint: raw (FC) training cannot load weights "
                "from privileged_actor_model. Continuing with random initialization. "
                "Pass a checkpoint trained with --obs-mode raw (same FC architecture) to "
                "warm-start, or use --init-policy-checkpoint \"\" explicitly."
            )
            init_policy_checkpoint = None
    if restore_checkpoint and init_policy_checkpoint:
        raise ValueError("Use either --restore-checkpoint or --init-policy-checkpoint, not both.")
    if restore_checkpoint:
        print(f"Restoring full trainer state from checkpoint: {restore_checkpoint}")
    if init_policy_checkpoint:
        print(f"Initializing policy weights from checkpoint: {init_policy_checkpoint}")
        INIT_CHECKPOINT_PATH = init_policy_checkpoint
        INIT_MODEL_CONFIG, INIT_ENV_CONFIG = BaselineTrainCallback._read_source_configs(
            init_policy_checkpoint
        )
    print(f"Saving run outputs under: {run_dir}")

    base_port = (
        args.base_port if args.base_port > 0 else 5005 + random.randint(0, 200) * 10
    )
    worker_id_offset = (
        args.worker_id_offset
        if args.worker_id_offset >= 0
        else int(time.time() * 1000) % 50000 + 1000
    )
    print(f"Using base_port={base_port}, worker_id_offset={worker_id_offset}")

    flatten_branched = bool((INIT_ENV_CONFIG or {}).get("flatten_branched", True))
    src_custom_cfg = ((INIT_MODEL_CONFIG or {}).get("custom_model_config", {}) or {})
    latent_dim = int(src_custom_cfg.get("latent_dim", 16))
    encoder_hiddens = src_custom_cfg.get("encoder_hiddens", [64, 64])
    actor_hiddens = src_custom_cfg.get("actor_hiddens", [256, 256])
    value_hiddens = src_custom_cfg.get("value_hiddens", [256, 256])
    activation = src_custom_cfg.get("activation", "relu")
    if INIT_MODEL_CONFIG and not use_privileged_model:
        fc_h = INIT_MODEL_CONFIG.get("fcnet_hiddens")
        if fc_h:
            actor_hiddens = list(fc_h)
        fc_act = INIT_MODEL_CONFIG.get("fcnet_activation")
        if fc_act:
            activation = fc_act

    probe_cfg = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player,
        "multiagent": True,
        "flatten_branched": flatten_branched,
        "obs_mode": args.obs_mode,
        "expected_obs_dim": 336,
        "enable_reward_shaping": True,
        "prox_weight": args.prox_weight,
        "goal_weight": args.goal_weight,
        "shaping_clip": args.shaping_clip,
        "d_prox_max": args.d_prox_max,
        "x_min": -15.0,
        "x_max": 15.0,
        "disable_shaping_on_terminal": True,
        "random_reset_prob": DEFAULT_RANDOM_RESET_PROB,
        "random_reset_ranges": DEFAULT_RANDOM_RESET_RANGES,
        "baseline_module": args.baseline_module,
        "base_port": base_port,
        "worker_id_offset": worker_id_offset,
    }
    temp_env = create_rllib_env_against_baseline(probe_cfg)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_vs_baseline",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": BaselineTrainCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda *_: "default"),
                "policies_to_train": ["default"],
            },
            "env": "SoccerAgainstBaseline",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
                "multiagent": True,
                "flatten_branched": flatten_branched,
                "obs_mode": args.obs_mode,
                "expected_obs_dim": 336,
                "enable_reward_shaping": True,
                "prox_weight": args.prox_weight,
                "goal_weight": args.goal_weight,
                "shaping_clip": args.shaping_clip,
                "d_prox_max": args.d_prox_max,
                "x_min": -15.0,
                "x_max": 15.0,
                "disable_shaping_on_terminal": True,
                "random_reset_prob": DEFAULT_RANDOM_RESET_PROB,
                "random_reset_ranges": DEFAULT_RANDOM_RESET_RANGES,
                "baseline_module": args.baseline_module,
                "base_port": base_port,
                "worker_id_offset": worker_id_offset,
            },
            "model": (
                {
                    "custom_model": "privileged_actor_model",
                    "custom_model_config": {
                        "obs_dim": 336,
                        "privileged_dim": 25,
                        "latent_dim": latent_dim,
                        "encoder_hiddens": encoder_hiddens,
                        "actor_hiddens": actor_hiddens,
                        "value_hiddens": value_hiddens,
                        "activation": activation,
                    },
                }
                if use_privileged_model
                else {
                    "vf_share_layers": True,
                    "fcnet_hiddens": actor_hiddens,
                    "fcnet_activation": activation,
                }
            ),
            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": args.train_batch_size,
            "num_sgd_iter": args.num_sgd_iter,
            "sgd_minibatch_size": args.sgd_minibatch_size,
            "lr": args.lr,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 150000000,
            "time_total_s": 72000,
            "episode_reward_mean": 1.9,
        },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir=run_dir,
        restore=restore_checkpoint,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    if use_privileged_model:
        export_dir = os.path.join(os.path.dirname(best_checkpoint), "split_stage1")
        export_split_weights(best_checkpoint, export_dir, policy_name="default")
        print(f"Exported split stage-1 weights: {export_dir}")
    else:
        print(
            "Skipping split_stage1 export (only defined for privileged_actor_model). "
            f"Best checkpoint: {best_checkpoint}"
        )
    print("Done training")
