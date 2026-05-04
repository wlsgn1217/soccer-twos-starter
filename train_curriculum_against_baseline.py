import argparse
import copy
import gym
import os
import pickle
import random
import re
import time
from collections import deque
from datetime import datetime
from glob import glob

import numpy as np
import ray
from mlagents_envs.exception import UnityWorkerInUseException
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.registry import get_trainable_cls
from soccer_twos import EnvType
import soccer_twos
import yaml

from curriculum_sampling import sample_player, sample_pos_vel
from custom_env_wrapper import (
    CustomEnvWrapper,
    _compute_worker_id,
    create_rllib_env as create_single_policy_rllib_env,
)
from dagger_utils import create_expert_agent, flatten_multidiscrete_action
from models.rllib_registration import register_privileged_actor_model
from tools.export_stage1_privileged_weights import export_split_weights


NUM_ENVS_PER_WORKER = 3
DEFAULT_BASELINE_MODULE = "ceia_baseline_agent"
DEFAULT_CURRICULUM_YAML = "curriculum_for_vs_baseline.yaml"
DEFAULT_INIT_POLICY_CHECKPOINT = (
    "./ray_results/PPO_curriculum/Privileged_Latent_Dim_16_Curriculum/checkpoint_000113/checkpoint-113"
)

INIT_CHECKPOINT_PATH = None
INIT_MODEL_CONFIG = None
INIT_ENV_CONFIG = None


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


def load_curriculum(path: str) -> dict:
    with open(path, "r") as f:
        curriculum = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(curriculum, dict):
        raise ValueError("Curriculum YAML must contain a top-level mapping.")
    if "stages" not in curriculum or "scenarios" not in curriculum:
        raise ValueError("Curriculum YAML must define 'stages' and 'scenarios'.")
    return curriculum


def _sample_entity_states(spec: dict) -> dict:
    sampled = {}
    for key, cfg in spec.items():
        sampled[int(key)] = sample_player(cfg)
    return sampled


def _mirror_state(state: dict) -> dict:
    mirrored = copy.deepcopy(state)
    if "position" in mirrored:
        mirrored["position"][0] = -float(mirrored["position"][0])
    if "velocity" in mirrored:
        mirrored["velocity"][0] = -float(mirrored["velocity"][0])
    return mirrored


def _sample_curriculum_reset(curriculum: dict, stage_idx: int, train_team: int):
    stages = curriculum["stages"]
    scenarios = curriculum["scenarios"]
    stage = stages[int(max(0, min(stage_idx, len(stages) - 1)))]
    mixture = stage.get("mixture", {}) or {}
    names = list(mixture.keys())
    probs = np.asarray([float(mixture[name]) for name in names], dtype=np.float64)
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError(f"Stage '{stage.get('name', stage_idx)}' has zero total probability.")
    probs = probs / total
    choice = str(np.random.choice(names, p=probs))
    if choice == "default":
        return choice, None, None

    scenario = scenarios.get(choice)
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario '{choice}' missing from curriculum YAML.")

    ball_state = sample_pos_vel(scenario["ball"])
    train_states = _sample_entity_states(scenario["train_players"])
    opponent_states = _sample_entity_states(scenario["opponent_players"])
    if int(train_team) == 1:
        ball_state = _mirror_state(ball_state)
        train_states = {pid: _mirror_state(st) for pid, st in train_states.items()}
        opponent_states = {pid: _mirror_state(st) for pid, st in opponent_states.items()}

    players_states = {}
    if int(train_team) == 0:
        players_states[0] = train_states[0]
        players_states[1] = train_states[1]
        players_states[2] = opponent_states[0]
        players_states[3] = opponent_states[1]
    else:
        players_states[2] = train_states[0]
        players_states[3] = train_states[1]
        players_states[0] = opponent_states[0]
        players_states[1] = opponent_states[1]
    return choice, ball_state, players_states


class CurriculumVsBaselineWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(
        self,
        env,
        baseline_module: str,
        obs_mode: str,
        expected_obs_dim: int,
        curriculum: dict,
        enable_reward_shaping=False,
        prox_weight=0.05,
        goal_weight=0.08,
        shaping_clip=0.05,
        d_prox_max=20.0,
        x_min=-15.0,
        x_max=15.0,
        disable_shaping_on_terminal=True,
    ):
        super().__init__(env)
        self.env = env
        self.curriculum = curriculum
        self.current_stage = 0
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
        self._flatten_nvec = self._infer_flatten_nvec(env)
        self._last_reset_mode = "default"

    def set_curriculum_stage(self, stage_idx: int):
        self.current_stage = int(stage_idx)

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

    def _maybe_apply_curriculum_reset(self):
        env_channel = self._resolve_env_channel(self.env)
        if env_channel is None:
            self._last_reset_mode = "default"
            return
        self._last_reset_mode, ball_state, players_states = _sample_curriculum_reset(
            self.curriculum,
            self.current_stage,
            self._train_team,
        )
        if ball_state is None or players_states is None:
            return
        env_channel.set_parameters(ball_state=ball_state, players_states=players_states)

    def reset(self):
        self._train_team = int(np.random.randint(0, 2))
        self._maybe_apply_curriculum_reset()
        self._last_raw_obs = self.env.reset()
        self._obs_wrapper._reset_potentials()
        transformed = self._obs_wrapper._transform_obs(self._last_raw_obs, info=None)
        train_ids = self._team_agent_ids(self._train_team)
        return {0: transformed[train_ids[0]], 1: transformed[train_ids[1]]}

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

        obs = {0: transformed[train_ids[0]], 1: transformed[train_ids[1]]}
        rew = {0: shaped_rew[train_ids[0]], 1: shaped_rew[train_ids[1]]}
        all_done = bool(raw_done.get("__all__", False))
        done = {0: all_done, 1: all_done, "__all__": all_done}
        info = {
            0: dict(raw_info.get(train_ids[0], {})),
            1: dict(raw_info.get(train_ids[1], {})),
        }
        info[0]["reward_metrics"] = dict(self._obs_wrapper._last_reward_metrics.get(train_ids[0], {}))
        info[1]["reward_metrics"] = dict(self._obs_wrapper._last_reward_metrics.get(train_ids[1], {}))
        info[0]["curriculum_reset_mode"] = self._last_reset_mode
        info[1]["curriculum_reset_mode"] = self._last_reset_mode
        return obs, rew, done, info


def create_rllib_env_curriculum_baseline(env_config: dict = None):
    cfg = (
        {k: v for k, v in env_config.items()}
        if hasattr(env_config, "items")
        else dict(env_config or {})
    )
    cfg["worker_id"] = _compute_worker_id(env_config or {}, cfg)

    obs_mode = cfg.pop("obs_mode", "privileged_dict")
    expected_obs_dim = int(cfg.pop("expected_obs_dim", 336))
    baseline_module = cfg.pop("baseline_module", DEFAULT_BASELINE_MODULE)
    curriculum = cfg.pop("curriculum", None)
    if curriculum is None:
        raise ValueError("env_config must include 'curriculum'")
    enable_reward_shaping = bool(cfg.pop("enable_reward_shaping", False))
    prox_weight = float(cfg.pop("prox_weight", 0.05))
    goal_weight = float(cfg.pop("goal_weight", 0.08))
    shaping_clip = float(cfg.pop("shaping_clip", 0.05))
    d_prox_max = float(cfg.pop("d_prox_max", 20.0))
    x_min = float(cfg.pop("x_min", -15.0))
    x_max = float(cfg.pop("x_max", 15.0))
    disable_shaping_on_terminal = bool(cfg.pop("disable_shaping_on_terminal", True))
    max_make_retries = int(cfg.pop("worker_make_retries", 8))
    retry_worker_stride = int(cfg.pop("worker_retry_stride", 1000))
    if retry_worker_stride <= 0:
        retry_worker_stride = 1000

    cfg.setdefault("variation", EnvType.multiagent_player)
    cfg.setdefault("multiagent", True)
    cfg.setdefault("flatten_branched", True)

    base_env = None
    last_error = None
    for attempt in range(max_make_retries + 1):
        try:
            base_env = soccer_twos.make(**cfg)
            break
        except UnityWorkerInUseException as exc:
            last_error = exc
            if attempt >= max_make_retries:
                raise
            cfg["worker_id"] = int(cfg["worker_id"]) + retry_worker_stride
            print(
                "[curriculum-vs-baseline-env] worker_id collision; "
                f"retry {attempt + 1}/{max_make_retries} with worker_id={cfg['worker_id']}"
            )
    if base_env is None:
        raise last_error if last_error is not None else RuntimeError("Failed to create env")

    return CurriculumVsBaselineWrapper(
        base_env,
        baseline_module=baseline_module,
        obs_mode=obs_mode,
        expected_obs_dim=expected_obs_dim,
        curriculum=curriculum,
        enable_reward_shaping=enable_reward_shaping,
        prox_weight=prox_weight,
        goal_weight=goal_weight,
        shaping_clip=shaping_clip,
        d_prox_max=d_prox_max,
        x_min=x_min,
        x_max=x_max,
        disable_shaping_on_terminal=disable_shaping_on_terminal,
    )


class CurriculumAgainstBaselineCallback(DefaultCallbacks):
    _REWARD_KEYS = ("base", "prox", "goal", "shape_total", "total")
    _did_warmstart = False
    _stage_idx = 0
    _base_reward_window = None
    _ma_window = 5
    _ma_threshold = 1.0
    _num_stages = 1

    @staticmethod
    def configure_from_curriculum(curriculum: dict):
        ma_cfg = curriculum.get("moving_average", {}) or {}
        CurriculumAgainstBaselineCallback._ma_window = int(ma_cfg.get("window", 5))
        CurriculumAgainstBaselineCallback._ma_threshold = float(ma_cfg.get("threshold", 1.0))
        CurriculumAgainstBaselineCallback._num_stages = len(curriculum.get("stages", []))
        CurriculumAgainstBaselineCallback._base_reward_window = deque(
            maxlen=max(1, CurriculumAgainstBaselineCallback._ma_window)
        )
        CurriculumAgainstBaselineCallback._stage_idx = 0
        CurriculumAgainstBaselineCallback._did_warmstart = False

    @staticmethod
    def _broadcast_stage(trainer, stage_idx: int):
        trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.set_curriculum_stage(stage_idx)
                if hasattr(env, "set_curriculum_stage")
                else None
            )
        )

    def on_episode_start(self, **kwargs):
        episode = kwargs["episode"]
        for key in self._REWARD_KEYS:
            episode.user_data[f"train_{key}"] = []
        episode.user_data["curriculum_reset_modes"] = []

    def on_episode_step(self, **kwargs):
        episode = kwargs["episode"]
        for slot_id in (0, 1):
            last_info = episode.last_info_for(slot_id)
            if not isinstance(last_info, dict):
                continue
            metrics = last_info.get("reward_metrics")
            if isinstance(metrics, dict):
                for key in self._REWARD_KEYS:
                    value = metrics.get(key)
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        episode.user_data[f"train_{key}"].append(float(value))
            reset_mode = last_info.get("curriculum_reset_mode")
            if isinstance(reset_mode, str):
                episode.user_data["curriculum_reset_modes"].append(reset_mode)

    def on_episode_end(self, **kwargs):
        episode = kwargs["episode"]
        for key in self._REWARD_KEYS:
            values = episode.user_data.get(f"train_{key}", [])
            if values:
                if key == "base":
                    episode.custom_metrics["train_base_sum"] = float(np.sum(values))
                else:
                    episode.custom_metrics[f"train_{key}_mean"] = float(np.mean(values))
        modes = episode.user_data.get("curriculum_reset_modes", [])
        if modes:
            mode = modes[-1]
            episode.custom_metrics[f"reset_mode_{mode}"] = 1.0

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
        params_path = CurriculumAgainstBaselineCallback._find_params_path(checkpoint_path)
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
        params_path = CurriculumAgainstBaselineCallback._find_params_path(checkpoint_path)
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
        src_cfg["num_workers"] = 0
        src_cfg["num_gpus"] = 0
        src_cfg["env"] = "SoccerWarmStart"
        src_env_cfg = dict(src_cfg.get("env_config", {}) or {})
        src_env_cfg["worker_id_offset"] = int(time.time() * 1000) % 50000 + 20000
        src_env_cfg["worker_make_retries"] = 8
        src_env_cfg["worker_retry_stride"] = 1000
        if "base_port" not in src_env_cfg:
            src_env_cfg["base_port"] = 5005 + random.randint(0, 200) * 10
        src_cfg["env_config"] = src_env_cfg

        src_trainer = get_trainable_cls("PPO")(env=src_cfg["env"], config=src_cfg)
        src_trainer.restore(checkpoint_path)
        src_policy = (
            src_trainer.get_policy("default")
            or src_trainer.get_policy("default_policy")
            or src_trainer.get_policy()
        )
        if src_policy is None:
            raise ValueError("Could not find default policy in source checkpoint")
        return src_policy.get_weights()

    def on_train_start(self, *, trainer, **kwargs):
        if INIT_CHECKPOINT_PATH and not CurriculumAgainstBaselineCallback._did_warmstart:
            print(f"---- Warm-starting from: {INIT_CHECKPOINT_PATH} ----")
            src_weights = CurriculumAgainstBaselineCallback._load_checkpoint_policy_weights(
                INIT_CHECKPOINT_PATH
            )
            trainer.get_policy("default").set_weights(src_weights)
            CurriculumAgainstBaselineCallback._did_warmstart = True
            CurriculumAgainstBaselineCallback._broadcast_stage(
                trainer, CurriculumAgainstBaselineCallback._stage_idx
            )
            print("---- Warm-start complete ----")

    def on_train_result(self, **info):
        result = info["result"]
        trainer = info["trainer"]

        default_policy = trainer.get_policy("default")
        if default_policy is not None and hasattr(default_policy.model, "last_latent_mean"):
            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["latent_mean"] = default_policy.model.last_latent_mean
            result["custom_metrics"]["latent_std"] = default_policy.model.last_latent_std

        custom_metrics = result.get("custom_metrics", {})
        base_sum = custom_metrics.get("train_base_sum_mean")
        if base_sum is None:
            base_sum = custom_metrics.get("train_base_sum")
        if base_sum is not None and CurriculumAgainstBaselineCallback._base_reward_window is not None:
            CurriculumAgainstBaselineCallback._base_reward_window.append(float(base_sum))
        ma_values = list(CurriculumAgainstBaselineCallback._base_reward_window or [])
        ma = float(np.mean(ma_values)) if ma_values else float("nan")
        result["curriculum_stage"] = CurriculumAgainstBaselineCallback._stage_idx
        result["curriculum_stage_name"] = info["trainer"].config["env_config"]["curriculum"]["stages"][
            CurriculumAgainstBaselineCallback._stage_idx
        ]["name"]
        result["curriculum_base_reward_ma"] = ma

        if (
            len(ma_values) >= CurriculumAgainstBaselineCallback._ma_window
            and ma >= CurriculumAgainstBaselineCallback._ma_threshold
            and CurriculumAgainstBaselineCallback._stage_idx < CurriculumAgainstBaselineCallback._num_stages - 1
        ):
            CurriculumAgainstBaselineCallback._stage_idx += 1
            CurriculumAgainstBaselineCallback._base_reward_window.clear()
            CurriculumAgainstBaselineCallback._broadcast_stage(
                trainer, CurriculumAgainstBaselineCallback._stage_idx
            )
            print(
                "---- Advancing curriculum stage to "
                f"{CurriculumAgainstBaselineCallback._stage_idx} ----"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train privileged PPO against ceia_baseline_agent with a moving-average curriculum."
    )
    parser.add_argument("--curriculum", type=str, default=DEFAULT_CURRICULUM_YAML)
    parser.add_argument("--baseline-module", type=str, default=DEFAULT_BASELINE_MODULE)
    parser.add_argument("--restore-checkpoint", type=str, default="")
    parser.add_argument("--init-policy-checkpoint", type=str, default=DEFAULT_INIT_POLICY_CHECKPOINT)
    parser.add_argument("--results-root", type=str, default="./ray_results")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--base-port", type=int, default=-1)
    parser.add_argument("--worker-id-offset", type=int, default=-1)
    parser.add_argument("--prox-weight", type=float, default=0.0)
    parser.add_argument("--goal-weight", type=float, default=0.7)
    parser.add_argument("--shaping-clip", type=float, default=0.25)
    parser.add_argument("--d-prox-max", type=float, default=8.0)
    args = parser.parse_args()

    curriculum_path = os.path.abspath(args.curriculum)
    curriculum = load_curriculum(curriculum_path)
    CurriculumAgainstBaselineCallback.configure_from_curriculum(curriculum)

    ray.init()
    register_privileged_actor_model()
    tune.registry.register_env("SoccerWarmStart", create_single_policy_rllib_env)
    tune.registry.register_env("SoccerAgainstBaselineCurriculum", create_rllib_env_curriculum_baseline)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_tag if args.run_tag else "curriculum_vs_baseline"
    run_dir = os.path.join(args.results_root, f"{run_label}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    restore_checkpoint = args.restore_checkpoint.strip() or None
    init_policy_checkpoint = args.init_policy_checkpoint.strip() or None
    if restore_checkpoint:
        restore_checkpoint = resolve_checkpoint_path(restore_checkpoint)
    if init_policy_checkpoint:
        init_policy_checkpoint = resolve_checkpoint_path(init_policy_checkpoint)
    if restore_checkpoint and init_policy_checkpoint:
        raise ValueError("Use either --restore-checkpoint or --init-policy-checkpoint, not both.")
    if init_policy_checkpoint:
        INIT_CHECKPOINT_PATH = init_policy_checkpoint
        INIT_MODEL_CONFIG, INIT_ENV_CONFIG = CurriculumAgainstBaselineCallback._read_source_configs(
            init_policy_checkpoint
        )

    base_port = args.base_port if args.base_port > 0 else 5005 + random.randint(0, 200) * 10
    worker_id_offset = (
        args.worker_id_offset
        if args.worker_id_offset >= 0
        else int(time.time() * 1000) % 50000 + 1000
    )
    flatten_branched = bool((INIT_ENV_CONFIG or {}).get("flatten_branched", True))
    src_custom_cfg = ((INIT_MODEL_CONFIG or {}).get("custom_model_config", {}) or {})
    latent_dim = int(src_custom_cfg.get("latent_dim", 16))
    encoder_hiddens = src_custom_cfg.get("encoder_hiddens", [64, 64])
    actor_hiddens = src_custom_cfg.get("actor_hiddens", [256, 256])
    value_hiddens = src_custom_cfg.get("value_hiddens", [256, 256])
    activation = src_custom_cfg.get("activation", "relu")

    env_cfg = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player,
        "multiagent": True,
        "flatten_branched": flatten_branched,
        "obs_mode": "privileged_dict",
        "expected_obs_dim": 336,
        "curriculum": curriculum,
        "enable_reward_shaping": True,
        "prox_weight": args.prox_weight,
        "goal_weight": args.goal_weight,
        "shaping_clip": args.shaping_clip,
        "d_prox_max": args.d_prox_max,
        "x_min": -15.0,
        "x_max": 15.0,
        "disable_shaping_on_terminal": True,
        "worker_make_retries": 8,
        "worker_retry_stride": 1000,
        "baseline_module": args.baseline_module,
        "base_port": base_port,
        "worker_id_offset": worker_id_offset,
    }
    temp_env = create_rllib_env_curriculum_baseline(env_cfg)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum_vs_baseline",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumAgainstBaselineCallback,
            "multiagent": {
                "policies": {"default": (None, obs_space, act_space, {})},
                "policy_mapping_fn": tune.function(lambda *_: "default"),
                "policies_to_train": ["default"],
            },
            "env": "SoccerAgainstBaselineCurriculum",
            "env_config": env_cfg,
            "model": {
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
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 150000000, "time_total_s": 72000},
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
    export_dir = os.path.join(os.path.dirname(best_checkpoint), "split_stage1")
    export_split_weights(best_checkpoint, export_dir, policy_name="default")
    print(f"Exported split stage-1 weights: {export_dir}")
    print("Done training")
