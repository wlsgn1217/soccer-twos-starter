import gym
import numpy as np
import soccer_twos
from mlagents_envs.exception import UnityWorkerInUseException
from privileged_features import (
    PRIVILEGED_OBS_DIM,
    TEAM_PRIVILEGED_OBS_DIM,
    build_privileged_vector,
    build_team_privileged_vector,
)
from utils import RLLibWrapper


def _cfg_get(env_config, key, default=None):
    """Read RLlib EnvContext fields (mapping-first; supports zero values)."""
    if env_config is None:
        return default
    if hasattr(env_config, "__contains__") and hasattr(env_config, "__getitem__"):
        try:
            if key in env_config:
                return env_config[key]
        except (TypeError, KeyError):
            pass
    if hasattr(env_config, key):
        return getattr(env_config, key)
    return default


def _compute_worker_id(env_config, cfg):
    """Unique Unity worker id for (worker_index, vector_index)."""
    worker_id_offset = int(_cfg_get(env_config, "worker_id_offset", _cfg_get(cfg, "worker_id_offset", 0)))
    wi = _cfg_get(env_config, "worker_index", None)
    if wi is None:
        wi = _cfg_get(cfg, "worker_index", None)
    if wi is None:
        # Driver / probe env fallback to avoid colliding with worker 0 by default.
        return int(_cfg_get(cfg, "worker_id", 99)) + worker_id_offset

    n = _cfg_get(env_config, "num_envs_per_worker", None)
    if n is None:
        n = _cfg_get(cfg, "num_envs_per_worker", 1)
    n = int(n)

    vi = _cfg_get(env_config, "vector_index", None)
    if vi is None:
        vi = _cfg_get(cfg, "vector_index", None)
    if vi is None:
        vi = _cfg_get(env_config, "env_idx", None)
    if vi is None:
        vi = _cfg_get(cfg, "env_idx", None)
    if vi is None:
        vi = _cfg_get(env_config, "env_index", None)
    if vi is None:
        vi = _cfg_get(cfg, "env_index", 0)
    vi = int(vi)
    return worker_id_offset + int(wi) * n + vi


def _coerce_digit_keys(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
    return out


def _looks_like_soccer_info(d):
    if not isinstance(d, dict) or not d:
        return False
    k0 = 0 if 0 in d else ("0" if "0" in d else None)
    if k0 is None:
        return False
    v = d[k0]
    return isinstance(v, dict) and "player_info" in v


def _normalize_info(info):
    """
    Normalize info to soccer_twos layout:
    {0: {player_info, ball_info}, 1: ..., 2: ..., 3: ...}
    RLlib single-policy often wraps this as {'agent0': <inner-or-None>}.
    """
    if info is None:
        return None
    if _looks_like_soccer_info(info):
        return _coerce_digit_keys(info)
    if isinstance(info, dict) and "agent0" in info:
        inner = info["agent0"]
        if inner is None:
            return None
        inner = _coerce_digit_keys(inner)
        if _looks_like_soccer_info(inner):
            return inner
    return None


def _policy_id_to_player_id(policy_id):
    if isinstance(policy_id, int):
        return policy_id
    if isinstance(policy_id, str) and policy_id.startswith("agent"):
        return int(policy_id[5:])
    return int(policy_id)


class CustomEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obs_mode="privileged",
        expected_obs_dim=336,
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
        self.obs_mode = obs_mode
        self.expected_obs_dim = int(expected_obs_dim)
        self.enable_reward_shaping = bool(enable_reward_shaping)
        self.prox_weight = float(prox_weight)
        self.goal_weight = float(goal_weight)
        self.shaping_clip = float(shaping_clip)
        self.d_prox_max = float(max(d_prox_max, 1e-6))
        self.x_min = float(min(x_min, x_max))
        self.x_max = float(max(x_min, x_max))
        self.disable_shaping_on_terminal = bool(disable_shaping_on_terminal)
        self._prev_prox_phi = {}
        self._prev_goal_phi = {}
        self._last_reward_metrics = {}
        # Gym and RLlib read `observation_space`, not `obs_space`. `super().__init__` copied the
        # inner (336-d) space; override when this wrapper changes the emitted observation.
        if obs_mode == "privileged":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(PRIVILEGED_OBS_DIM,),
                dtype=np.float32,
            )
        elif obs_mode == "privileged_dict":
            self.observation_space = gym.spaces.Dict(
                {
                    "obs": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.expected_obs_dim,),
                        dtype=np.float32,
                    ),
                    "privileged": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(PRIVILEGED_OBS_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        elif obs_mode == "privileged_dict_team":
            self.observation_space = gym.spaces.Dict(
                {
                    "obs": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.expected_obs_dim,),
                        dtype=np.float32,
                    ),
                    "privileged": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(TEAM_PRIVILEGED_OBS_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        elif obs_mode == "raw":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.expected_obs_dim,),
                dtype=np.float32,
            )

    @staticmethod
    def _agent_id_to_team_id(agent_id: int) -> int:
        return 0 if int(agent_id) < 2 else 1

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._reset_potentials()
        return self._transform_obs(obs, info=None)

    def step(self, action_dict):
        obs, rew, done, info = self.env.step(action_dict)

        new_obs = self._transform_obs(obs, info)
        new_rew = self._shape_reward(obs, rew, info, done)

        return new_obs, new_rew, done, info

    def _transform_obs(self, obs_dict, info):
        # obs_dict: {agent_id: np.ndarray} in multiagent mode, or np.ndarray in single-player mode
        is_dict = isinstance(obs_dict, dict)

        if info is None:
            # reset(): no player_info / ball_info yet (populated after first step in soccer_twos)
            if self.obs_mode == "privileged":
                zeros = np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
                if not is_dict:
                    return zeros
                return {aid: zeros.copy() for aid in obs_dict}
            if self.obs_mode == "privileged_dict_team":
                if not is_dict:
                    obs = np.asarray(obs_dict, dtype=np.float32).reshape(-1)
                    self._validate_obs_dim(obs)
                    return {
                        "obs": obs,
                        "privileged": np.zeros(TEAM_PRIVILEGED_OBS_DIM, dtype=np.float32),
                    }
                out = {}
                for aid, o in obs_dict.items():
                    obs = np.asarray(o, dtype=np.float32).reshape(-1)
                    self._validate_obs_dim(obs)
                    out[aid] = {
                        "obs": obs,
                        "privileged": np.zeros(TEAM_PRIVILEGED_OBS_DIM, dtype=np.float32),
                    }
                return out
            if self.obs_mode == "privileged_dict":
                if not is_dict:
                    obs = np.asarray(obs_dict, dtype=np.float32).reshape(-1)
                    self._validate_obs_dim(obs)
                    return {
                        "obs": obs,
                        "privileged": np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32),
                    }
                out = {}
                for aid, o in obs_dict.items():
                    obs = np.asarray(o, dtype=np.float32).reshape(-1)
                    self._validate_obs_dim(obs)
                    out[aid] = {
                        "obs": obs,
                        "privileged": np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32),
                    }
                return out
            if self.obs_mode == "raw":
                if not is_dict:
                    obs = np.asarray(obs_dict, dtype=np.float32).reshape(-1)
                    self._validate_obs_dim(obs)
                    return obs
                return {
                    aid: np.asarray(o, dtype=np.float32).reshape(-1)
                    for aid, o in obs_dict.items()
                }
            if not is_dict:
                return np.asarray(obs_dict, dtype=np.float32)
            return {aid: np.asarray(o, dtype=np.float32) for aid, o in obs_dict.items()}

        if not is_dict:
            # single-player mode: obs is a plain array, info keyed by agent id 0
            if self.obs_mode == "privileged":
                info = _normalize_info(info)
                if info is None:
                    return np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
                return build_privileged_vector(info, agent_id=0)
            if self.obs_mode == "privileged_dict":
                obs = np.asarray(obs_dict, dtype=np.float32).reshape(-1)
                self._validate_obs_dim(obs)
                info = _normalize_info(info)
                privileged = (
                    np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
                    if info is None
                    else build_privileged_vector(info, agent_id=0)
                )
                return {"obs": obs, "privileged": privileged}
            if self.obs_mode == "privileged_dict_team":
                obs = np.asarray(obs_dict, dtype=np.float32).reshape(-1)
                self._validate_obs_dim(obs)
                info = _normalize_info(info)
                privileged = (
                    np.zeros(TEAM_PRIVILEGED_OBS_DIM, dtype=np.float32)
                    if info is None
                    else build_team_privileged_vector(info, team_id=0)
                )
                return {"obs": obs, "privileged": privileged}
            return np.asarray(obs_dict, dtype=np.float32)

        out = {}
        info_norm = _normalize_info(info)
        for policy_id, obs in obs_dict.items():
            if self.obs_mode == "privileged":
                if info_norm is None:
                    out[policy_id] = np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
                    continue
                agent_id = _policy_id_to_player_id(policy_id)
                out_agent = build_privileged_vector(info_norm, agent_id=agent_id)
            elif self.obs_mode == "privileged_dict":
                obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
                self._validate_obs_dim(obs_vec)
                agent_id = _policy_id_to_player_id(policy_id)
                privileged = (
                    np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
                    if info_norm is None
                    else build_privileged_vector(info_norm, agent_id=agent_id)
                )
                out_agent = {"obs": obs_vec, "privileged": privileged}
            elif self.obs_mode == "privileged_dict_team":
                obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
                self._validate_obs_dim(obs_vec)
                agent_id = _policy_id_to_player_id(policy_id)
                privileged = (
                    np.zeros(TEAM_PRIVILEGED_OBS_DIM, dtype=np.float32)
                    if info_norm is None
                    else build_team_privileged_vector(
                        info_norm, team_id=self._agent_id_to_team_id(agent_id)
                    )
                )
                out_agent = {"obs": obs_vec, "privileged": privileged}
            else:
                out_agent = np.asarray(obs, dtype=np.float32)
            out[policy_id] = out_agent

        return out

    def _validate_obs_dim(self, obs_vec):
        if obs_vec.shape[0] != self.expected_obs_dim:
            raise ValueError(
                f"Expected raw obs dim {self.expected_obs_dim}, got {obs_vec.shape[0]}"
            )

    def _shape_reward(self, obs_dict, rew_dict, info, done):
        if not isinstance(rew_dict, dict):
            return rew_dict

        info_norm = _normalize_info(info)
        done_all = False
        if isinstance(done, dict):
            done_all = bool(done.get("__all__", False))
        elif done is not None:
            done_all = bool(done)

        base_reward = {}
        for agent_id, rew in rew_dict.items():
            player_id = _policy_id_to_player_id(agent_id)
            base_reward[agent_id] = float(rew)
            self._last_reward_metrics[player_id] = {
                "base": float(rew),
                "prox": 0.0,
                "goal": 0.0,
                "shape_total": 0.0,
                "total": float(rew),
            }

        if not self.enable_reward_shaping:
            return base_reward
        if self.disable_shaping_on_terminal and done_all:
            return base_reward
        if info_norm is None:
            return base_reward

        ball_pos = self._extract_ball_position(info_norm)
        if ball_pos is None:
            return base_reward

        shaped = dict(base_reward)
        for agent_id in rew_dict:
            player_id = _policy_id_to_player_id(agent_id)
            prox_phi = self._proximity_potential(info_norm, player_id, ball_pos)
            goal_phi = self._goal_progress_potential(
                float(ball_pos[0]),
                self._agent_id_to_team_id(player_id),
            )

            prev_prox = self._prev_prox_phi.get(player_id)
            prev_goal = self._prev_goal_phi.get(player_id)
            prox_delta = 0.0 if prev_prox is None else prox_phi - prev_prox
            goal_delta = 0.0 if prev_goal is None else goal_phi - prev_goal

            prox_term = self.prox_weight * prox_delta
            goal_term = self.goal_weight * goal_delta
            shape_total = float(
                np.clip(prox_term + goal_term, -self.shaping_clip, self.shaping_clip)
            )
            shaped[agent_id] = base_reward[agent_id] + shape_total
            self._last_reward_metrics[player_id] = {
                "base": base_reward[agent_id],
                "prox": prox_term,
                "goal": goal_term,
                "shape_total": shape_total,
                "total": shaped[agent_id],
            }

            self._prev_prox_phi[player_id] = prox_phi
            self._prev_goal_phi[player_id] = goal_phi

        return shaped

    def _reset_potentials(self):
        self._prev_prox_phi = {}
        self._prev_goal_phi = {}
        self._last_reward_metrics = {}

    @staticmethod
    def _extract_position_from_info(info: dict, player_id: int):
        player_info = info.get(player_id, {}).get("player_info", {})
        position = player_info.get("position")
        if position is None:
            return None
        arr = np.asarray(position, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            return None
        return arr[:2]

    @staticmethod
    def _extract_ball_position(info: dict):
        for player_id in (0, 1, 2, 3):
            ball_info = info.get(player_id, {}).get("ball_info", {})
            position = ball_info.get("position")
            if position is None:
                continue
            arr = np.asarray(position, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                return arr[:2]
        return None

    def _proximity_potential(self, info: dict, player_id: int, ball_pos: np.ndarray) -> float:
        player_pos = self._extract_position_from_info(info, player_id)
        if player_pos is None:
            return 0.0
        dist = float(np.linalg.norm(player_pos - ball_pos))
        return float(1.0 - float(np.clip(dist / self.d_prox_max, 0.0, 1.0)))

    def _goal_progress_potential(self, ball_x: float, team_id: int) -> float:
        span = max(self.x_max - self.x_min, 1e-6)
        if int(team_id) == 0:
            val = (ball_x - self.x_min) / span
        else:
            val = (self.x_max - ball_x) / span
        return float(np.clip(val, 0.0, 1.0))


def create_rllib_env(env_config: dict = {}):
    # Copy so we do not mutate Ray's env_config dict in place
    cfg = {k: v for k, v in env_config.items()} if hasattr(env_config, "items") else dict(env_config)

    cfg["worker_id"] = _compute_worker_id(env_config, cfg)

    obs_mode = cfg.pop("obs_mode", "privileged")
    expected_obs_dim = int(cfg.pop("expected_obs_dim", 336))
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

    env = None
    last_error = None
    for attempt in range(max_make_retries + 1):
        try:
            env = soccer_twos.make(**cfg)
            break
        except UnityWorkerInUseException as exc:
            last_error = exc
            if attempt >= max_make_retries:
                raise
            cfg["worker_id"] = int(cfg["worker_id"]) + retry_worker_stride
            print(
                "[custom-env] worker_id collision; "
                f"retry {attempt + 1}/{max_make_retries} with worker_id={cfg['worker_id']}"
            )
    if env is None:
        raise last_error if last_error is not None else RuntimeError("Failed to create env")

    env = CustomEnvWrapper(
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

    if "multiagent" in cfg and not cfg["multiagent"]:
        return env
    return RLLibWrapper(env)
