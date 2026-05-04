"""YAML range sampling for curriculum tasks (stdlib only; no gym/Ray)."""
from random import choice
from random import uniform as randfloat


def _sample_axis(spec):
    """Sample one coordinate from a single [lo, hi] or a union [[lo, hi], ...]."""
    if spec is None or (isinstance(spec, list) and len(spec) == 0):
        raise ValueError("axis spec must be non-empty")
    if (
        len(spec) == 2
        and isinstance(spec[0], (int, float))
        and isinstance(spec[1], (int, float))
    ):
        lo, hi = float(spec[0]), float(spec[1])
        if lo > hi:
            lo, hi = hi, lo
        return randfloat(lo, hi)
    if all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (int, float))
        and isinstance(item[1], (int, float))
        for item in spec
    ):
        a, b = choice(spec)
        lo, hi = float(a), float(b)
        if lo > hi:
            lo, hi = hi, lo
        return randfloat(lo, hi)
    raise ValueError(f"invalid axis spec: {spec!r}")


BOT_MIN_DIST_FROM_BALL = 2.5
_BOT_SPAWN_MAX_TRIES = 32


def sample_vec(range_dict):
    return [
        _sample_axis(range_dict["x"]),
        _sample_axis(range_dict["y"]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s


def dist_sq_xy(a, b) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def sample_players_states(task: dict, ball_position) -> dict:
    """Sample all player states; enforce min distance from ball for agents 1–3."""
    spec = task["ranges"]["players"]
    out = {}
    for player_key, player_spec in spec.items():
        pid = int(player_key) if not isinstance(player_key, int) else player_key
        if pid == 0:
            out[pid] = sample_player(player_spec)
            continue
        for _ in range(_BOT_SPAWN_MAX_TRIES):
            st = sample_player(player_spec)
            pos = st["position"]
            if dist_sq_xy(pos, ball_position) >= BOT_MIN_DIST_FROM_BALL**2:
                out[pid] = st
                break
        else:
            out[pid] = sample_player(player_spec)
    return out
