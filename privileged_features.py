import numpy as np

# Agent layout: team A = {0, 1}, team B = {2, 3}
TEAMMATE = {0: 1, 1: 0, 2: 3, 3: 2}
TEAM = {0: 1, 1: 1, 2: -1, 3: -1}
OPPONENT_IDS = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

PRIVILEGED_OBS_DIM = 25
TEAM_PRIVILEGED_OBS_DIM = 25  # 4 players × (pos_2 + vel_2 + rot_1) + ball (pos_2 + vel_2) + team_tag (1)


def flat_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def build_privileged_vector(info_norm, agent_id: int) -> np.ndarray:
    """Build privileged feature vector for a specific agent id."""
    pi = lambda aid: info_norm[aid]["player_info"]
    mate_id = TEAMMATE[agent_id]

    self_pos = pi(agent_id)["position"]
    self_vel = pi(agent_id)["velocity"]
    self_rot = pi(agent_id)["rotation_y"]
    team_pos = pi(mate_id)["position"]
    team_vel = pi(mate_id)["velocity"]
    team_rot = pi(mate_id)["rotation_y"]
    opponent_pos = [pi(i)["position"] for i in OPPONENT_IDS[agent_id]]
    opponent_vel = [pi(i)["velocity"] for i in OPPONENT_IDS[agent_id]]
    opponent_rot = [pi(i)["rotation_y"] for i in OPPONENT_IDS[agent_id]]
    ball_pos = info_norm[agent_id]["ball_info"]["position"]
    ball_vel = info_norm[agent_id]["ball_info"]["velocity"]
    team_id = TEAM[agent_id]

    pieces = [
        flat_f32(self_pos),
        flat_f32(self_vel),
        flat_f32(self_rot),
        flat_f32(team_pos),
        flat_f32(team_vel),
        flat_f32(team_rot),
        flat_f32(team_id),
    ]
    for p in opponent_pos:
        pieces.append(flat_f32(p))
    for v in opponent_vel:
        pieces.append(flat_f32(v))
    for r in opponent_rot:
        pieces.append(flat_f32(r))
    pieces.extend([flat_f32(ball_pos), flat_f32(ball_vel)])
    out = np.concatenate(pieces).astype(np.float32)
    if out.shape[0] != PRIVILEGED_OBS_DIM:
        raise ValueError(
            f"Expected privileged dim {PRIVILEGED_OBS_DIM}, got {out.shape[0]}"
        )
    return out


def _team_player_order(team_id: int):
    if int(team_id) == 0:
        return (0, 1, 2, 3), 1.0
    if int(team_id) == 1:
        return (2, 3, 0, 1), -1.0
    raise ValueError(f"Unsupported team_id={team_id}; expected 0 or 1")


def build_team_privileged_vector(info_norm, team_id: int) -> np.ndarray:
    """
    Team-perspective privileged vector.

    Feature order is strict and deterministic:
      team 0 -> agent0, agent1, agent2, agent3, ball, team_tag(+1)
      team 1 -> agent2, agent3, agent0, agent1, ball, team_tag(-1)

    For each player, append [position(2), velocity(2), rotation_y(1)].
    Then append ball [position(2), velocity(2)] and team tag (1).
    """
    pi = lambda aid: info_norm[aid]["player_info"]
    ordered_players, team_tag = _team_player_order(team_id)

    pieces = []
    for agent_id in ordered_players:
        pieces.append(flat_f32(pi(agent_id)["position"]))
        pieces.append(flat_f32(pi(agent_id)["velocity"]))
        pieces.append(flat_f32(pi(agent_id)["rotation_y"]))

    # Ball state is shared, so read from the first team player entry.
    ball_anchor = ordered_players[0]
    ball_pos = info_norm[ball_anchor]["ball_info"]["position"]
    ball_vel = info_norm[ball_anchor]["ball_info"]["velocity"]
    pieces.extend([flat_f32(ball_pos), flat_f32(ball_vel), flat_f32(team_tag)])

    out = np.concatenate(pieces).astype(np.float32)
    if out.shape[0] != TEAM_PRIVILEGED_OBS_DIM:
        raise ValueError(
            f"Expected team privileged dim {TEAM_PRIVILEGED_OBS_DIM}, got {out.shape[0]}"
        )
    return out
