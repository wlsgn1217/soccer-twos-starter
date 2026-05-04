"""Curriculum task table and player sampling for privileged training (stdlib only; no Ray)."""
from pathlib import Path

import yaml

from curriculum_sampling import (
    BOT_MIN_DIST_FROM_BALL,
    dist_sq_xy,
    sample_player,
    sample_pos_vel,
    sample_players_states,
    sample_vec,
)

CURRICULUM_PATH = Path(__file__).resolve().parent / "curriculum_for_privileged.yaml"

# Early privileged stages use stationary teammates/opponents; last task uses random_players.
CURRICULUM_ADVANCE_REWARD_MEAN = 1.5
STOP_EPISODE_REWARD_MEAN = 1.9

with open(CURRICULUM_PATH) as f:
    tasks = yaml.load(f, Loader=yaml.FullLoader)["tasks"]
