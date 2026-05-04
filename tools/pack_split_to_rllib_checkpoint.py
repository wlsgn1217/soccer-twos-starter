"""
Pack split_stage1 weights (actor + privileged_encoder, optional value) into an
RLlib PPO checkpoint directory so previleged_agent.RayAgent can load it via
CHECKPOINT_PATH (same flow as curriculum training checkpoints).

Ray RLlib checkpoints are a binary bundle (weights + optimizer + metadata). We do
not reimplement that format from scratch. Instead we **clone** a curriculum PPO
checkpoint shell (read-only), overwrite actor + privileged_encoder with your
split weights, and **write a new checkpoint** under split_dir. The template file
on disk is never modified.

If --template-checkpoint is omitted, the same default path as previleged_agent
(agent_ray.DEFAULT_CHECKPOINT_PATH layout) is tried under the repo root, then
under previleged_agent/. Override with env PRIVILEGED_RLLIB_TEMPLATE_CHECKPOINT.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from typing import Optional

# Allow `python tools/pack_split_to_rllib_checkpoint.py` from any CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ray
import torch
from ray import tune
from ray.tune.registry import get_trainable_cls

from custom_env_wrapper import create_rllib_env
from models.rllib_registration import register_privileged_actor_model

# Same relative layout as previleged_agent/agent_ray.py DEFAULT_CHECKPOINT_PATH
# (repo_root/ray_results/.../checkpoint-1574).
_DEFAULT_TEMPLATE_REL = os.path.join(
    "ray_results",
    "privileged_from_curriculum_20260407_234102",
    "PPO_selfplay_rec",
    "PPO_Soccer_c474a_00000_0_2026-04-07_23-41-03",
    "checkpoint_001574",
    "checkpoint-1574",
)


def resolve_template_checkpoint(explicit: Optional[str]) -> str:
    """Curriculum RLlib checkpoint file used as read-only shell for packing."""
    if explicit:
        return os.path.abspath(explicit)
    env_path = os.environ.get(
        "PRIVILEGED_RLLIB_TEMPLATE_CHECKPOINT",
        os.environ.get("PRIVILEGED_RLlib_TEMPLATE_CHECKPOINT", ""),
    ).strip()
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)
    candidates = [
        os.path.join(_REPO_ROOT, _DEFAULT_TEMPLATE_REL),
        os.path.join(_REPO_ROOT, "previleged_agent", _DEFAULT_TEMPLATE_REL),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "No template RLlib checkpoint found. Train or copy a privileged PPO checkpoint, "
        "or set PRIVILEGED_RLLIB_TEMPLATE_CHECKPOINT to a checkpoint *file* path, "
        "or pass --template-checkpoint explicitly. Tried:\n  "
        + "\n  ".join(candidates)
    )


def _find_params_pkl(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cur = checkpoint_dir
    for _ in range(8):
        p = os.path.join(cur, "params.pkl")
        if os.path.isfile(p):
            return p
        cur = os.path.dirname(cur)
    raise FileNotFoundError(
        f"Could not find params.pkl near template checkpoint: {checkpoint_path}"
    )


def pack_split_to_rllib_checkpoint(
    split_dir: str,
    template_checkpoint: Optional[str] = None,
    policy_name: str = "default",
    pack_subdir: str = "rllib_checkpoint",
) -> str:
    split_dir = os.path.abspath(split_dir)
    template_checkpoint = resolve_template_checkpoint(template_checkpoint)
    actor_path = os.path.join(split_dir, "actor_weights.pt")
    encoder_path = os.path.join(split_dir, "privileged_encoder_weights.pt")
    value_path = os.path.join(split_dir, "value_weights.pt")

    if not os.path.isfile(actor_path) or not os.path.isfile(encoder_path):
        raise FileNotFoundError(
            f"split_dir must contain actor_weights.pt and privileged_encoder_weights.pt: {split_dir}"
        )
    if os.path.isdir(template_checkpoint):
        raise IsADirectoryError(
            "template_checkpoint must be an RLlib checkpoint *file* (e.g. .../checkpoint-1574), "
            f"not a directory. Got: {template_checkpoint}"
        )
    if not os.path.isfile(template_checkpoint):
        raise FileNotFoundError(f"template_checkpoint not found: {template_checkpoint}")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_privileged_actor_model()
    # params.pkl uses env "Soccer" registered during training; register here for standalone pack.
    tune.registry.register_env("Soccer", create_rllib_env)

    params_path = _find_params_pkl(template_checkpoint)
    with open(params_path, "rb") as f:
        config = pickle.load(f)
    config["num_workers"] = 0
    config["num_gpus"] = 0

    cls = get_trainable_cls("PPO")
    trainer = cls(env=config["env"], config=config)
    trainer.restore(template_checkpoint)

    policy = trainer.get_policy(policy_name)
    if policy is None:
        policy = trainer.get_policy("default_policy")
    if policy is None:
        raise RuntimeError("Could not resolve policy for weight injection.")

    model = policy.model
    if not hasattr(model, "actor_backbone") or not hasattr(model, "privileged_encoder"):
        raise RuntimeError("Policy model is not PrivilegedActorModel (missing split modules).")

    actor_state = torch.load(actor_path, map_location="cpu")
    encoder_state = torch.load(encoder_path, map_location="cpu")
    model.actor_backbone.load_state_dict(actor_state, strict=True)
    model.privileged_encoder.load_state_dict(encoder_state, strict=True)
    if os.path.isfile(value_path) and hasattr(model, "value_backbone"):
        model.value_backbone.load_state_dict(torch.load(value_path, map_location="cpu"), strict=True)

    save_root = os.path.join(split_dir, pack_subdir)
    os.makedirs(save_root, exist_ok=True)
    checkpoint_path = trainer.save(save_root)
    print(f"[pack] Using read-only template: {template_checkpoint}")

    ck_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    dest_params = os.path.join(ck_dir, "params.pkl")
    if not os.path.isfile(dest_params):
        shutil.copy2(params_path, dest_params)

    meta = {
        "split_dir": split_dir,
        "template_checkpoint": template_checkpoint,
        "template_params_pkl": params_path,
        "packed_checkpoint": checkpoint_path,
        "policy_name": policy_name,
    }
    with open(os.path.join(split_dir, "rllib_pack_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    trainer.stop()
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing actor_weights.pt and privileged_encoder_weights.pt (e.g. .../split_stage1)",
    )
    parser.add_argument(
        "--template-checkpoint",
        default=None,
        help="Optional RLlib checkpoint *file* for params/value shell (default: same as "
        "previleged_agent curriculum path under repo, or PRIVILEGED_RLLIB_TEMPLATE_CHECKPOINT). "
        "Template is never overwritten.",
    )
    parser.add_argument("--policy-name", default="default")
    parser.add_argument("--pack-subdir", default="rllib_checkpoint")
    args = parser.parse_args()

    out = pack_split_to_rllib_checkpoint(
        split_dir=args.split_dir,
        template_checkpoint=args.template_checkpoint,
        policy_name=args.policy_name,
        pack_subdir=args.pack_subdir,
    )
    print(f"Packed RLlib checkpoint: {out}")
    print("Use with previleged_agent / watch.py, e.g.:")
    print(f"  export CHECKPOINT_PATH={out}")


if __name__ == "__main__":
    main()
