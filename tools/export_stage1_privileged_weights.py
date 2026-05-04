import json
import os
import pickle

import ray
import torch
from ray import tune
from ray.tune.registry import get_trainable_cls

from custom_env_wrapper import create_rllib_env
from models.rllib_registration import register_privileged_actor_model


def _load_config_from_checkpoint(checkpoint_path: str):
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError("Could not find params.pkl near checkpoint")
    with open(config_path, "rb") as f:
        return pickle.load(f)


def export_split_weights(
    checkpoint_path: str,
    output_dir: str,
    policy_name: str = "default",
):
    os.makedirs(output_dir, exist_ok=True)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_privileged_actor_model()
    config = _load_config_from_checkpoint(checkpoint_path)
    config["num_workers"] = 0
    config["num_gpus"] = 0

    # Register the env under "Soccer" and also under whatever name the checkpoint used,
    # so that checkpoints trained with a different env string (e.g. "SoccerAgainstBaseline")
    # can be restored without a gym lookup error.
    tune.registry.register_env("Soccer", create_rllib_env)
    src_env = config.get("env", "Soccer")
    if src_env != "Soccer":
        tune.registry.register_env(src_env, create_rllib_env)

    cls = get_trainable_cls("PPO")
    trainer = cls(env=config["env"], config=config)
    trainer.restore(checkpoint_path)

    policy = trainer.get_policy(policy_name)
    if policy is None:
        policy = trainer.get_policy("default_policy")
    if policy is None:
        raise ValueError("Could not resolve a policy for export")

    model = policy.model
    if not hasattr(model, "export_split_state_dict"):
        raise ValueError("Policy model does not implement split export")

    split = model.export_split_state_dict()
    torch.save(split["privileged_encoder"], os.path.join(output_dir, "privileged_encoder_weights.pt"))
    torch.save(split["actor_backbone"], os.path.join(output_dir, "actor_weights.pt"))
    torch.save(split["value_backbone"], os.path.join(output_dir, "value_weights.pt"))

    obs_space = policy.observation_space
    action_space = policy.action_space
    meta = {
        "version": 1,
        "model": "privileged_actor_model",
        "latent_dim": int(split["latent_dim"]),
        "observation_space": str(obs_space),
        "action_space": str(action_space),
        "policy_name": policy_name,
        "checkpoint_path": checkpoint_path,
    }
    with open(os.path.join(output_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export stage-1 split privileged weights")
    parser.add_argument("--checkpoint", required=True, help="RLlib checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Export directory")
    parser.add_argument("--policy-name", default="default", help="Policy name")
    args = parser.parse_args()

    out = export_split_weights(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
    )
    print(f"Exported stage-1 split weights to: {out}")
