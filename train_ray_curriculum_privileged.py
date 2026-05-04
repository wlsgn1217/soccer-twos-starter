import os
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from custom_env_wrapper import create_rllib_env
from models.rllib_registration import register_privileged_actor_model
from privileged_curriculum_sampling import (
    CURRICULUM_ADVANCE_REWARD_MEAN,
    STOP_EPISODE_REWARD_MEAN,
    tasks,
)
from curriculum_sampling import sample_players_states, sample_pos_vel
from tools.export_stage1_privileged_weights import export_split_weights


NUM_ENVS_PER_WORKER = 3

current = 0
config_fns = {
    "none": lambda *_: None,
    "random_bots": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}


class CurriculumUpdateCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks

        for env in base_env.get_unwrapped():
            config_fns[tasks[current]["config_fn"]](env)
            ball_state = sample_pos_vel(tasks[current]["ranges"]["ball"])
            ball_pos = ball_state["position"]
            env.env_channel.set_parameters(
                ball_state=ball_state,
                players_states=sample_players_states(tasks[current], ball_pos),
            )

    def on_train_result(self, **info):
        global current
        default_policy = info["trainer"].get_policy()
        if default_policy is not None and hasattr(default_policy.model, "last_latent_mean"):
            info["result"].setdefault("custom_metrics", {})
            info["result"]["custom_metrics"]["latent_mean"] = default_policy.model.last_latent_mean
            info["result"]["custom_metrics"]["latent_std"] = default_policy.model.last_latent_std
        if info["result"]["episode_reward_mean"] > CURRICULUM_ADVANCE_REWARD_MEAN:
            if current < len(tasks) - 1:
                print("---- Updating tasks!!! ----")
                current += 1
                print(f"Current task: {current} - {tasks[current]['name']}")


if __name__ == "__main__":
    ray.init()
    register_privileged_actor_model()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumUpdateCallback,
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
                "opponent_policy": lambda *_: 0,
                "obs_mode": "privileged_dict",
                "expected_obs_dim": 336,
            },
            "model": {
                "custom_model": "privileged_actor_model",
                "custom_model_config": {
                    "obs_dim": 336,
                    "privileged_dim": 25,
                    "latent_dim": 16,
                    "encoder_hiddens": [64, 64],
                    "actor_hiddens": [256, 256],
                    "value_hiddens": [256, 256],
                    "activation": "relu",
                },
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 150000000,
            "time_total_s": 72000,  # 2h
            "episode_reward_mean": STOP_EPISODE_REWARD_MEAN,
        },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    export_dir = os.path.join(os.path.dirname(best_checkpoint), "split_stage1")
    export_split_weights(best_checkpoint, export_dir, policy_name="default")
    print(f"Exported split stage-1 weights: {export_dir}")
    print("Done training")
