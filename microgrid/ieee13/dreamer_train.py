import os
import numpy as np

from pathlib import Path
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import register_env

from powergrid.envs.single_agent.ieee13_mg import IEEE13Env
from powergrid.utils.utils import NormalizeActionWrapper
from torch.utils.tensorboard import SummaryWriter

num_cpus = 1
num_gpus = 1

def make_env(env_config):
    return NormalizeActionWrapper(IEEE13Env(env_config=env_config))

env_config = {}
register_env("pg-ieee13", make_env)

parser = add_rllib_example_script_args(
    default_iters=10000000,
    default_reward=8000.0,
    default_timesteps=50000,
)

parser.set_defaults(
    checkpoint_freq=1000,
    verbose=0,
    no_tune=False,
    num_cpus=num_cpus,
    num_gpus=0,
)

# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values to set up `config` below.
args = parser.parse_args()
# If we use >1 GPU and increase the batch size accordingly, we should also
# increase the number of envs per worker.

from ray.rllib.callbacks.callbacks import RLlibCallback

class LogAcrobotAngle(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.writer = None
        self.global_step = 0
        self.last_episode_return = 0.0

    def _init_writer(self):
        if self.writer is None:
            pid = os.getpid()
            home_dir = Path.home()
            logdir = "{}/ray_results/ieee13_mg/dreamer/pid{}".format(home_dir, pid)
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logdir)

    def on_episode_created(self, *, episode, **kwargs):
        self._init_writer()
        # Initialize an empty list in the `custom_data` property of `episode`.
        episode.custom_data["safety"] = []
        episode.custom_data["reward"] = []
        episode.custom_data["return"] = []

    def on_episode_step(self, *, episode, env, **kwargs):
        env = env.envs[0].unwrapped
        # Compute the angle at every episode step and store it temporarily in episode:
        reward = np.sum(list(env.reward.values()))
        safety = np.sum(list(env.safety.values()))

        # one scalar per step (window=1 -> report current value)
        self.writer.add_scalar("train/reward", reward, self.global_step)
        self.writer.add_scalar("train/safety", safety, self.global_step)
        self.writer.add_scalar("train/return", self.last_episode_return, self.global_step)
        self.global_step += 1
        
        scaled_reward = reward * env.cfg["reward_scale"]
        penalty = np.clip(env.cfg["safety_scale"] * safety, 0.0, env.cfg["max_penalty"])

        episode.custom_data["reward"].append(reward)
        episode.custom_data["safety"].append(safety)
        episode.custom_data["return"].append(scaled_reward - penalty)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        ep_reward = np.sum(episode.custom_data["reward"])
        ep_safety = np.sum(episode.custom_data["safety"])
        ep_return = np.sum(episode.custom_data["return"])

        # Log the resulting average angle - per episode - to the MetricsLogger.
        metrics_logger.log_value("ep_safety", ep_reward, reduce="mean", window=1)
        metrics_logger.log_value("ep_reward", ep_safety, reduce="mean", window=1)
        metrics_logger.log_value("ep_return", ep_return, reduce="mean", window=1)

        self.last_episode_return = ep_return

    def on_algorithm_end(self, **kwargs):
        if self.writer:
            self.writer.flush()
            self.writer.close()


config = (
    DreamerV3Config()
    # Use image observations.
    .environment(
        "pg-ieee13",
        env_config={
            "load_scale": 0.6,
            "reward_scale": 1.0,
            "safety_scale": 10000.0,
            "max_penalty": 10000.0,
            "train": True,
        },
    )
    .env_runners(
        num_cpus_per_env_runner=num_cpus,
        num_gpus_per_env_runner=0,
    )
    .learners(
        num_cpus_per_learner=num_cpus,
        num_gpus_per_learner=num_gpus,
    )
    .training(
        model_size="XS", 
        training_ratio=1024,
    )
    .callbacks(
        callbacks_class=LogAcrobotAngle,
    )
)

# import ray
# import numpy as np
# from ray.rllib.utils.metrics import (
#     DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
#     ENV_RUNNER_RESULTS,
#     EPISODE_RETURN_MEAN,
#     EVALUATION_RESULTS,
#     NUM_ENV_STEPS_TRAINED,
#     NUM_ENV_STEPS_SAMPLED_LIFETIME,
# )
# algo = config.build_algo()
# for i in range(1000000):
#     results = algo.train()
#     if ENV_RUNNER_RESULTS in results:
#         mean_return = results[ENV_RUNNER_RESULTS].get(
#             EPISODE_RETURN_MEAN, np.nan
#         )
#         print(f"iter={i} R={mean_return}", end="")
#     if EVALUATION_RESULTS in results:
#         Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
#             EPISODE_RETURN_MEAN
#         ]
#         print(f" R(eval)={Reval}", end="")
#     print()

# ray.shutdown()

if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment
    run_rllib_example_script_experiment(config, args)

