import os
import numpy as np

from pathlib import Path
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args

from powergrid.envs.single_agent.ieee34_mg import IEEE34Env
from torch.utils.tensorboard import SummaryWriter

env_config = {}

parser = add_rllib_example_script_args(
    default_iters=10000000,
    default_reward=8000.0,
    default_timesteps=50000,
)

parser.set_defaults(
    checkpoint_freq=10,
    verbose=0,
    no_tune=False,
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
            logdir = "{}/ray_results/ieee34_mg/ppo/pid{}".format(pid)
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logdir)

    def on_episode_start(self, *, episode, **kwargs):
        self._init_writer()
        # Start of new episode, reset current episode return
        episode.custom_data["current_return"] = 0.0

    def on_episode_step(self, *, episode, env, **kwargs):
        env = env.envs[0].unwrapped
        # Compute the angle at every episode step and store it temporarily in episode:
        reward = np.sum(list(env.reward.values()))
        safety = np.sum(list(env.safety.values()))

        scaled_reward = reward * env.cfg["reward_scale"]
        penalty = np.clip(env.cfg["safety_scale"] * safety, 0.0, env.cfg["max_penalty"])
        episode.custom_data["current_return"] += scaled_reward - penalty

        # one scalar per step (window=1 -> report current value)
        self.writer.add_scalar("train/reward", reward, self.global_step)
        self.writer.add_scalar("train/safety", safety, self.global_step)
        self.writer.add_scalar("train/return", self.last_episode_return, self.global_step)
        self.global_step += 1

    def on_episode_end(self, *, episode, **kwargs):
        # Store this episode's final return for use in the next one
        self.last_episode_return = episode.custom_data["current_return"]

    def on_algorithm_end(self, **kwargs):
        if self.writer:
            self.writer.flush()
            self.writer.close()

config = (
    PPOConfig()
    # Use image observations.
    .environment(
        IEEE34Env,
        env_config={
            "load_scale": 1.0,
            "reward_scale": 1.0,
            "safety_scale": 10000.0,
            "max_penalty": 10000.0,
            "train": True,
        },
    )
    .env_runners(num_env_runners=0)
    .callbacks(
        callbacks_class=LogAcrobotAngle,
    )
)

if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    run_rllib_example_script_experiment(config, args)
