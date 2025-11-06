import os
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from powergrid.envs.single_agent.ieee34_mg import IEEE34Env

LOG_DIR = "./logs/ppo"
SAVE_DIR = "./models/ppo"
TB_LOG = os.path.join(LOG_DIR, "tb")
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000           # steps between evals
N_EVAL_EPISODES = 5
SEED = 42
N_ENVS = 1                   # increase for vectorized training if env supports it
DETERMINISTIC_RUN = True     # for trajectory extraction
ROLLOUT_EPISODES = 2         # how many episodes to save actions for
TRAJ_CSV = "./trajectories/ppo_control_trajectory.csv"

MG_ID = 1
HORIZON = 10

# (Optional) custom callback to log extra info from env during training
class InfoLoggerCallback(BaseCallback):
    """
    If your env puts domain metrics into `info` (e.g., {"line_overload": ..., "shed": ...}),
    this logs their episode means to TensorBoard. Compatible with Monitor wrapper.
    """
    def __init__(self, keys_to_log: Optional[List[str]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.keys = keys_to_log or []
        self.buffer: Dict[str, List[float]] = {k: [] for k in self.keys}

    def _on_step(self) -> bool:
        # VecEnv: infos is a list; for DummyVecEnv length is 1
        infos = self.locals.get("infos", [])
        for info in infos:
            for k in self.keys:
                if k in info and isinstance(info[k], (int, float, np.number)):
                    self.buffer[k].append(float(info[k]))
        # When episode ends, flush episode-averages
        dones = self.locals.get("dones", [])
        if np.any(dones):
            for k, vals in self.buffer.items():
                if vals:
                    self.logger.record(f"train_info/{k}_mean", float(np.mean(vals)))
            self.buffer = {k: [] for k in self.keys}
        return True


def make_env(seed: int = None):
    """
    Wrap env with Monitor to record episode rewards/lengths.
    If you don't have a registered id, replace with your constructor, e.g.:
        from my_pkg.envs import PowerGridEnv
        env = PowerGridEnv(config=...)
    """
    def _init():
        env_config = {
            "load_scale": 1.0,
            "reward_scale": 1.0,
            "safety_scale": 10000.0,
            "max_penalty": 10000.0,
            "train": True,
        }
        env = IEEE34Env(env_config=env_config)
        env = Monitor(env)
        if hasattr(env, "reset"):
            try:
                env.reset(seed=seed)
            except TypeError:
                # Older gym API
                env.seed(seed)
        return env
    return _init


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TRAJ_CSV), exist_ok=True)

    # Logger (TensorBoard)
    new_logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    # Train env(s)
    vec_env = DummyVecEnv([make_env(SEED + i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)  # keeps ep_rew_mean, ep_len_mean, etc.

    # Eval env (single)
    eval_env = DummyVecEnv([make_env(SEED + 10_000)])
    eval_env = VecMonitor(eval_env)

    # Model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        tensorboard_log=TB_LOG,
        seed=SEED,
        verbose=1,
        device="auto",
    )
    model.set_logger(new_logger)

    # Callbacks: save best model on eval return, regular checkpoints, optional info logger
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ // max(N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(save_freq=EVAL_FREQ, save_path=SAVE_DIR, name_prefix="ppo_ckpt")
    info_logger = InfoLoggerCallback(keys_to_log=[
        # e.g., put your env info keys here:
        # "line_overload", 
        # "voltage_violation", 
    ])

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, ckpt_callback, info_logger],
        progress_bar=True,
    )

    # Load the "best" model found by EvalCallback (falls back to current if none)
    best_path = os.path.join(SAVE_DIR, "best_model.zip")
    if os.path.exists(best_path):
        model = PPO.load(best_path, env=vec_env, device="auto")

    # We roll out deterministically and save obs/action/reward/done + selected info keys
    import csv

    # single-env instance for rollout
    env_config = {
        "load_scale": 1.0,
        "reward_scale": 1.0,
        "safety_scale": 10000.0,
        "max_penalty": 10000.0,
        "train": True,
    },
    rollout_env = IEEE34Env(env_config=env_config)
    obs, _ = rollout_env.reset(seed=SEED + 777)

    # Decide which info keys to record (customize for your env)
    info_keys_to_keep = [
        # e.g., 
        # "time_step", 
        # "bus_voltage", 
        # "line_loading", 
        # "generator_setpoint"
    ]

    header = ["episode", "t", "reward", "done"]
    # Add obs and action fields (flattened)
    # If obs/action are dict/spaces, adapt below accordingly
    # For simplicity, assume Box spaces
    obs_size = np.array(obs).size
    header += [f"obs_{i}" for i in range(obs_size)]

    # we need action space size too
    dummy_action = model.policy.action_net.weight.detach().cpu().numpy() if hasattr(model.policy, "action_net") else None
    # safer: sample from env’s action space
    a_sample = rollout_env.action_space.sample()
    act_size = np.array(a_sample).size
    header += [f"act_{i}" for i in range(act_size)]
    header += [f"info_{k}" for k in info_keys_to_keep]

    with open(TRAJ_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        episode = 0
        for ep in range(ROLLOUT_EPISODES):
            t = 0
            # reset per episode
            obs, _ = rollout_env.reset(seed=SEED + 777 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=DETERMINISTIC_RUN)
                step_out = rollout_env.step(action)
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)

                o = np.array(obs).ravel()
                a = np.array(action).ravel()
                row = [episode, t, float(reward), int(done)]
                row += o.tolist()
                row += a.tolist()
                row += [info.get(k, np.nan) for k in info_keys_to_keep]
                writer.writerow(row)

                obs = next_obs
                t += 1
            episode += 1

    print(f"[OK] Training complete. Best model (if any) in: {SAVE_DIR}")
    print(f"[OK] Control trajectory saved to: {TRAJ_CSV}")
    print(f"[TIP] Launch TensorBoard with: tensorboard --logdir {TB_LOG}")


if __name__ == "__main__":
    main()
