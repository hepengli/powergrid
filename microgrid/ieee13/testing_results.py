import numpy as np
import pickle

from dataclasses import dataclass, field, asdict
from typing import List

from pathlib import Path
from test_utils import run_test, build_algo, DefaultEpisodeLog

from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.tune.registry import register_env

from powergrid.envs.single_agent.ieee13_mg import IEEE13Env
from powergrid.utils.utils import NormalizeActionWrapper

@dataclass
class IEEE13EpisodeLog(DefaultEpisodeLog):
    soc1: List[float] = field(default_factory=list)
    p_ess1: List[float] = field(default_factory=list)
    p_dg1: List[float] = field(default_factory=list)
    q_dg1: List[float] = field(default_factory=list)
    p_dg2: List[float] = field(default_factory=list)
    q_dg2: List[float] = field(default_factory=list)
    p_grid: List[float] = field(default_factory=list)
    q_grid: List[float] = field(default_factory=list)
    vm: List[np.ndarray] = field(default_factory=list)
    sampled_obs: List[np.ndarray] = field(default_factory=list)
    dreamed_obs: List[np.ndarray] = field(default_factory=list)

    def add_step(self, env) -> None:
        super().add_step(env)
        dev = env.devices
        self.soc1.append(dev["ESS1"].state.soc)
        self.p_ess1.append(dev["ESS1"].state.P)
        self.p_dg1.append(dev["DG1"].state.P)
        self.q_dg1.append(dev["DG1"].state.Q)
        self.p_dg2.append(dev["DG2"].state.P)
        self.q_dg2.append(dev["DG2"].state.Q)
        self.p_grid.append(dev["Grid"].state.P)
        self.q_grid.append(dev["Grid"].state.Q)
        self.vm.append(env.net.res_bus.vm_pu.values)


algo_name = "dreamerv3"  # "ppo", "sac", "dreamerv3"

match algo_name:
    case "dreamerv3":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "DreamerV3_2025-09-13_21-01-36"
            / "DreamerV3_pg-ieee13_5cd72_00000_0_2025-09-13_21-01-36"
            / "checkpoint_000047"
        )
        algo_class = DreamerV3Config
    case  "ppo":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "PPO_2025-09-13_20-55-53"
            / "PPO_IEEE13Env_908a8_00000_0_2025-09-13_20-55-53"
            / "checkpoint_000010"
        )
        algo_class = PPOConfig
    case  "sac":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "SAC_2025-09-13_21-25-09"
            / "SAC_pg-ieee13_a7564_00000_0_2025-09-13_21-25-09"
            / "checkpoint_000540"
        )
        algo_class = SACConfig


def make_env(env_config):
    return NormalizeActionWrapper(IEEE13Env(env_config=env_config))

register_env("pg-ieee13", make_env)
env = "pg-ieee13" if algo_name in ["sac", "dreamerv3"] else IEEE13Env

SEED = 2025091321

ENV_CONFIG = {
    "load_scale": 0.6,
    "reward_scale": 1.0,
    "safety_scale": 5000.0,
    "max_penalty": 5000.0,
    "train": False,
    "seed": SEED,
}

DREAM_TOD = 5
DREAM_H   = 15
DAYS      = 366
N_STEPS   = 24 * DAYS

algo, env = build_algo(CHECKPOINT, algo_class, env, ENV_CONFIG, SEED)
runlog, env = run_test(
    algo=algo, 
    env=env, 
    steps=N_STEPS, 
    episode_logger=IEEE13EpisodeLog,
    dream_tod=DREAM_TOD if algo_name == "dreamerv3" else -1,
    dream_h=DREAM_H if algo_name == "dreamerv3" else 0,
)

data = asdict(runlog)
with open("./ieee13_{}.pkl".format(algo_name), "wb") as f:
    pickle.dump(data, f)

algo.stop()