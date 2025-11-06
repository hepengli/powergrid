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

from powergrid.envs.single_agent.cigre_mv import CIGREMVEnv
from powergrid.utils.utils import NormalizeActionWrapper


@dataclass
class CIGREMVEpisodeLog(DefaultEpisodeLog):
    soc1: List[float] = field(default_factory=list)
    p_ess1: List[float] = field(default_factory=list)
    soc2: List[float] = field(default_factory=list)
    p_ess2: List[float] = field(default_factory=list)
    p_rfc_1: List[float] = field(default_factory=list)
    q_rfc_1: List[float] = field(default_factory=list)
    p_rfc_2: List[float] = field(default_factory=list)
    q_rfc_2: List[float] = field(default_factory=list)
    p_fc: List[float] = field(default_factory=list)
    q_fc: List[float] = field(default_factory=list)
    p_chp: List[float] = field(default_factory=list)
    q_chp: List[float] = field(default_factory=list)
    p_grid: List[float] = field(default_factory=list)
    q_grid: List[float] = field(default_factory=list)
    vm: List[np.ndarray] = field(default_factory=list)
    sampled_obs: List[np.ndarray] = field(default_factory=list)
    dreamed_obs: List[np.ndarray] = field(default_factory=list)

    def add_step(self, env) -> None:
        super().add_step(env)
        dev = env.devices
        self.soc1.append(dev["Battery 1"].state.soc)
        self.p_ess1.append(dev["Battery 1"].state.P)
        self.soc2.append(dev["Battery 2"].state.soc)
        self.p_ess2.append(dev["Battery 2"].state.P)
        self.p_rfc_1.append(dev["Residential fuel cell 1"].state.P)
        self.q_rfc_1.append(dev["Residential fuel cell 1"].state.Q)
        self.p_rfc_2.append(dev["Residential fuel cell 2"].state.P)
        self.q_rfc_2.append(dev["Residential fuel cell 2"].state.Q)
        self.p_fc.append(dev["Fuel cell"].state.P)
        self.q_fc.append(dev["Fuel cell"].state.Q)
        self.p_chp.append(dev["CHP diesel"].state.P)
        self.q_chp.append(dev["CHP diesel"].state.Q)
        self.p_grid.append(dev["GRID"].state.P)
        self.q_grid.append(dev["GRID"].state.Q)
        self.vm.append(env.net.res_bus.vm_pu.values)

algo_name = "sac"  # dreamerv3, ppo, sac

match algo_name:
    case "dreamerv3":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "DreamerV3_2025-10-19_13-35-49"
            / "DreamerV3_pg-cigre_mv_0d964_00000_0_2025-10-19_13-35-49"
            / "checkpoint_000047"
        )
        algo_class = DreamerV3Config
    case  "ppo":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "PPO_2025-10-23_11-45-45"
            / "PPO_CIGREMVEnv_56d83_00000_0_2025-10-23_11-45-45"
            / "checkpoint_000012"
        )
        algo_class = PPOConfig
    case  "sac":
        CHECKPOINT = (
            Path("/Users/hepeng.li/ray_results")
            / "SAC_2025-10-22_14-47-10"
            / "SAC_pg-cigre_mv_8492a_00000_0_2025-10-22_14-47-10"
            / "checkpoint_000049"
        )
        algo_class = SACConfig


def make_env(env_config):
    return NormalizeActionWrapper(CIGREMVEnv(env_config=env_config))

register_env("pg-ieee13", make_env)
env = "pg-ieee13" if algo_name in ["sac", "dreamerv3"] else CIGREMVEnv

SEED = 2025102214

ENV_CONFIG = {
    "load_scale": 1.0,
    "reward_scale": 1.0,
    "safety_scale": 10000.0,
    "max_penalty": 10000.0,
    "train": False,
    "seed": SEED
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
    episode_logger=CIGREMVEpisodeLog,
    dream_tod=DREAM_TOD if algo_name == "dreamerv3" else -1,
    dream_h=DREAM_H if algo_name == "dreamerv3" else 0,
)

data = asdict(runlog)
with open("./cigre_mv_{}.pkl".format(algo_name), "wb") as f:
    pickle.dump(data, f)

algo.stop()