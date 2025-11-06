#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import types
import numpy as np
import matplotlib.pyplot as plt

from utils_plot import compare_dream_vs_sample_heatmap  # expects (env, runlog)
from powergrid.envs.single_agent.ieee34_mg import IEEE34Env
from powergrid.utils.utils import NormalizeActionWrapper


def build_env():
    env_cfg = {
        "load_scale": 1.0,
        "reward_scale": 1.0,
        "safety_scale": 10000.0,
        "max_penalty": 10000.0,
        "train": False,
    }
    return NormalizeActionWrapper(IEEE34Env(env_config=env_cfg))


def get_episode(data, day_idx: int):
    episodes = data.get("episodes", None)
    if not isinstance(episodes, list) or len(episodes) == 0:
        raise ValueError("No episodes found in pickle (expected data['episodes'] to be a non-empty list).")
    if not (0 <= day_idx < len(episodes)):
        raise IndexError(f"day_idx {day_idx} out of range [0, {len(episodes)-1}]")
    return episodes[day_idx]  # this is a dict with keys from IEEE34EpisodeLog


def plot_schedules_for_day(ep, day_idx: int, savepath: str | None = None):
    """
    ep: one episode dict from IEEE34EpisodeLog (24 steps)
    """
    hours = np.arange(24)

    # Scalars per step (length 24 lists)
    soc1  = np.asarray(ep["soc1"], dtype=float)
    soc2  = np.asarray(ep["soc2"], dtype=float)
    p_ess1 = np.asarray(ep["p_ess1"], dtype=float)
    p_ess2 = np.asarray(ep["p_ess2"], dtype=float)
    p_dg1  = np.asarray(ep["p_dg1"], dtype=float)
    p_dg2  = np.asarray(ep["p_dg2"], dtype=float)
    p_grid = np.asarray(ep["p_grid"], dtype=float)

    # Voltages: list of arrays (nbus,) per hour -> envelope
    vm_list = ep["vm"]  # list of 24 arrays
    vm = np.stack(vm_list, axis=0)  # (24, nbus)
    vmax = vm.max(axis=1)
    vmin = vm.min(axis=1)

    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    axs[0].plot(hours, p_ess1, label="ESS1 P")
    axs[0].plot(hours, p_ess2, label="ESS2 P")
    axs[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axs[0].set_ylabel("Battery P")
    axs[0].legend(frameon=False); axs[0].grid(alpha=0.3)

    axs[1].plot(hours, soc1, label="ESS1 SoC")
    axs[1].plot(hours, soc2, label="ESS2 SoC")
    axs[1].set_ylabel("SoC")
    axs[1].legend(frameon=False); axs[1].grid(alpha=0.3)

    axs[2].plot(hours, p_dg1, label="DG1 P")
    axs[2].plot(hours, p_dg2, label="DG2 P")
    axs[2].set_ylabel("DG P")
    axs[2].legend(frameon=False); axs[2].grid(alpha=0.3)

    axs[3].plot(hours, p_grid, label="Grid P")
    axs[3].axhline(0, color="gray", linestyle="--", linewidth=1)
    axs[3].set_ylabel("Grid P")
    axs[3].legend(frameon=False); axs[3].grid(alpha=0.3)

    axs[4].plot(hours, vmax, label="Vmax (p.u.)")
    axs[4].plot(hours, vmin, label="Vmin (p.u.)")
    axs[4].axhline(1.05, color="red", linestyle="--", linewidth=1)
    axs[4].axhline(0.95, color="red", linestyle="--", linewidth=1)
    axs[4].set_ylabel("Voltage (p.u.)")
    axs[4].set_xlabel("Hour")
    axs[4].legend(frameon=False); axs[4].grid(alpha=0.3)

    plt.suptitle(f"DreamerV3 schedules — IEEE-34 — Day {day_idx}", y=0.98)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def call_heatmap_for_day(ep, env, savepath: str | None = None):
    """
    compare_dream_vs_sample_heatmap expects a runlog-like object:
      runlog.episodes[-1].dreamed_obs, runlog.episodes[-1].sampled_obs
    We'll wrap the single episode in a tiny namespace.
    """
    dreamed = np.array(ep.get("dreamed_obs", []))
    sampled = np.array(ep.get("sampled_obs", []))
    if dreamed.size == 0 or sampled.size == 0:
        print("Heatmap: dreamed_obs/sampled_obs not found in episode dict. "
              "You must call compare_dream_vs_sample_heatmap(env, runlog) during testing "
              "with the LIVE runlog to generate the figure.")
        return

    Ep = types.SimpleNamespace(dreamed_obs=dreamed, sampled_obs=sampled)
    Runlog = types.SimpleNamespace(episodes=[Ep], days=1)
    compare_dream_vs_sample_heatmap(env, Runlog, fname=savepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", default="./results/ieee34_dreamerv3.pkl", help="Path to asdict(runlog) pickle")
    parser.add_argument("--day", type=int, default=80, help="Day index in [0, 365]")
    parser.add_argument("--save_heatmap", default="./figures/ieee34_day_heatmap.png")
    parser.add_argument("--save_sched", default="./figures/ieee34_day_schedules.png")
    args = parser.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    ep = get_episode(data, args.day)
    env = build_env()

    # schedules first (always available from episode fields)
    plot_schedules_for_day(ep, args.day, savepath=args.save_sched)

    # heatmap next (only if dreamed/sample obs were serialized in the episode)
    call_heatmap_for_day(ep, env, savepath=args.save_heatmap)


if __name__ == "__main__":
    main()
