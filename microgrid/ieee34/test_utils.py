import numpy as np
import torch

from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from ray.rllib.algorithms.dreamerv3.utils.summaries import reconstruct_obs_from_h_and_z
from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.utils.torch_utils import inverse_symlog

def unwrap_env(algo):
    """Return the first underlying Gym env (unwrap vector + wrappers)."""
    e = getattr(algo, "env_runner", None)
    if e is None:
        raise RuntimeError("algo.env_runner not found; use rollout or train() instead.")
    env = e.env
    # peel wrappers
    while hasattr(env, "env"):
        env = env.env
    # handle vector envs
    if hasattr(env, "envs"):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env

def in_cyclic_window(x, start, length, period):
    """True if x lies in [start, start+length] modulo period (inclusive)."""
    if length < 0:
        return False
    end = (start + length) % period
    if start <= end:
        return start <= x <= end
    else:
        return x >= start or x <= end

def dream_trajectory(alg, T):
    states = alg.env_runner._cached_to_module['state_in']
    start_is_terminated = torch.Tensor([0.0])
    dreamer_model = alg.env_runner.module.dreamer_model
    dream_data = dreamer_model.dream_trajectory(states, start_is_terminated, T, alg.config.gamma)
    dreamed_obs_H_B = reconstruct_obs_from_h_and_z(
        h_t0_to_H=dream_data["h_states_t0_to_H_BxT"].numpy(),  # [0] b/c reduce=None (list)
        z_t0_to_H=dream_data["z_states_prior_t0_to_H_BxT"].numpy(),
        dreamer_model=dreamer_model,
        obs_dims_shape=alg.env_runner._cached_to_module["obs"].shape[2:],
        framework="torch",
    )
    computed_float_obs_B_T_dims = np.swapaxes(dreamed_obs_H_B, 0, 1)[0:1]
    symlog_obs = do_symlog_obs(alg.env_runner.env.single_observation_space, alg.config.symlog_obs)
    if symlog_obs:
        computed_float_obs_B_T_dims = inverse_symlog(torch.Tensor(computed_float_obs_B_T_dims)).numpy()

    return computed_float_obs_B_T_dims[0]

@dataclass
class DefaultEpisodeLog:
    ret: float = 0.0
    safety: float = 0.0

    def add_step(self, env) -> None:
        self.ret += float(np.sum(list(getattr(env, "reward", {}).values()))) if hasattr(env, "reward") else 0.0
        self.safety += float(np.sum(list(getattr(env, "safety", {}).values()))) if hasattr(env, "safety") else 0.0

@dataclass
class RunLog:
    episodes: List[DefaultEpisodeLog] = field(default_factory=list)

    @property
    def days(self) -> int:
        return len(self.episodes)

    @property
    def total_return(self) -> float:
        return float(sum(ep.ret for ep in self.episodes))

    @property
    def total_safety(self) -> float:
        return float(sum(ep.safety for ep in self.episodes))


def build_algo(checkpoint, algo_config, env, env_config, seed):
    """Build DreamerV3, restore checkpoint, unwrap env."""
    config = (
        algo_config()
        .environment(env, env_config=env_config)
        .env_runners(num_env_runners=0)
        .evaluation(evaluation_duration_unit="timesteps", evaluation_duration=1)
        .debugging(seed=seed)
    )
    algo = config.build_algo()
    algo.restore_from_path(str(checkpoint))
    env = unwrap_env(algo)
    return algo, env

def run_test(
    algo,
    env,
    steps: int,
    episode_logger: Callable[[], DefaultEpisodeLog] = DefaultEpisodeLog,
    dream_tod: int = -1,
    dream_h: int = 0,
) -> RunLog:
    """Advance env one step per algo.evaluate(); collect logs & dream/sample obs."""
    runlog = RunLog()
    current_ep: DefaultEpisodeLog = episode_logger()

    for t in range(steps - 1):
        tod = env.t % env.episode_length  # 0..23

        if tod == 0:
            current_ep = episode_logger()
            runlog.episodes.append(current_ep)
            print(f"Episode {runlog.days-1}")

        # advance exactly ONE step
        _ = algo.evaluate()
        env = unwrap_env(algo)  # refresh env reference

        # dream once per episode & sample real obs in the dream window
        if tod == dream_tod:
            current_ep.dreamed_obs.append(dream_trajectory(algo, dream_h))
        if in_cyclic_window(tod, dream_tod, dream_h, period=env.episode_length):
            current_ep.sampled_obs.append(env._get_obs())

        # log current env state
        current_ep.add_step(env)

    return runlog, env  # return env for plotting-time shape inference


def build_obs_slices(env) -> Dict[str, slice]:
    """Compute index slices for (device, vm, va, load, line) inside the flat obs vector."""
    obs = np.array([], dtype=np.float32)
    for dev in env.devices.values():
        obs = np.concatenate([obs, dev.state.as_vector().astype(np.float32)])
    device_obs_size = obs.shape[0]
    vm_obs_size = len(env.net.res_bus)
    va_obs_size = len(env.net.res_bus)
    load_obs_size = env.net.load[["p_mw", "q_mvar"]].values.size
    line_obs_size = len(env.net.res_line)

    idx = {}
    i = 0
    idx["device"] = slice(i, i + device_obs_size); i += device_obs_size
    idx["vm"]     = slice(i, i + vm_obs_size);     i += vm_obs_size
    idx["va"]     = slice(i, i + va_obs_size);     i += va_obs_size
    idx["load"]   = slice(i, i + load_obs_size);   i += load_obs_size
    idx["line"]   = slice(i, i + line_obs_size);   i += line_obs_size
    idx["total"]  = i
    return idx

# def compare_dream_vs_sample_heatmap(env, runlog, day, fname=None):
#     """Heatmap comparing dreamed vs sampled obs (voltage angles removed)."""
#     idx = build_obs_slices(env)
#     last_ep = runlog.episodes[day]

#     dreamed_obs = np.array(last_ep.dreamed_obs)[0]   # (H, D)
#     sampled_obs = np.array(last_ep.sampled_obs)      # (H, D)
#     assert dreamed_obs.shape[0] == sampled_obs.shape[0]
#     assert dreamed_obs.shape[1] == idx["total"]

#     def drop_va(x):
#         return np.concatenate([x[:, :idx["vm"].stop], x[:, idx["va"].stop:]], axis=-1)

#     dreamed_wo_va = drop_va(dreamed_obs)
#     sampled_wo_va = drop_va(sampled_obs)
#     diff = dreamed_wo_va - sampled_wo_va
#     m = float(np.abs(diff).max())

#     fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
#     im0 = ax[0].imshow(dreamed_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
#     ax[0].set_title('dreamed_obs'); fig.colorbar(im0, ax=ax[0], shrink=0.8)

#     im1 = ax[1].imshow(sampled_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
#     ax[1].set_title('sampled_obs'); fig.colorbar(im1, ax=ax[1], shrink=0.8)

#     im2 = ax[2].imshow(diff.T, aspect='auto', cmap='coolwarm', vmin=-m, vmax=m, origin='lower')
#     ax[2].set_title('difference = dreamed - sampled')
#     fig.colorbar(im2, ax=ax[2], shrink=0.8, label='Δ value')

#     for a in ax:
#         a.set_xlabel('Prediction horizon', fontsize=11)
#         a.set_ylabel('Observation element', fontsize=11)
#         a.tick_params(labelsize=9)

#     if fname:
#         plt.savefig(fname, bbox_inches="tight", dpi=200)
#     plt.show()







from matplotlib.patches import Rectangle
from matplotlib import patheffects

def drop_va_slices(idx):
    # Map original slices -> new map after removing VA
    L_dev  = idx["device"].stop - idx["device"].start
    L_vm   = idx["vm"].stop     - idx["vm"].start
    L_load = idx["load"].stop   - idx["load"].start
    L_line = idx["line"].stop   - idx["line"].start
    i = 0
    new = {}
    new["device"] = slice(i, i + L_dev);  i += L_dev
    new["vm"]     = slice(i, i + L_vm);   i += L_vm
    new["load"]   = slice(i, i + L_load); i += L_load
    new["line"]   = slice(i, i + L_line); i += L_line
    new["total"]  = i
    return new


def annotate_sections_on_ax(ax, idx_wo_va, names_colors=None, text_ax="center"):
    """
    Draw dashed separator lines and centered labels for sections on ONE axes.
    No background fill is used.

    text_ax: "center" -> label centered horizontally,
             float in [0,1] -> axes x-coordinate for label placement (e.g., 0.04 = left gutter)
    """
    if names_colors is None:
        names_colors = [
            ("device", "#1f77b4"),
            ("vm",     "#d62728"),
            ("load",   "#2ca02c"),
            ("line",   "#ff7f0e"),
        ]

    x0, x1 = ax.get_xlim()
    x_text = 0.5 if text_ax == "center" else float(text_ax)

    for name, color in names_colors:
        s = idx_wo_va.get(name)
        if s is None:
            continue
        y_top = s.stop - 0.5
        y_mid = (s.start + s.stop - 1) / 2.0

        # dashed boundary
        ax.hlines(y_top, x0, x1, colors=color, linestyles="--", linewidth=0.8, alpha=0.6)
        if name == "device":
            name = "ess+dg"

        # label with thin white stroke (no background box)
        txt = ax.text(
            x_text, y_mid, name.upper(),
            color=color,
            transform=ax.get_yaxis_transform() if text_ax != "center" else
                      (lambda: None)(),  # placeholder; we'll set transform below
            ha="center" if text_ax == "center" else "left",
            va="center",
            fontsize=9.5, fontweight="bold",
        )
        # set transform depending on placement
        if text_ax == "center":
            # x in axes coords, y in data coords
            from matplotlib.transforms import blended_transform_factory
            txt.set_transform(blended_transform_factory(ax.transAxes, ax.transData))
        else:
            txt.set_transform(blended_transform_factory(ax.transAxes, ax.transData))

        # white outline for readability
        txt.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])

    ax.set_ylim(-0.5, idx_wo_va["total"] - 0.5)


def compare_dream_vs_sample_heatmap(env, runlog, day, fname=None,
                                    suptitle="IEEE-34 (Winter) — January 15",
                                    annotate_on=0):
    """Heatmap comparing dreamed vs sampled obs with VA removed and centered annotations on one panel."""
    idx = build_obs_slices(env)
    ep  = runlog.episodes[day]

    dreamed_obs = np.array(ep.dreamed_obs)[0]   # (H, D)
    sampled_obs = np.array(ep.sampled_obs)      # (H, D)
    assert dreamed_obs.shape[0] == sampled_obs.shape[0]
    assert dreamed_obs.shape[1] == idx["total"]

    # drop VA
    def drop_va(x):
        return np.concatenate([x[:, :idx["vm"].stop], x[:, idx["va"].stop:]], axis=-1)

    dreamed_wo_va = drop_va(dreamed_obs)
    sampled_wo_va = drop_va(sampled_obs)

    # slice map after removing VA
    def drop_va_slices(idx):
        L_dev  = idx["device"].stop - idx["device"].start
        L_vm   = idx["vm"].stop     - idx["vm"].start
        L_load = idx["load"].stop   - idx["load"].start
        L_line = idx["line"].stop   - idx["line"].start
        i = 0
        new = {}
        new["device"] = slice(i, i+L_dev);  i += L_dev
        new["vm"]     = slice(i, i+L_vm);   i += L_vm
        new["load"]   = slice(i, i+L_load); i += L_load
        new["line"]   = slice(i, i+L_line); i += L_line
        new["total"]  = i
        return new

    idx_wo_va = drop_va_slices(idx)

    diff = dreamed_wo_va - sampled_wo_va
    m = float(np.abs(diff).max()) if diff.size else 1.0

    # Turn off constrained_layout to create room for suptitle
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.subplots_adjust(wspace=0.25, top=0.82)  # <- leaves space for the title

    # # AFTER (smaller figure + shared y + tighter pads)
    # fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True, sharey=True)
    # fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.0, wspace=0.06)

    im0 = ax[0].imshow(dreamed_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
    ax[0].set_title('dreamed_obs'); fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(sampled_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
    ax[1].set_title('sampled_obs'); fig.colorbar(im1, ax=ax[1], shrink=0.8)

    im2 = ax[2].imshow(diff.T, aspect='auto', cmap='coolwarm', vmin=-m, vmax=m, origin='lower')
    ax[2].set_title('difference = dreamed - sampled')
    fig.colorbar(im2, ax=ax[2], shrink=0.8, label='Δ value')

    # annotate only one panel (0, 1, or 2). Use centered text without background.
    if annotate_on in (0, 1, 2):
        annotate_sections_on_ax(ax[annotate_on], idx_wo_va, text_ax="center")

    for a in ax:
        a.set_xlabel('Prediction horizon', fontsize=11)
        a.set_ylabel('Observation element', fontsize=11)
        a.tick_params(labelsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()