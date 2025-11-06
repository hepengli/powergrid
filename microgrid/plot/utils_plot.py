import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tensorboard.backend.event_processing import event_accumulator

from pathlib import Path
from typing import Dict, List, Tuple

""" Training plot utils"""
def moving_average(a, w):
    if w <= 1:
        return a
    pad = w // 2
    # Reflect padding to avoid edge shrinkage
    a_pad = np.pad(a, (pad, pad), mode='reflect')
    kernel = np.ones(w) / w
    return np.convolve(a_pad, kernel, mode='valid')

def aggregate(metric_array, smooth=1):
    """
    metric_array: shape (seeds, T)
    Returns mean[T], std[T] after (optional) smoothing per-seed.
    """
    if metric_array.ndim != 2:
        raise ValueError("Expected (seeds, T)")
    seeds, T = metric_array.shape
    smoothed = np.vstack([moving_average(metric_array[i], smooth) for i in range(seeds)])
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)
    return mean, std

def _k_formatter(x, pos):
    # Show e.g. 12,500 -> 12.5k; -300 -> -300 (no k if < 1000)
    ax = abs(x)
    if ax >= 1000:
        val = x / 1000.0
        # strip trailing .0
        s = f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"

_kfmt = mtick.FuncFormatter(_k_formatter)

def plot_metric(metric_name, series_dict, x, smooth=1, fname=None):
    plt.figure(figsize=(5, 4))
    for label, d in series_dict.items():
        mean, std = aggregate(d[metric_name], smooth=smooth)
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax = plt.gca()
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(_kfmt)
    ax.yaxis.set_major_formatter(_kfmt)
    # Optional: keep a sensible number of ticks
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))

    plt.xlabel("Environment Steps", fontsize=11)
    plt.ylabel(metric_name.capitalize(), fontsize=11)
    # plt.title(f"{metric_name.capitalize()} vs Environment Steps")
    plt.xlim(0,50000)
    plt.legend(fontsize=9)
    plt.tight_layout(pad=0.5)
    plt.grid(True, alpha=0.3)
    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()

def _discover_event_files(root: Path, max_seeds: int | None = None) -> List[Path]:
    """
    Find the newest TensorBoard event file in each run directory under `root`.
    Returns a list of Paths, sorted by mtime (newest first). If max_seeds is
    provided, returns at most that many.
    """
    # Collect candidates grouped by their parent run directory (e.g., pid folders)
    per_run_latest: Dict[Path, Tuple[float, Path]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fnmatch.fnmatch(fname, "events.out.tfevents.*"):
                fpath = Path(dirpath) / fname
                run_dir = Path(dirpath)  # assume each run has its own folder
                mtime = fpath.stat().st_mtime
                cur = per_run_latest.get(run_dir)
                if (cur is None) or (mtime > cur[0]):
                    per_run_latest[run_dir] = (mtime, fpath)

    # Take the newest file per run, then sort runs by file mtime desc
    picked = [rec[1] for rec in per_run_latest.values()]
    picked.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if max_seeds is not None:
        picked = picked[:max_seeds]
    return picked

def _moving_to_arrays(ea: event_accumulator.EventAccumulator, tag: str) -> np.ndarray:
    scalars = ea.Scalars(tag)
    if not scalars:
        return np.asarray([], dtype=np.float32)
    return np.asarray([s.value for s in scalars], dtype=np.float32)

def read_tb_events_auto(
    case: str,
    algo: str,
    base_dir: str,
    tags: Dict[str, str] | None = None,
    prefix: str = "train/",
    max_seeds: int | None = 3,
) -> Dict[str, np.ndarray]:
    """
    Auto-discover and load TensorBoard scalar series for multiple seeds.

    Returns dict of arrays shaped (num_seeds, T) with T = min length across seeds for that metric.
    """
    if tags is None:
        tags = {"reward": "reward", "safety": "safety", "return": "return"}

    root = Path(base_dir) / case / algo
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    event_files = _discover_event_files(root, max_seeds=max_seeds)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {root}")

    per_metric_series: Dict[str, List[np.ndarray]] = {k: [] for k in tags}
    lengths_by_metric: Dict[str, List[int]] = {k: [] for k in tags}

    for ef in event_files:
        ea = event_accumulator.EventAccumulator(str(ef))
        ea.Reload()
        for out_key, tag_suffix in tags.items():
            full_tag = f"{prefix}{tag_suffix}" if prefix else tag_suffix
            arr = _moving_to_arrays(ea, full_tag)
            per_metric_series[out_key].append(arr)
            lengths_by_metric[out_key].append(arr.shape[0])

    aligned: Dict[str, np.ndarray] = {}
    for out_key, series_list in per_metric_series.items():
        if not series_list:
            aligned[out_key] = np.zeros((0, 0), dtype=np.float32)
            continue
        T = min(lengths_by_metric[out_key]) if lengths_by_metric[out_key] else 0
        if T == 0:
            aligned[out_key] = np.zeros((len(series_list), 0), dtype=np.float32)
            continue
        trimmed = [s[:T] if s.size >= T else np.pad(s, (0, T - s.size)) for s in series_list]
        aligned[out_key] = np.stack(trimmed, axis=0).astype(np.float32, copy=False)

    return aligned


""" Testing plot utils"""

def _kfmt(x, pos):
    ax = abs(x)
    if ax >= 1000:
        s = f"{x/1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"


def plot_soc(runlog, max_days: int = 7, fname=None):
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    n = min(max_days, runlog.days)
    for d in range(n):
        ax.plot(runlog.episodes[d].soc, label=f"Day {d}")
    ax.set_xlabel("Hour", fontsize=11)
    ax.set_ylabel("SOC", fontsize=11)
    ax.legend(ncol=min(4, n), frameon=False)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_kfmt))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_kfmt))
    ax.tick_params(labelsize=9)
    plt.tight_layout(pad=0)
    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()

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


def compare_dream_vs_sample_heatmap(env, runlog, fname=None):
    """Heatmap comparing dreamed vs sampled obs (voltage angles removed)."""
    idx = build_obs_slices(env)
    last_ep = runlog.episodes[-1]

    dreamed_obs = np.array(last_ep.dreamed_obs)[0]   # (H, D)
    sampled_obs = np.array(last_ep.sampled_obs)      # (H, D)
    assert dreamed_obs.shape[0] == sampled_obs.shape[0]
    assert dreamed_obs.shape[1] == idx["total"]

    def drop_va(x):
        return np.concatenate([x[:, :idx["vm"].stop], x[:, idx["va"].stop:]], axis=-1)

    dreamed_wo_va = drop_va(dreamed_obs)
    sampled_wo_va = drop_va(sampled_obs)
    diff = dreamed_wo_va - sampled_wo_va
    m = float(np.abs(diff).max())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    im0 = ax[0].imshow(dreamed_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
    ax[0].set_title('dreamed_obs'); fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(sampled_wo_va.T, aspect='auto', cmap='viridis', origin='lower')
    ax[1].set_title('sampled_obs'); fig.colorbar(im1, ax=ax[1], shrink=0.8)

    im2 = ax[2].imshow(diff.T, aspect='auto', cmap='coolwarm', vmin=-m, vmax=m, origin='lower')
    ax[2].set_title('difference = dreamed - sampled')
    fig.colorbar(im2, ax=ax[2], shrink=0.8, label='Δ value')

    for a in ax:
        a.set_xlabel('Prediction horizon', fontsize=11)
        a.set_ylabel('Observation element', fontsize=11)
        a.tick_params(labelsize=9)

    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()

