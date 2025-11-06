"""
Plot DreamerV3 schedules for the IEEE-34 microgrid on two representative days:
a winter-like day and a summer-like day (indices configurable).

Expected pickle structure (from `asdict(runlog)`):
  data = {
      "episodes": [
          {
              "soc1": [24 floats], "p_ess1": [24 floats],
              "soc2": [24 floats], "p_ess2": [24 floats],
              "p_dg1": [24 floats], "q_dg1": [24 floats],
              "p_dg2": [24 floats], "q_dg2": [24 floats],
              "p_grid": [24 floats], "q_grid": [24 floats],
              "vm": [24 arrays of shape (n_buses,)],
              # Optional:
              # "price": [24 floats],
              # "line_loading": [24 arrays of shape (n_lines,)] OR "line_loading_pct": same in %
              # "dreamed_obs": array (H,D), "sampled_obs": array (H,D)  <-- (for heatmaps)
          },
          ...
      ]
  }
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------
# Utility helpers
# -----------------------------
def get_episode(data: dict, day_idx: int) -> dict:
    episodes = data.get("episodes", None)
    if not isinstance(episodes, list) or len(episodes) == 0:
        raise ValueError("data['episodes'] must be a non-empty list of episode dicts.")
    if not (0 <= day_idx < len(episodes)):
        raise IndexError(f"day_idx {day_idx} out of range [0, {len(episodes)-1}]")
    return episodes[day_idx]


def _get_optional_vec(ep: dict, keys):
    """Return the first available vector among `keys`, else None."""
    for k in keys:
        if k in ep and isinstance(ep[k], (list, tuple, np.ndarray)) and len(ep[k]) == 24:
            return np.asarray(ep[k], dtype=float)
    return None


def _compute_import_export(p_grid_24: np.ndarray):
    """Split grid exchange into import (>=0) and export (>=0 after sign flip)."""
    p_imp = np.clip(p_grid_24, a_min=0.0, a_max=None)
    p_exp = np.clip(-p_grid_24, a_min=0.0, a_max=None)
    return p_imp, p_exp


def _max_line_loading(ep: dict):
    """
    Return (24,) array of max line loading in % if available.
    Looks for 'line_loading_pct' (preferred) or 'line_loading' (assumes fraction 0..1).
    """
    if "line_loading_pct" in ep:
        arrs = ep["line_loading_pct"]
        if isinstance(arrs, list) and len(arrs) == 24:
            return np.array([np.max(np.asarray(a, dtype=float)) for a in arrs], dtype=float)
    if "line_loading" in ep:
        arrs = ep["line_loading"]
        if isinstance(arrs, list) and len(arrs) == 24:
            # assume 0..1 fraction => convert to %
            return 100.0 * np.array([np.max(np.asarray(a, dtype=float)) for a in arrs], dtype=float)
    return None


def _voltage_envelope(ep: dict):
    """Return (vmin,vmax) each shape (24,) from list of per-step bus voltages."""
    vm_list = ep.get("vm", None)
    if not isinstance(vm_list, list) or len(vm_list) != 24:
        return None, None
    vm = np.stack([np.asarray(x, dtype=float) for x in vm_list], axis=0)  # (24, n_buses)
    return vm.min(axis=1), vm.max(axis=1)


def _maybe_invert_battery_sign(p_ess, want_positive_charge=True):
    """
    If your logged convention is opposite to the desired plot (positive=charge, negative=discharge),
    set `want_positive_charge=True` and flip the sign if needed.
    Here we assume your P is +discharge / -charge; so flip to show +charge / -discharge.
    """
    # Heuristic: if average is positive, we assume +discharge -> flip
    if want_positive_charge and np.nanmean(p_ess) > 0:
        return -p_ess
    return p_ess


# -----------------------------
# Plotting for one day
# -----------------------------
def plot_day(ep: dict, day_idx: int, out_png: str | None = None, title_prefix: str = "IEEE-34"):
    """
    Create a multi-panel figure for a single day:
      (1) price (if available)
      (2) DG1 & DG2 P
      (3) ESS1/ESS2 bar P (+charge, –discharge) + SoC (twin axis)
      (4) grid import/export
      (5) voltage min/max with 0.95/1.05 p.u. limits
      (6) max line loading %
    """
    hours = np.arange(24)

    # Required fields
    p_dg1  = np.asarray(ep["p_dg1"], dtype=float)
    p_dg2  = np.asarray(ep["p_dg2"], dtype=float)
    p_ess1 = np.asarray(ep["p_ess1"], dtype=float)
    p_ess2 = np.asarray(ep["p_ess2"], dtype=float)
    soc1   = np.asarray(ep["soc1"], dtype=float)
    soc2   = np.asarray(ep["soc2"], dtype=float)
    p_grid = np.asarray(ep["p_grid"], dtype=float)

    # Optional
    price  = _get_optional_vec(ep, ["price", "elec_price", "p_price"])
    vmin, vmax = _voltage_envelope(ep)
    load_pct   = None # _max_line_loading(ep)  # None if missing

    # Convert sign for battery bars: plot positive=charge, negative=discharge
    p_ess1_plot = _maybe_invert_battery_sign(p_ess1, want_positive_charge=True)
    p_ess2_plot = _maybe_invert_battery_sign(p_ess2, want_positive_charge=True)

    # Split grid import/export
    p_imp, p_exp = _compute_import_export(p_grid)

    # Layout: 6 rows if price and line loading exist, fewer otherwise
    rows = 6
    show_price = price is not None
    show_loading = load_pct is not None
    if not show_price:
        rows -= 1
    if not show_loading:
        rows -= 1

    fig, axs = plt.subplots(rows, 1, figsize=(8, 1.6 * rows), sharex=True)
    if rows == 1:
        axs = [axs]  # in case only one panel

    r = 0
    if day_idx == 14:
        date = "January 15"
    elif day_idx == 196:
        date = "July 15"
    else:
        date = f"Day {day_idx}"
    # (1) Price
    if show_price:
        ax = axs[r]
        ax.plot(hours, price, lw=2)
        ax.set_ylabel("Price ($/MWh)")
        ax.set_title(f"{title_prefix} — {date}")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        r += 1

    # (2) DGs
    ax = axs[r]
    ax.plot(hours, p_dg1, lw=2, label="DG1 P")
    ax.plot(hours, p_dg2, lw=2, label="DG2 P")
    ax.set_ylabel("DG P (MW)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    r += 1

    # (3) Batteries: bars + twin SoC
    ax = axs[r]
    barw = 0.4
    ax.bar(hours - barw/2, p_ess1_plot, width=barw, label="ESS1 (chg + / dis –)", alpha=0.8)
    ax.bar(hours + barw/2, p_ess2_plot, width=barw, label="ESS2 (chg + / dis –)", alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Battery P (MW)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax2 = ax.twinx()
    ax2.plot(hours, soc1, lw=2, color="tab:purple", label="ESS1 SoC")
    ax2.plot(hours, soc2, lw=2, color="tab:green", label="ESS2 SoC")
    ax2.set_ylabel("SoC")
    # combine legends
    # h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2, l2, frameon=False, ncol=2, loc="upper right")
    r += 1

    # (4) Grid import/export
    ax = axs[r]
    ax.bar(hours - 0.2, p_imp, width=0.4, label="Import", alpha=0.85)
    ax.bar(hours + 0.2, p_exp, width=0.4, label="Export", alpha=0.85)
    ax.set_ylabel("Grid P")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    r += 1

    # (5) Voltage envelope
    if vmin is not None and vmax is not None:
        ax = axs[r]
        ax.plot(hours, vmax, lw=2, label="Vmax")
        ax.plot(hours, vmin, lw=2, label="Vmin")
        ax.axhline(1.05, color="red", ls="--", lw=1)
        ax.axhline(0.95, color="red", ls="--", lw=1)
        ax.set_ylabel("Voltage (p.u.)")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        r += 1

    # (6) Max line loading %
    if show_loading:
        ax = axs[r]
        ax.plot(hours, load_pct, lw=2)
        ax.set_ylabel("Max line loading (%)")
        ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        r += 1

    axs[-1].set_xlabel("Hour")
    fig.align_ylabels(axs)           # align left y-labels
    fig.subplots_adjust(left=0.12)   # optional: make a consistent left margin
    plt.xlim(0, 23)
    plt.xticks(np.arange(0, 24, 2), labels=[f"{h:02d}:00" for h in range(1, 24, 2)])
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="./results/ieee34_dreamerv3.pkl", help="path to asdict(runlog) pickle")
    ap.add_argument("--winter_day", type=int, default=14, help="index in [0,365] (e.g., mid-Jan)")
    ap.add_argument("--summer_day", type=int, default=196, help="index in [0,365] (e.g., mid-Jul)")
    ap.add_argument("--outdir", default="./figures", help="where to save PNGs")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    # Winter-like example
    ep_w = get_episode(data, args.winter_day)
    out_w = os.path.join(args.outdir, f"ieee34_day_{args.winter_day}_schedules.png")
    plot_day(ep_w, args.winter_day, out_png=out_w, title_prefix="IEEE-34 (winter)")

    # Summer-like example
    ep_s = get_episode(data, args.summer_day)
    out_s = os.path.join(args.outdir, f"ieee34_day_{args.summer_day}_schedules.png")
    plot_day(ep_s, args.summer_day, out_png=out_s, title_prefix="IEEE-34 (summer)")


if __name__ == "__main__":
    main()
