import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from utils_plot import _kfmt  # use your formatter directly

DAYS = 366
systems = ["ieee13", "cigre_mv", "ieee34"]

# Shared tick formatter
formatter = mtick.FuncFormatter(_kfmt)

for sys in systems:
    # Load results
    with open(f"./results/{sys}_dreamerv3.pkl", "rb") as f:
        d3 = pickle.load(f)
    with open(f"./results/{sys}_ppo.pkl", "rb") as f:
        ppo = pickle.load(f)
    with open(f"./results/{sys}_sac.pkl", "rb") as f:
        sac = pickle.load(f)
    with open(f"./results/{sys}_misocp.pkl", "rb") as f:
        miscop = pickle.load(f)

    # Extract daily cost and safety
    dreamerv3_cost = [-ep["ret"] for ep in d3["episodes"]]
    ppo_cost = [-ep["ret"] for ep in ppo["episodes"]]
    sac_cost = [-ep["ret"] for ep in sac["episodes"]]
    miscop_cost = [ep["obj"] for ep in miscop["results"]]

    dreamerv3_safety = [ep["safety"] for ep in d3["episodes"]]
    ppo_safety = [ep["safety"] for ep in ppo["episodes"]]
    sac_safety = [ep["safety"] for ep in sac["episodes"]]

    x = np.arange(1, DAYS + 1)

    # Compute cumulative sums
    cum_cost = {
        "DreamerV3": np.cumsum(dreamerv3_cost),
        "PPO": np.cumsum(ppo_cost),
        "SAC": np.cumsum(sac_cost),
        "MISOCP": np.cumsum(miscop_cost),
    }
    cum_safety = {
        "DreamerV3": np.cumsum(dreamerv3_safety),
        "PPO": np.cumsum(ppo_safety),
        "SAC": np.cumsum(sac_safety),
    }

    # ---- Plot Cumulative Cost ----
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, y in cum_cost.items():
        ax.plot(x, y, label=label, linewidth=1.8)
        print(label, y[-1])
        input()
    # ax.set_title(f"{sys.upper()} - Cumulative Cost", fontsize=12)
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Cumulative Cost ($)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=9)
    plt.xlim(0,366)
    plt.tight_layout()
    plt.savefig(f"./figures/{sys}_cumulative_cost.png", dpi=200)
    plt.close(fig)

    # ---- Plot Cumulative Safety ----
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, y in cum_safety.items():
        ax.plot(x, y, label=label, linewidth=1.8)
    # ax.set_title(f"{sys.upper()} - Cumulative Safety", fontsize=12)
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Cumulative Safety", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=9)
    plt.xlim(0,366)
    plt.tight_layout()
    plt.savefig(f"./figures/{sys}_cumulative_safety.png", dpi=200)
    plt.close(fig)

    print(f"Saved formatted plots for {sys}")
