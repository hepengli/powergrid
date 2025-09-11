# PowerGrid Gym Environment

A lightweight, production-style **Gymnasium** environment for **power grid control** built on **pandapower** with a small, modular set of **device models** (DG/RES/ESS/Shunt/Transformer/Grid). It’s designed for RL/control research: clean action/observation spaces, centralized safety metrics, pluggable rewards, and tidy code structure.

---

## Highlights

- **Gymnasium-compatible** single-agent env (`GridBaseEnv`)
- **Pandapower** integration with idempotent device→network attachment
- **Mixed actions**: continuous (`Box`) + optional discrete (`Discrete`/`MultiDiscrete`) via a `Dict` action space
- **Action normalization** wrapper so agents act in `[-1, 1]`
- **Safety framework** (`SafetySpec`, `total_safety`) to mix penalties (over-rating, PF, SOC, voltage, loading, ROC)
- **Device library**: `DG`, `RES` (solar/wind), `ESS`, `Shunt`, `Transformer` (OLTC), `Grid`
- **Cost helpers**: quadratic, piecewise linear, ramping, tap wear, energy settlement
- **Unit tests** for devices and grid logic

---

## Installation

```bash
# Python 3.12 recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# inside the repo root
pip install -U pip
pip install -e ".[dev]"    # or: pip install -r requirements.txt


# Quick Start
from powergrid.envs.single_agent.ieee13_env import IEEE13Env
from powergrid.envs.grid_base_env import NormalizeActionWrapper

# Create and wrap: agent acts in [-1,1] for the continuous part
env = NormalizeActionWrapper(IEEE13Env({"episode_length": 24, "train": True}))

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("reward=", reward, "converged=", info.get("converged"))
