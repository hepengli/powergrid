# Multi-Agent Quickstart Guide

Welcome to the PowerGrid multi-agent control framework! This guide will help you get started with building and training multi-agent reinforcement learning systems for power grid control.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [GridAgent vs DeviceAgent](#gridagent-vs-deviceagent)
5. [Vertical vs Horizontal Protocols](#vertical-vs-horizontal-protocols)
6. [Tutorial 1: Simple 2-Microgrid Setup](#tutorial-1-simple-2-microgrid-setup)
7. [Tutorial 2: Training with RLlib MAPPO](#tutorial-2-training-with-rllib-mappo)
8. [Tutorial 3: P2P Trading Example](#tutorial-3-p2p-trading-example)
9. [FAQ and Troubleshooting](#faq-and-troubleshooting)

---

## Introduction

The PowerGrid multi-agent framework provides:

- **Hierarchical control**: GridAgents coordinate DeviceAgents (ESS, DG, RES)
- **Flexible coordination**: Vertical (parent→child) and horizontal (peer↔peer) protocols
- **MARL compatibility**: PettingZoo API for RLlib, QMIX, MADDPG, etc.
- **Realistic simulation**: Pandapower power flow, cost functions, safety constraints

**Key Benefits:**
- Scale from 2 to 10+ microgrids
- Mix centralized and decentralized control
- Plug-and-play coordination protocols
- Ready for research and production

---

## Architecture Overview

### Three-Layer Hierarchy

```
┌─────────────────────────────────────────────────┐
│         Environment (PettingZoo)                │
│  - Manages GridAgents                           │
│  - Runs horizontal protocols (P2P trading)      │
│  - Executes power flow                          │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌─────┐     ┌─────┐     ┌─────┐
    │ MG1 │     │ MG2 │     │ MG3 │   GridAgents (Level 2)
    │Grid │     │Grid │     │Grid │   - Microgrid controllers
    │Agent│     │Agent│     │Agent│   - RL-controllable
    └─────┘     └─────┘     └─────┘   - Run vertical protocols
       │           │           │
   ┌───┼───┐   ┌───┼───┐   ┌───┼───┐
   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
  ESS DG  PV  ESS DG  PV  ESS DG  PV  DeviceAgents (Level 1)
                                      - Physical devices
                                      - Respond to setpoints/prices
```

### Key Components

1. **Environment** (`MultiAgentPowerGridEnv`):
   - PettingZoo `ParallelEnv` interface
   - Manages all GridAgents
   - Executes coordination protocols
   - Runs pandapower simulation

2. **GridAgent**:
   - Controls a microgrid
   - RL-trainable policy
   - Coordinates subordinate devices
   - Can trade with peer GridAgents

3. **DeviceAgent**:
   - Wraps physical device (ESS, DG, RES)
   - Responds to coordination signals
   - Updates device state

---

## Quick Start

### Installation

```bash
# Install PowerGrid with multi-agent support
pip install -e .

# Install RLlib (optional, for training)
pip install "ray[rllib]==2.9.0"

# Install PettingZoo
pip install pettingzoo>=1.24.0
```

### 30-Second Example

```python
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.devices import ESS, DG

# Configure environment
config = {
    'microgrids': [
        {
            'name': 'MG1',
            'network': IEEE13Bus('MG1'),
            'devices': [
                ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                    capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2),
                DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.66,
                   cost_curve_coefs=[100, 72.4, 0.5]),
            ],
            'vertical_protocol': 'none',
            'dataset': {...}  # Load/solar/wind data
        },
        # Add more microgrids...
    ],
    'horizontal_protocol': 'none',
    'episode_length': 24,
    'train': True,
}

# Create environment
env = MultiAgentPowerGridEnv(config)

# Run episode
obs, info = env.reset()
for t in range(24):
    actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
    obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## GridAgent vs DeviceAgent

### GridAgent (Level 2)

**What it is:**
- A microgrid controller
- The primary RL-trainable agent
- Manages multiple devices

**Responsibilities:**
- Observe aggregated state from devices
- Compute high-level actions (prices, setpoints, or direct control)
- Coordinate subordinate DeviceAgents
- Trade with peer GridAgents

**Example:**
```python
from powergrid.agents import GridAgent, DeviceAgent
from powergrid.devices import ESS, DG

# Create device agents
ess_agent = DeviceAgent('ESS1', device=ESS(...))
dg_agent = DeviceAgent('DG1', device=DG(...))

# Create grid agent
grid_agent = GridAgent(
    agent_id='MG1',
    subordinates=[ess_agent, dg_agent],
    vertical_protocol=PriceSignalProtocol(),
    centralized=True  # Output joint action for all devices
)
```

### DeviceAgent (Level 1)

**What it is:**
- A wrapper around a physical device
- Managed by a GridAgent
- Not directly RL-trainable (responds to signals)

**Responsibilities:**
- Observe device state (P, Q, SOC)
- Execute device dynamics
- Respond to coordination signals (prices, setpoints)

**Device Types:**
- `ESS`: Energy Storage System (battery)
- `DG`: Distributed Generator (diesel, natural gas)
- `RES`: Renewable Energy Source (solar, wind)

---

## Vertical vs Horizontal Protocols

### Vertical Protocols (Agent-Owned)

**Purpose:** Parent → subordinate coordination

**Ownership:** Each GridAgent owns its own vertical protocol

**Use Cases:**
- Price signals (economic dispatch)
- Setpoint commands (centralized control)
- Constraint broadcasting

**Available Protocols:**
- `NoProtocol`: No coordination, devices act independently
- `PriceSignalProtocol`: Broadcast marginal price to devices
- `SetpointProtocol`: Assign power setpoints to devices

**Example:**
```python
from powergrid.agents.protocols import PriceSignalProtocol

grid_agent = GridAgent(
    agent_id='MG1',
    subordinates=[...],
    vertical_protocol=PriceSignalProtocol(initial_price=50.0),
    centralized=True
)

# During step, GridAgent runs:
# signals = vertical_protocol.coordinate(subordinate_obs, parent_action)
# → {'ESS1': {'price': 50.0}, 'DG1': {'price': 50.0}}
```

### Horizontal Protocols (Environment-Owned)

**Purpose:** Peer ↔ peer coordination

**Ownership:** The environment owns and runs horizontal protocols

**Use Cases:**
- Peer-to-peer energy trading
- Distributed consensus (frequency regulation)
- Cooperative market bidding

**Available Protocols:**
- `NoHorizontalProtocol`: No peer coordination
- `PeerToPeerTradingProtocol`: Market-based energy trading
- `ConsensusProtocol`: Gossip-based consensus

**Example:**
```python
config = {
    'microgrids': [...],
    'horizontal_protocol': 'p2p_trading',  # Enable P2P trading
    'topology': None,  # Fully connected (all can trade)
}

env = MultiAgentPowerGridEnv(config)

# During step, environment runs:
# 1. Collect bids/offers from all agents
# 2. Clear market (match buyers and sellers)
# 3. Send trade confirmations to agents
```

---

## Tutorial 1: Simple 2-Microgrid Setup

Let's build a simple 2-microgrid system from scratch.

### Step 1: Import Dependencies

```python
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.devices import ESS, DG
import pickle

# Load dataset
with open('data/data2024.pkl', 'rb') as f:
    dataset = pickle.load(f)

def read_data(d, load_area, renew_area):
    return {
        'load': d['load'][load_area],
        'solar': d['solar'][renew_area],
        'wind': d['wind'][renew_area],
        'price': d['price']['LMP']
    }
```

### Step 2: Configure Microgrids

```python
config = {
    'network': None,  # No base network (standalone microgrids)
    'microgrids': [
        {
            'name': 'MG1',
            'network': IEEE13Bus('MG1'),
            'devices': [
                ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                    capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2, init_soc=0.5),
                DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.66,
                   cost_curve_coefs=[100, 72.4, 0.5011]),
                DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
            ],
            'vertical_protocol': 'none',
            'dataset': read_data(dataset, 'AVA', 'NP15')
        },
        {
            'name': 'MG2',
            'network': IEEE13Bus('MG2'),
            'devices': [
                ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                    capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2, init_soc=0.5),
                DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.60,
                   cost_curve_coefs=[100, 51.6, 0.4615]),
                DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
            ],
            'vertical_protocol': 'none',
            'dataset': read_data(dataset, 'BANCMID', 'NP15')
        }
    ],
    'horizontal_protocol': 'none',
    'episode_length': 24,
    'train': True,
    'penalty': 10,
    'share_reward': False
}
```

### Step 3: Create and Run Environment

```python
# Create environment
env = MultiAgentPowerGridEnv(config)

print(f"Agents: {env.possible_agents}")
print(f"Action spaces: {env.action_spaces}")

# Reset
obs, info = env.reset(seed=42)

# Run one episode
for t in range(24):
    # Sample random actions
    actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}

    # Step
    obs, rewards, dones, truncated, infos = env.step(actions)

    # Print progress
    if t % 6 == 0:
        print(f"Hour {t}: MG1 reward={rewards['MG1']:.2f}, "
              f"MG2 reward={rewards['MG2']:.2f}")

print("Episode complete!")
```

**Run the full example:**
```bash
python examples/multi_agent/simple_2mg.py
```

---

## Tutorial 2: Training with RLlib MAPPO

Train a shared policy using Multi-Agent PPO (MAPPO).

### Step 1: Install RLlib

```bash
pip install "ray[rllib]==2.9.0"
```

### Step 2: Train with Default Settings

```bash
# Train for 100 iterations with shared policy
python examples/train_mappo_microgrids.py --iterations 100

# Train with independent policies (IPPO)
python examples/train_mappo_microgrids.py --iterations 100 --independent-policies

# With W&B logging
python examples/train_mappo_microgrids.py --iterations 100 --wandb
```

### Step 3: Custom Training Script

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2

# Initialize Ray
ray.init()

# Create environment
def env_creator(config):
    env = MultiAgentMicrogridsV2(config)
    return ParallelPettingZooEnv(env)

# Configure PPO
config = (
    PPOConfig()
    .environment(env=env_creator, env_config={'train': True, 'penalty': 10})
    .framework("torch")
    .training(
        train_batch_size=4000,
        lr=5e-5,
        gamma=0.99,
    )
    .multi_agent(
        policies={'shared_policy': (None, None, None, {})},
        policy_mapping_fn=lambda aid, *args: 'shared_policy',
    )
    .rollouts(num_rollout_workers=4)
)

# Build and train
algo = config.build()

for i in range(100):
    result = algo.train()
    print(f"Iteration {i+1}: reward = {result['episode_reward_mean']:.2f}")

    # Save checkpoint
    if (i + 1) % 10 == 0:
        algo.save(f"./checkpoints/iter_{i+1}")

algo.stop()
ray.shutdown()
```

### Step 4: Evaluate Trained Policy

```python
# Load checkpoint
algo = PPOConfig().build()
algo.restore("./checkpoints/iter_100")

# Run evaluation
env = env_creator({'train': False, 'penalty': 10})
obs, info = env.reset()

for t in range(24):
    actions = {}
    for aid in env.possible_agents:
        action = algo.compute_single_action(obs[aid], policy_id='shared_policy')
        actions[aid] = action

    obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## Tutorial 3: P2P Trading Example

Enable peer-to-peer energy trading between microgrids.

### Concept

- Agents with excess generation → **Sellers** (offer low prices)
- Agents with net demand → **Buyers** (bid high prices)
- Environment clears market and confirms trades

### Configuration

```python
config = {
    'microgrids': [
        {'name': 'MG1', ...},
        {'name': 'MG2', ...},
        {'name': 'MG3', ...},
    ],
    'horizontal_protocol': 'p2p_trading',  # Enable P2P trading
    'topology': None,  # Fully connected (all can trade with all)
}

env = MultiAgentPowerGridEnv(config)
```

### Trading Flow

1. **Agents observe** their net demand
2. **Environment collects** bids (buyers) and offers (sellers)
3. **Market clears**: Match highest bids with lowest offers
4. **Trade confirmations** sent to agents via messages

### Run Example

```bash
python examples/multi_agent/p2p_trading_3mg.py
```

### Check Trades

```python
# After step, check agent mailbox for trades
for agent_id, agent in env.agents.items():
    for msg in agent.mailbox:
        if 'trades' in msg.content:
            trades = msg.content['trades']
            print(f"{agent_id} trades: {trades}")
            # Output: [{'counterparty': 'MG2', 'quantity': 0.3, 'price': 55.0}]
```

---

## FAQ and Troubleshooting

### Q: How do I add a new device type?

**A:** Subclass `Device` and implement `step()`, `compute_cost()`, `compute_safety()`:

```python
from powergrid.devices import Device

class MyDevice(Device):
    def __init__(self, name, bus, ...):
        super().__init__()
        # Initialize device parameters

    def step(self):
        # Update device state (dynamics)
        pass

    def compute_cost(self):
        # Calculate operational cost
        self.cost = ...

    def compute_safety(self, converged):
        # Calculate safety violations
        self.safety = ...
```

### Q: Can I customize the reward function?

**A:** Yes, subclass `MultiAgentPowerGridEnv` and override `_compute_rewards()`:

```python
class CustomEnv(MultiAgentPowerGridEnv):
    def _compute_rewards(self, converged):
        rewards = {}
        for aid, agent in self.agents.items():
            # Custom reward logic
            reward = -agent.cost + 0.5 * efficiency_bonus
            rewards[aid] = reward
        return rewards
```

### Q: How do I create a custom protocol?

**A:** Subclass `VerticalProtocol` or `HorizontalProtocol`:

```python
from powergrid.agents.protocols import VerticalProtocol

class MyProtocol(VerticalProtocol):
    def coordinate(self, subordinate_observations, parent_action=None):
        # Custom coordination logic
        signals = {}
        for sub_id, obs in subordinate_observations.items():
            signals[sub_id] = {'custom_signal': compute_signal(obs)}
        return signals
```

See the [Protocol Guide](protocol_guide.md) for detailed instructions.

### Q: Power flow doesn't converge. What should I do?

**A:** Common causes:
1. **Device limits violated**: Check `min_p_mw`, `max_p_mw` constraints
2. **SOC out of bounds**: Ensure ESS doesn't hit 0% or 100%
3. **Large power mismatches**: Scale down device ratings or loads

**Solutions:**
- Increase `penalty` coefficient (default: 10)
- Add action clipping in your policy
- Tune device parameters

### Q: Training is slow. How to speed up?

**A:**
1. Increase `num_rollout_workers` (parallel simulation)
2. Reduce `train_batch_size` (faster iterations)
3. Use GPU: `config.resources(num_gpus=1)`
4. Reduce `episode_length` (e.g., 12 instead of 24)

### Q: How do I visualize results?

**A:** Use W&B or Tensorboard:

```bash
# With W&B
python examples/train_mappo_microgrids.py --wandb --wandb-project my-project

# With Tensorboard (RLlib default)
tensorboard --logdir ~/ray_results
```

---

## Next Steps

- **Advanced Protocols**: Read the [Protocol Guide](protocol_guide.md)
- **API Reference**: See docstrings in `powergrid.envs.multi_agent`
- **Examples**: Explore `examples/multi_agent/`
- **Research**: Implement your own coordination algorithms!

---

## Getting Help

- **Issues**: https://github.com/yourorg/powergrid/issues
- **Discussions**: https://github.com/yourorg/powergrid/discussions
- **Email**: support@yourorg.com

Happy training! ⚡️
