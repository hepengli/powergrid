# PowerGrid: Async Multi-Agent RL for Real-World Power Systems

**Design Proposal** | Version 1.0 | 2025-10-07

---

## Executive Summary

This proposal outlines the evolution of PowerGrid from a single-agent RL environment into a **modular, hierarchical multi-agent platform** for power system research. The design targets two distinct user groups:

1. **Energy Researchers**: Domain experts who need to build custom devices and environments without deep RL knowledge
2. **RL Researchers**: Algorithm developers who want pre-built power grid systems with plug-and-play datasets

The architecture enables **asynchronous multi-agent control**, **hierarchical coordination**, and **real-world deployment** while maintaining backward compatibility with the current API.

---

## Current Architecture

### System Overview

The current implementation follows a **centralized single-agent** pattern where:

- Individual devices (DG, ESS, RES, Shunt) are passive components with `state` and `action` attributes
- `GridBaseEnv` acts as a monolithic coordinator that:
  - Concatenates all device actions into a single action vector
  - Distributes action slices to devices
  - Runs centralized power flow
  - Aggregates rewards from all devices
- RL algorithms interact with one unified agent



### Limitations

1. **No agent autonomy**: Devices cannot make independent decisions or learn local policies
2. **No communication**: Devices cannot coordinate directly (e.g., price signals, consensus)
3. **Synchronous execution**: All devices act simultaneously at the same frequency
4. **Tight coupling**: Adding new device types or coordination schemes requires core changes
5. **Limited scalability**: Single policy for 10+ devices becomes intractable

---

## Design Goals

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Support hierarchical multi-agent control (device/grid/system levels) | Must Have |
| FR2 | Enable agent-to-agent communication protocols | Must Have |
| FR3 | Plug-and-play device registration without core modifications | Must Have |
| FR4 | YAML-based environment configuration for non-programmers | Should Have |
| FR5 | Asynchronous agent execution with variable time-steps | Should Have |
| FR6 | Standard MARL API compatibility (PettingZoo/RLlib) | Must Have |
| FR7 | Dataset management system with pre-loaded real-world data | Should Have |
| FR8 | Hardware-in-the-loop (HIL) support for deployment | Nice to Have |

### Non-Functional Requirements

- **Backward Compatibility**: Existing `GridBaseEnv` code must continue working
- **Performance**: Support 100+ agents with <1s/step on CPU
- **Extensibility**: Users add devices/protocols via plugins, not core edits
- **Documentation**: 10-minute quickstart for both user personas

---

## Proposed Architecture

### High-Level Design

**Phased Implementation** (clarified through detailed design):

**Phase 1 (Week 3-4)**: Two-level hierarchy
```
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ GridAgent MG1 │  │ GridAgent MG2 │  │ GridAgent MG3 │  ← Level 2 (RL-controlled)
│  (Microgrid)  │  │  (Microgrid)  │  │  (Microgrid)  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
   ┌────┴───┐         ┌────┴───┐        ┌────┴───┐
   │        │         │        │        │        │
┌──▼──┐ ┌──▼──┐   ┌──▼──┐ ┌──▼──┐  ┌──▼──┐ ┌──▼──┐
│ ESS │ │ DG  │   │ ESS │ │Solar│  │ ESS │ │ DG  │  ← Level 1 (DeviceAgents)
└─────┘ └─────┘   └─────┘ └─────┘  └─────┘ └─────┘

Horizontal Protocol (Environment-owned): GridAgent ↔ GridAgent
Vertical Protocol (Agent-owned): GridAgent → DeviceAgents
```

**Phase 2 (Week 11-12)**: Three-level hierarchy
```
┌─────────────────────────────────────────────────────────┐
│                    SystemAgent (ISO)                    │  ← Level 3 (RL or rule-based)
│              (Market clearing, LMP computation)          │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼─────────┐  ┌────────▼────────┐
│   GridAgent MG1 │  │  GridAgent MG2  │  ← Level 2 (RL-controlled)
│  (MG Controller)│  │  (MG Controller)│
└───────┬─────────┘  └────────┬────────┘
        │                     │
   ┌────┴───┐            ┌────┴───┐
   │        │            │        │
┌──▼──┐ ┌──▼──┐      ┌──▼──┐ ┌──▼──┐
│ ESS │ │ DG  │      │Solar│ │ ESS │  ← Level 1 (DeviceAgents)
└─────┘ └─────┘      └─────┘ └─────┘
```

**Key Architectural Insights**:
1. **GridAgents are RL-controllable agents** (microgrid controllers), matching existing `MultiAgentMicrogrids`
2. **DeviceAgents are subordinates** managed internally by GridAgents
3. **Protocols split by ownership**:
   - Vertical (agent-owned): Parent → child (e.g., GridAgent → DeviceAgents)
   - Horizontal (environment-owned): Peer ↔ peer (e.g., GridAgent ↔ GridAgent)
4. **SystemAgent deferred** to Week 11-12 (not needed for basic multi-agent functionality)

### Key Components

#### 1. Agent Abstraction Layer

**Location**: `powergrid/agents/`

```python
# agents/base.py
class Agent(ABC):
    """Base class for all agents in the hierarchy."""

    @abstractmethod
    def observe(self, global_state: Dict) -> Observation:
        """Extract relevant observations from global state."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Compute action from observation."""

    @abstractmethod
    def receive_message(self, message: Message, sender: AgentID):
        """Handle incoming communication."""

    @abstractmethod
    def send_message(self, message: Message, recipients: List[AgentID]):
        """Send message to other agents."""

# agents/device_agent.py
class DeviceAgent(Agent):
    """Wraps a Device as an autonomous agent."""

    def __init__(self, device: Device, policy: Policy = None):
        self.device = device
        self.policy = policy or RandomPolicy()
        self.mailbox = []

    def observe(self, global_state):
        # Local device state + neighbor info
        return {
            'local': self.device.state.as_vector(),
            'bus_voltage': global_state['bus_vm'][self.device.bus_id],
            'messages': self.mailbox.copy()
        }

    def act(self, obs):
        return self.policy.forward(obs)

# agents/grid_agent.py
class GridCoordinatorAgent(Agent):
    """Manages a set of device agents."""

    def __init__(self, devices: List[DeviceAgent], protocol: Protocol):
        self.devices = devices
        self.protocol = protocol

    def step(self):
        # Collect observations
        obs = {dev.id: dev.observe(self.global_state) for dev in self.devices}

        # Run coordination protocol (e.g., price signal)
        setpoints = self.protocol.coordinate(obs)

        # Execute actions
        actions = {dev.id: dev.act(obs[dev.id], setpoints[dev.id])
                   for dev in self.devices}
        return actions
```

**Benefits**:
- Device agents can be trained independently (curriculum learning)
- Grid agents can use classical control (MPC) or learned policies (MAPPO)
- Clear separation of concerns (device physics vs. coordination logic)

---

#### 2. Communication Protocol Layer

**Location**: `powergrid/agents/protocols.py`

**Design Insight**: Protocols split into **two types** based on ownership and scope:

##### **Vertical Protocols** (Agent-Owned)

Parent → child coordination. Each agent owns its vertical protocol to coordinate its subordinates.

```python
# agents/protocols.py
class VerticalProtocol(ABC):
    """Parent → child coordination protocol (agent-owned)."""

    @abstractmethod
    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute coordination signals for subordinates."""

class PriceSignalProtocol(VerticalProtocol):
    """Parent broadcasts price to subordinates."""

    def coordinate(self, subordinate_observations, parent_action=None):
        # Update price from parent action
        price = parent_action if parent_action else self.default_price
        # Broadcast to all subordinates
        return {sub_id: {'price': price} for sub_id in subordinate_observations}

class SetpointProtocol(VerticalProtocol):
    """Parent assigns power setpoints to subordinates."""

    def coordinate(self, subordinate_observations, parent_action=None):
        # parent_action is dict of {subordinate_id: setpoint}
        return {sub_id: {'setpoint': parent_action[sub_id]}
                for sub_id in parent_action}
```

##### **Horizontal Protocols** (Environment-Owned)

Peer ↔ peer coordination. Environment owns and runs horizontal protocols as they require global view.

```python
class HorizontalProtocol(ABC):
    """Peer ↔ peer coordination protocol (environment-owned)."""

    @abstractmethod
    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Coordinate peer agents (requires global view)."""

class PeerToPeerTradingProtocol(HorizontalProtocol):
    """P2P energy trading marketplace."""

    def coordinate(self, agents, observations, topology=None):
        # Collect bids/offers from all agents
        bids, offers = self._collect_bids_offers(observations)
        # Clear market (environment acts as auctioneer)
        trades = self._clear_market(bids, offers)
        # Return trade confirmations
        return self._generate_trade_signals(trades)

class ConsensusProtocol(HorizontalProtocol):
    """Distributed consensus via gossip algorithm."""

    def coordinate(self, agents, observations, topology=None):
        # Iterative averaging until convergence
        values = {aid: obs.local['value'] for aid, obs in observations.items()}
        for _ in range(self.max_iters):
            values = self._average_with_neighbors(values, topology)
        return {aid: {'consensus_value': v} for aid, v in values.items()}
```

**Key Architectural Insight**:
- **Vertical protocols** are decentralized (each agent manages its own children)
- **Horizontal protocols** are centralized (environment provides marketplace/infrastructure)

**Use Cases**:
- **Vertical**: GridAgent → DeviceAgents (price signals, setpoints, reserve requirements)
- **Horizontal**: GridAgent ↔ GridAgent (P2P trading, frequency regulation, consensus)
- **Energy researchers**: Implement domain-specific protocols (droop control, volt-var, ADMM)
- **RL researchers**: Study communication-efficient MARL (gossip learning, federated RL)

---

#### 3. Multi-Agent Environment API

**Location**: `powergrid/envs/multi_agent/`

```python
# envs/multi_agent/base.py
from pettingzoo import ParallelEnv

class MultiAgentPowerGridEnv(ParallelEnv):
    """
    PettingZoo-compatible multi-agent environment.

    Supports:
    - Heterogeneous agents (different obs/action spaces)
    - Partial observability
    - Agent death/spawn (e.g., device failures)
    - Asynchronous actions (via action masking)
    """

    def __init__(self, config: Dict):
        self.topology = load_topology(config['network'])
        self.agents = self._build_agents(config['agents'])
        self.protocol = load_protocol(config['coordination'])
        self.scheduler = AsyncScheduler(config.get('async', False))

        # Standard PettingZoo attributes
        self.possible_agents = list(self.agents.keys())
        self.action_spaces = {a: agent.action_space
                             for a, agent in self.agents.items()}
        self.observation_spaces = {a: agent.observation_space
                                   for a, agent in self.agents.items()}

    def step(self, actions: Dict[str, Action]):
        """
        Execute one environment step.

        In async mode, only agents with action_mask=True act this step.
        """
        # 1. Set device actions
        for agent_id, action in actions.items():
            self.agents[agent_id].set_action(action)

        # 2. Update device states
        for agent in self.agents.values():
            agent.device.update_state()

        # 3. Push to pandapower network
        self._sync_to_pandapower()

        # 4. Solve power flow
        converged = self._solve_power_flow()

        # 5. Compute rewards
        rewards = self._compute_rewards(converged)

        # 6. Run coordination protocol
        self.protocol.step(self.agents)

        # 7. Get observations
        obs = {a: agent.observe(self.global_state)
               for a, agent in self.agents.items()}

        # 8. Check termination
        dones = self._check_done()
        infos = self._build_info(converged)

        return obs, rewards, dones, dones, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for agent in self.agents.values():
            agent.reset(self.np_random)
        obs = {a: agent.observe(self.global_state)
               for a, agent in self.agents.items()}
        return obs, {}
```

**Integration with RL Libraries**:

```python
# RLlib integration
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(MultiAgentPowerGridEnv, env_config={
        'network': 'ieee34',
        'coordination': 'price_signal'
    })
    .multi_agent(
        policies={'dg_policy', 'ess_policy'},
        policy_mapping_fn=lambda agent_id: f"{agent_id.split('_')[0]}_policy"
    )
)
algo = config.build()
algo.train()
```

---

#### 4. Plug-and-Play Configuration

**Location**: `configs/`, `powergrid/config/`

##### YAML Environment Definition

```yaml
# configs/ieee34_microgrid.yaml
name: IEEE34_Microgrid_Demo

# Network topology
network:
  base: ieee34  # Load from powergrid/networks/ieee34.py
  modifications:
    - add_bus: {name: "MG1_PCC", vn_kv: 12.47, at_bus: 800}

# Agent definitions
agents:
  # Device-level agents
  - id: ess_1
    type: ESS
    bus: 800
    params:
      capacity: 1.0  # MWh
      max_p_mw: 0.5
      soc_init: 0.5
    policy: learned  # or 'rule_based', 'mpc'

  - id: dg_1
    type: DG
    bus: 806
    params:
      p_range: [0, 0.5]
      fuel_cost: 50  # $/MWh
    policy: learned

  - id: solar_1
    type: RES
    bus: 808
    params:
      p_max: 0.3
      type: solar
    policy: mppt  # Max power point tracking

  # Grid-level coordinator (optional)
  - id: mg_controller
    type: GridCoordinator
    sub_agents: [ess_1, dg_1, solar_1]
    policy: centralized  # Single policy controls all sub-agents

# Coordination mechanism
coordination:
  protocol: price_signal
  params:
    update_freq: 900  # seconds (15 min)
    lmp_solver: dcopf

# Datasets
datasets:
  load: data/2024_load.csv
  solar: data/2024_solar.csv
  wind: data/2024_wind.csv
  price: data/caiso_lmp_2024.csv

# Simulation settings
simulation:
  episode_length: 96  # time steps (15-min intervals = 1 day)
  time_step: 900  # seconds
  async: false  # Synchronous for now

# Training settings
training:
  train: true
  reward_scale: 1.0
  safety_scale: 10.0
  max_penalty: 100.0
```

##### Python Loader

```python
# config/loader.py
class ConfigLoader:
    """Load environments from YAML configs."""

    @staticmethod
    def load(config_path: str) -> MultiAgentPowerGridEnv:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Build network
        net = load_network(cfg['network']['base'])
        apply_modifications(net, cfg['network'].get('modifications', []))

        # Build agents
        agents = {}
        for agent_cfg in cfg['agents']:
            agent = AgentFactory.create(
                agent_type=agent_cfg['type'],
                agent_id=agent_cfg['id'],
                **agent_cfg.get('params', {})
            )
            agents[agent_cfg['id']] = agent

        # Build environment
        env = MultiAgentPowerGridEnv(
            net=net,
            agents=agents,
            coordination=load_protocol(cfg['coordination']),
            datasets=load_datasets(cfg['datasets']),
            **cfg.get('simulation', {})
        )
        return env

# Usage
env = ConfigLoader.load("configs/ieee34_microgrid.yaml")
```

**Benefits**:
- Non-programmers define experiments via YAML
- Version control for reproducibility
- Rapid prototyping (change devices/protocols without code)

---

#### 5. Device Plugin System

**Location**: `powergrid/plugins/`, user-defined directories

```python
# plugins/registry.py
DEVICE_REGISTRY = {}

def register_device(name: str):
    """Decorator to register custom devices."""
    def decorator(cls):
        DEVICE_REGISTRY[name] = cls
        return cls
    return decorator

# User-defined plugin
# my_devices/hvac.py
from powergrid.agents import DeviceAgent
from powergrid.plugins import register_device

@register_device("HVAC")
class HVACAgent(DeviceAgent):
    """Heating/cooling system with thermal dynamics."""

    def __init__(self, temp_range=(18, 26), thermal_mass=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temp_range = temp_range
        self.thermal_mass = thermal_mass
        self.temperature = 22.0  # Initial temp
        self.set_action_space()

    def set_action_space(self):
        # Action: heating/cooling power (kW)
        self.action.dim_c = 1
        self.action.range = np.array([[-5.0], [5.0]])  # -5kW (cooling) to 5kW (heating)

    def update_state(self):
        # Thermal dynamics
        power = self.action.c[0]
        ambient = self.dataset['ambient_temp'][self.t]
        heat_loss = (self.temperature - ambient) / self.thermal_mass

        self.temperature += (power - heat_loss) * self.dt / 3600
        self.state.P = power / 1000  # Convert to MW

    def update_cost_safety(self):
        self.cost = abs(self.state.P) * self.electricity_price

        # Safety: temperature comfort violation
        if self.temperature < self.temp_range[0]:
            self.safety = self.temp_range[0] - self.temperature
        elif self.temperature > self.temp_range[1]:
            self.safety = self.temperature - self.temp_range[1]
        else:
            self.safety = 0.0

# Load custom devices
from powergrid.plugins import load_plugins
load_plugins("my_devices/")

# Now HVAC is available in YAML configs
# agents:
#   - id: hvac_1
#     type: HVAC
#     params:
#       temp_range: [20, 24]
```

**Benefits**:
- Energy researchers contribute domain-specific devices
- No changes to core codebase
- Community ecosystem of device models

---

#### 6. Dataset Management

**Location**: `powergrid/datasets/`

```python
# datasets/loaders.py
class DatasetRegistry:
    """Registry of pre-loaded datasets."""

    DATASETS = {
        'caiso_2024': {
            'load': 'https://oasis.caiso.com/load_2024.csv',
            'solar': 'https://oasis.caiso.com/solar_2024.csv',
            'price': 'https://oasis.caiso.com/lmp_2024.csv'
        },
        'ercot_2023': {...},
        'nyiso_2024': {...}
    }

    @staticmethod
    def load(name: str, cache_dir='~/.powergrid/data') -> Dict[str, np.ndarray]:
        """Download and cache dataset."""
        if name not in DatasetRegistry.DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

        dataset = {}
        for key, url in DatasetRegistry.DATASETS[name].items():
            cache_path = Path(cache_dir) / f"{name}_{key}.npy"
            if cache_path.exists():
                dataset[key] = np.load(cache_path)
            else:
                df = pd.read_csv(url)
                arr = preprocess(df)
                np.save(cache_path, arr)
                dataset[key] = arr
        return dataset

# datasets/preprocessors.py
class TimeseriesAligner:
    """Align multiple timeseries to common timestamps."""

    def align(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        # Find common time range
        start = max(df.index[0] for df in datasets.values())
        end = min(df.index[-1] for df in datasets.values())

        # Resample to common frequency
        aligned = {}
        for name, df in datasets.items():
            aligned[name] = df.loc[start:end].resample('15T').mean().values
        return aligned

# Usage in environment
env = MultiAgentPowerGridEnv(config={
    'network': 'ieee34',
    'dataset': 'caiso_2024'  # Auto-downloaded
})
```

**Pre-loaded Datasets**:
- **CAISO**: California ISO (2018-2024)
- **ERCOT**: Texas (2020-2024)
- **NYISO**: New York (2019-2024)
- **Pecan Street**: Residential solar/load (2013-2018)
- **NREL**: Synthetic distribution feeders

---

#### 7. Hierarchical Multi-Agent Architecture

**Example: Three-Level Control**

```python
# envs/hierarchical.py
class HierarchicalGridEnv(MultiAgentPowerGridEnv):
    """
    Level 3: System Operator (ISO)
        - Objective: Minimize total system cost
        - Actions: Set price signals, reserve requirements
        - Frequency: Hourly

    Level 2: Microgrid Controllers
        - Objective: Minimize local cost + follow ISO signals
        - Actions: Dispatch setpoints for devices
        - Frequency: 15 minutes

    Level 1: Device Agents
        - Objective: Track setpoints + local constraints
        - Actions: P/Q setpoints
        - Frequency: 1 minute
    """

    def __init__(self, config):
        super().__init__(config)

        # Build hierarchy
        self.device_agents = {a: agent for a, agent in self.agents.items()
                             if agent.level == 1}
        self.grid_agents = {a: agent for a, agent in self.agents.items()
                           if agent.level == 2}
        self.system_agent = next(a for a in self.agents.values() if a.level == 3)

    def step(self, actions):
        # Step 1: System agent acts (slow timescale)
        if self.t % self.system_freq == 0:
            system_action = actions[self.system_agent.id]
            price_signals = self.system_agent.act(system_action)

            # Broadcast to grid agents
            for grid_agent in self.grid_agents.values():
                grid_agent.receive_message({'prices': price_signals})

        # Step 2: Grid agents act (medium timescale)
        if self.t % self.grid_freq == 0:
            for grid_id, grid_agent in self.grid_agents.items():
                grid_action = actions[grid_id]
                setpoints = grid_agent.act(grid_action)

                # Send to subordinate devices
                for device_id in grid_agent.subordinates:
                    self.device_agents[device_id].receive_message({
                        'setpoint': setpoints[device_id]
                    })

        # Step 3: Device agents act (fast timescale)
        device_actions = {dev_id: actions[dev_id]
                         for dev_id in self.device_agents}

        # Execute standard step with device actions
        return super().step(device_actions)
```

**Research Applications**:
- **Feudal RL**: Options framework, temporal abstraction
- **Meta-learning**: Transfer policies across hierarchy levels
- **Decomposition**: Benders, ADMM with learned subproblems

---

## Implementation Roadmap

**Timeline**: 3 months | **Team Size**: 3 engineers

### Team Structure

| Role | Responsibilities | Focus Areas |
|------|------------------|-------------|
| **Architect** | Core infrastructure, API design, integration | Agent abstraction, multi-agent env, async scheduler |
| **Domain Engineer** | Networks, devices, protocols | Device plugins, network templates, coordination protocols |
| **DevOps/QA** | Testing, documentation, CI/CD | Unit tests, benchmarks, tutorials, deployment |

---

### Month 1: Foundation & Core API

#### Week 1-2: Agent Abstraction Layer

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Design `Agent` base class and hierarchy | `agents/base.py`, `agents/device_agent.py`, `agents/grid_agent.py` |
| Architect | Implement action/observation interfaces | Support for partial observability, heterogeneous spaces |
| Domain | Refactor existing devices to `DeviceAgent` | Convert DG, ESS, RES, Shunt, Grid to agents |
| DevOps | Set up test infrastructure | Pytest fixtures, device test templates |

**Deliverables**:
- ✅ `powergrid/agents/` module with base classes
- ✅ All existing devices work as agents
- ✅ Unit tests for agent lifecycle (reset, act, observe)

---

#### Week 3-4: Multi-Agent Environment API

**Architectural Decisions** (refined through detailed design):
- **GridAgents** are the primary RL-controllable agents (microgrid controllers)
- **DeviceAgents** are subordinates managed internally by GridAgents
- **Two protocol types**:
  - **Vertical protocols** (agent-owned): Parent → child coordination (e.g., GridAgent → DeviceAgents)
  - **Horizontal protocols** (environment-owned): Peer ↔ peer coordination (e.g., GridAgent ↔ GridAgent)
- **SystemAgent deferred** to Week 11-12 (not needed for core multi-agent functionality)

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Refactor protocol system | Split into `VerticalProtocol` and `HorizontalProtocol` base classes |
| Architect | Implement `MultiAgentPowerGridEnv` (PettingZoo) | `envs/multi_agent/pettingzoo_env.py` with protocol coordination |
| Architect | Update GridAgent for vertical protocols | Add `coordinate_subordinates()` method |
| Domain | Implement horizontal protocols | P2P trading, consensus protocols |
| Domain | Create 3 example environments | Simple 2-MG, P2P trading 3-MG, MultiAgentMicrogrids V2 |
| DevOps | Unit tests for protocols | Test vertical and horizontal coordination separately |
| DevOps | Integration tests with RLlib | Train MAPPO on 3-microgrid environment |

**Deliverables**:
- ✅ PettingZoo-compatible environment with GridAgents
- ✅ Vertical protocol system (PriceSignal, Setpoint, NoProtocol)
- ✅ Horizontal protocol system (P2P Trading, Consensus)
- ✅ 3 working example environments
- ✅ Integration tests with MAPPO training
- ✅ Backward compatibility with existing `MultiAgentMicrogrids`

**Milestone**: Train MAPPO on 3-microgrid environment with P2P trading, achieve convergence (reward > -50)

---

### Month 2: Configuration & Extensibility

#### Week 5-6: YAML Configuration System

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Design config schema | `config/schema.yaml`, validation logic |
| Architect | Implement `ConfigLoader` | `config/loader.py`, `AgentFactory` |
| Domain | Create 10+ example configs | Single-agent, multi-agent, hierarchical, various networks |
| DevOps | Write config validator and linter | CLI tool: `powergrid validate config.yaml` |
| DevOps | Configuration documentation | Tutorial: "Build Env from YAML" |

**Deliverables**:
- ✅ Full YAML-based environment creation
- ✅ 10+ ready-to-use configs
- ✅ Config validator tool

---

#### Week 7-8: Device Plugin System & Datasets

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Implement plugin registry | `plugins/registry.py`, auto-discovery |
| Domain | Create 5+ new device types | HVAC, EV charger, electrolyzer, fuel cell, wind turbine |
| Domain | Build dataset registry | `datasets/loaders.py`, CAISO/ERCOT/NYISO integration |
| Domain | Dataset preprocessing pipeline | Alignment, interpolation, missing data handling |
| DevOps | Plugin development guide | Tutorial: "Create Custom Device in 15 Minutes" |
| DevOps | Dataset documentation | Available datasets, how to add custom data |

**Deliverables**:
- ✅ Plugin system with 5+ example devices
- ✅ 3+ real-world datasets (CAISO, ERCOT, NYISO)
- ✅ Plugin development tutorial

**Milestone**: External user creates custom device plugin and contributes it

---

### Month 3: Advanced Features & Polish

#### Week 9-10: Advanced Protocols & Network Templates

**Note**: Basic vertical/horizontal protocols implemented in Week 3-4. This week adds advanced features.

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Implement async message passing | Time-delayed, lossy communication channels |
| Domain | Build advanced coordination protocols | ADMM, droop control, volt-var, federated learning |
| Domain | Network templates library | 10+ networks (IEEE 13/34/123, CIGRE MV/LV, Pecan Street) |
| DevOps | Protocol benchmarks | Compare centralized vs. decentralized vs. peer-to-peer |

**Deliverables**:
- ✅ Async message passing with delays/losses
- ✅ 5+ advanced coordination protocols
- ✅ 10+ network templates
- ✅ Benchmark comparison study

---

#### Week 11-12: Three-Level Hierarchy & Finalization

**Focus**: Add SystemAgent (Level 3) for ISO/market operator control.

| Owner | Task | Deliverables |
|-------|------|-------------|
| Architect | Implement `SystemAgent` class | `agents/system_agent.py`, Level 3 agent for ISO operations |
| Architect | Extend environment for 3-level hierarchy | Support SystemAgent → GridAgent → DeviceAgent |
| Architect | Multi-rate scheduler | GridAgents and SystemAgent act at different frequencies |
| Domain | Create 3-level hierarchical examples | ISO → Microgrid Controllers → Devices |
| Domain | LMP-based market clearing protocol | SystemAgent computes LMP, GridAgents respond to prices |
| Domain | Safety constraint wrappers | PPO-Lagrangian, CPO integration |
| DevOps | Complete documentation | API reference, 10+ tutorials, deployment guide |
| DevOps | Benchmark suite | 5 standard tasks (voltage, dispatch, peak shaving, etc.) |
| DevOps | CI/CD pipeline | GitHub Actions, auto-tests, coverage >80% |

**Deliverables**:
- ✅ SystemAgent (Level 3) implementation
- ✅ Three-level hierarchical control examples
- ✅ Multi-rate execution (ISO/GridAgent/Device at different frequencies)
- ✅ LMP-based coordination protocol
- ✅ Complete documentation (100+ pages)
- ✅ Benchmark suite with leaderboard
- ✅ Production-ready v2.0 release

**Milestone**: Public release with blog post, demo videos, research paper

---

## Success Criteria (End of Month 3)

### Technical Metrics

- ✅ **Performance**: Train 100 agents at >10 steps/sec (10x improvement)
- ✅ **Coverage**: >80% code coverage, all critical paths tested
- ✅ **Compatibility**: Works with RLlib, SB3, CleanRL, Tianshou
- ✅ **Scalability**: Support 100+ agents in single environment
- ✅ **API Stability**: Frozen API for v2.0, backward compatible with v1.0

### User Adoption Metrics

- ✅ **Documentation**: 10+ tutorials, 100+ pages of docs
- ✅ **Examples**: 20+ working examples (configs + notebooks)
- ✅ **Datasets**: 3+ real-world datasets pre-loaded
- ✅ **Devices**: 15+ device types (10 built-in + 5 plugin examples)
- ✅ **Networks**: 10+ network templates

### Community Metrics

- 🎯 **Alpha testers**: 5+ external researchers using beta
- 🎯 **Papers**: 2+ research papers in progress using v2.0
- 🎯 **Contributions**: 3+ community PRs (devices, configs, docs)

---

## Weekly Sync & Risk Management

### Weekly Cadence

**Monday**: Sprint planning, task assignment
**Wednesday**: Mid-week checkpoint, blocker resolution
**Friday**: Demo + retrospective

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PettingZoo API complexity | Medium | High | Start with simple parallel env, defer AEC if needed |
| Dataset integration delays | Medium | Medium | Use synthetic data as fallback, defer real data to Week 7 |
| Performance bottlenecks | Low | High | Profile early (Week 4), optimize critical paths only |
| Scope creep | High | High | **Strict scope freeze after Week 4**, defer extras to v2.1 |

### Deferred to v2.1 (Post-3 Month)

- Advanced async simulation (event-driven SimPy)
- Hardware-in-the-loop (HIL) support
- ONNX model export for edge deployment
- Cloud deployment (Docker, K8s)
- GraphQL API for web integration

---

## Resource Requirements

### Compute

- 1x GPU workstation (training benchmarks)
- 1x CPU server (CI/CD, testing)
- Cloud credits: $500/month (Ray cluster, large-scale tests)

### External Dependencies

- Dataset access: CAISO OASIS, ERCOT, NYISO APIs
- Test networks: IEEE, CIGRE public models

---

## Contingency Plan

If behind schedule at Month 2:

**Priority 1 (Must Have)**:
- Agent abstraction + PettingZoo API
- YAML configs
- 5+ example environments

**Priority 2 (Should Have)**:
- Plugin system (can ship with 2-3 examples instead of 5)
- Datasets (use synthetic if real data delayed)

**Priority 3 (Nice to Have)**:
- Communication protocols (defer to v2.1)
- Hierarchical envs (defer to v2.1)
- Advanced benchmarks (ship basic version)

---

## Migration Strategy

### Backward Compatibility

**Existing code continues working**:

```python
# Old API (still supported)
from powergrid.base_env import GridBaseEnv
env = GridBaseEnv(...)
```

**Opt-in to new features**:

```python
# New API
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
env = MultiAgentPowerGridEnv.from_config("configs/my_env.yaml")

# Or convert existing env
from powergrid.migration import to_multi_agent
ma_env = to_multi_agent(old_env, split_by='device')
```

### Deprecation Timeline

- **v2.0**: Introduce multi-agent API, mark `NetworkedGridEnv` as legacy
- **v2.1-2.5**: Both APIs supported, encourage migration
- **v3.0**: Deprecate old `NetworkedGridEnv`, keep `GridBaseEnv` for single-agent

---

## Documentation Plan

### For Energy Researchers

**Quickstart**: "Build a Custom Device in 10 Minutes"
```
1. Copy device template
2. Implement physics in update_state()
3. Define cost/safety metrics
4. Register with @register_device
5. Use in YAML config
```

**Tutorials**:
- Custom device: HVAC, EV charger, electrolyzer
- Custom protocols: Droop control, volt-var curve
- Dataset integration: Load custom CSV files

---

### For RL Researchers

**Quickstart**: "Train MAPPO on IEEE 34-Bus"
```
1. Install: pip install powergrid[rl]
2. Load config: env = PowerGridEnv.from_config('ieee34')
3. Train: ray.rllib.train(PPO, env)
4. Evaluate: rollout trained_policy.zip
```

**Tutorials**:
- Baseline algorithms: PPO, SAC, MAPPO
- Safe RL: CPO, PPO-Lagrangian
- Hierarchical: Feudal RL, options
- Communication: Graph neural networks

---

## Technical Specifications

### Performance Requirements

| Metric | Target | Current |
|--------|--------|---------|
| Steps/sec (10 agents) | >100 | ~50 |
| Steps/sec (100 agents) | >10 | N/A |
| Memory per agent | <10 MB | ~5 MB |
| Training time (1M steps) | <1 hour (GPU) | ~2 hours |

### Supported Platforms

- **OS**: Linux, macOS, Windows
- **Python**: 3.9-3.12
- **Accelerators**: CPU, CUDA, MPS (Apple Silicon)
- **Distributed**: Ray, Dask

### Dependencies

**Core**:
- gymnasium >= 0.29
- pandapower >= 2.14
- numpy >= 1.24
- pettingzoo >= 1.24

**RL (optional)**:
- ray[rllib] >= 2.9
- stable-baselines3 >= 2.3
- tianshou >= 1.0

**Deployment (optional)**:
- onnx >= 1.15
- pymodbus >= 3.5

---

## Open Questions

### Design Decisions

1. **Action Space for Hierarchical Agents**:
   - Option A: Grid agents output setpoints, devices track them
   - Option B: Grid agents output rewards/penalties, devices optimize locally
   - **Recommendation**: Support both via `HierarchicalProtocol` interface

2. **Async Simulation**:
   - Option A: Event-driven (SimPy-style)
   - Option B: Multi-rate synchronous (sub-step devices at different frequencies)
   - **Recommendation**: Start with B (simpler), add A in Phase 3

3. **Observation Space**:
   - Option A: Full observability (all agent states)
   - Option B: Local + communication (realistic)
   - **Recommendation**: Support both, default to B

### Research Questions

1. How to handle **non-stationarity** in multi-agent credit assignment?
2. Optimal **communication topology** (fully-connected vs. neighbors)?
3. **Scalability** of learned policies to different grid sizes?

---

## Success Criteria

### Adoption Metrics (Year 1)

- **100+** GitHub stars
- **10+** research papers using PowerGrid 2.0
- **5+** community device contributions
- **3+** courses/tutorials using the platform

### Technical Metrics

- **Training speed**: 2x faster than v1.0
- **Agent scalability**: Support 100+ agents
- **API stability**: <5 breaking changes per year

---

## Conclusion

This proposal transforms PowerGrid into a **flexible, scalable platform** for multi-agent power systems research while maintaining the simplicity that makes it accessible to energy researchers. The phased approach ensures we deliver value early (Phase 1 MVP) while building toward ambitious goals (hierarchical control, HIL deployment).

**Next Steps**:
1. Review and approve design
2. Create GitHub project board
3. Begin Phase 1 implementation
4. Recruit early adopters for beta testing

---

## Appendices

### A. Component Diagram


### B. Example Configs

See `configs/examples/` for:
- `ieee13_single.yaml`: Single-agent baseline
- `ieee34_multi.yaml`: Multi-agent with 5 devices
- `hierarchical_3level.yaml`: Three-level control

### C. API Reference

See `docs/api/` for detailed API documentation (generated from docstrings).

---

## Architectural Design Summary

### Refined Decisions (Post Week 3-4 Design)

**1. Agent Hierarchy**
- **Level 1 (DeviceAgent)**: Individual DERs managed by GridAgents
- **Level 2 (GridAgent)**: RL-controllable microgrid controllers (primary agents)
- **Level 3 (SystemAgent)**: ISO/market operator (deferred to Week 11-12)

**2. Protocol System**
- **Vertical Protocols** (agent-owned, decentralized):
  - Each agent owns its vertical protocol to coordinate subordinates
  - Examples: PriceSignalProtocol, SetpointProtocol
  - Used for: GridAgent → DeviceAgents

- **Horizontal Protocols** (environment-owned, centralized):
  - Environment runs horizontal protocols as they need global view
  - Examples: PeerToPeerTradingProtocol, ConsensusProtocol
  - Used for: GridAgent ↔ GridAgent peer coordination

**3. No "Hierarchical" Protocol Type**
- Hierarchical control = vertical protocols applied recursively at multiple levels
- SystemAgent uses vertical protocol → GridAgents
- GridAgent uses vertical protocol → DeviceAgents
- No separate protocol type needed

**4. Extensibility**
- Level-based architecture (not type-checking)
- Any future agent types inherit from `Agent` base class
- Protocol system works with any agent regardless of type
- Examples of future agents: VehicleAgent, LoadAgent, FeederAgent

### Implementation Philosophy

- **Open-Closed Principle**: Open for extension (new agent types), closed for modification (existing protocols work)
- **Separation of Concerns**: Vertical (agent-owned) vs. Horizontal (environment-owned)
- **Backward Compatibility**: Existing code continues working, new features opt-in
- **Phased Delivery**: Core functionality first (Week 3-4), advanced features later (Week 11-12)

---

**Document Version**: 1.1
**Last Updated**: 2025-10-19
**Authors**: PowerGrid Development Team
**Status**: Design Refined (Week 3-4 Ready for Implementation)
