# Agent Abstraction Layer

This module provides the core agent abstractions for hierarchical multi-agent control in PowerGrid.

## Architecture Overview

The agent abstraction layer enables:
- **Hierarchical Control**: Device-level, grid-level, and system-level agents
- **Agent Autonomy**: Each agent has independent observation, action, and decision-making
- **Communication**: Inter-agent message passing for coordination
- **Flexible Policies**: Plug-and-play support for learned (RL) and rule-based policies

## Core Components

### 1. Agent Base Class (`base.py`)

Abstract base class for all agents:

```python
from powergrid.agents import Agent, Observation

class MyAgent(Agent):
    def observe(self, global_state):
        # Extract relevant observations
        obs = Observation()
        obs.local["my_value"] = global_state["data"]
        return obs

    def act(self, observation):
        # Compute action
        return self.action_space.sample()

    def reset(self, *, seed=None):
        super().reset(seed=seed)
```

**Key Features**:
- `observe()`: Extract agent-specific observations from global state
- `act()`: Compute actions based on observations
- `receive_message()` / `send_message()`: Inter-agent communication
- `reset()`: Initialize agent state

### 2. DeviceAgent (`device_agent.py`)

Wraps existing `Device` objects as autonomous agents:

```python
from powergrid.agents import DeviceAgent
from powergrid.devices.storage import ESS

# Create device
ess = ESS(
    name="ess_1",
    bus=800,
    min_p_mw=-0.5,
    max_p_mw=0.5,
    capacity=1.0,
)

# Wrap as agent
agent = DeviceAgent(
    device=ess,
    partial_obs=False,  # Full observability
)

# Use in environment
global_state = {"bus_vm": {800: 1.05}, "price": 50.0}
obs = agent.observe(global_state)
action = agent.act(obs)
```

**Key Features**:
- Automatic action/observation space construction from device
- Support for partial observability (local state only)
- Compatible with existing `Device` interface
- Pluggable policies (learned or rule-based)

### 3. GridCoordinatorAgent (`grid_agent.py`)

Manages a set of subordinate device agents with coordination protocols:

```python
from powergrid.agents import GridCoordinatorAgent, PriceSignalProtocol

# Create subordinate agents
sub_agents = [agent1, agent2, agent3]

# Create coordinator with price-based coordination
coordinator = GridCoordinatorAgent(
    agent_id="mg_controller",
    subordinates=sub_agents,
    protocol=PriceSignalProtocol(initial_price=50.0),
)

# Coordinator observes and acts
obs = coordinator.observe(global_state)
coordinator.act(obs)  # Broadcasts price to subordinates
```

**Coordination Protocols**:
- `NoProtocol`: Independent agent operation
- `PriceSignalProtocol`: Broadcast marginal prices
- `SetpointProtocol`: Distribute power setpoints
- Custom protocols via `Protocol` interface

### 4. Observation & Message (`base.py`)

Structured data classes for observations and communication:

```python
from powergrid.agents import Observation, Message

# Observation
obs = Observation(
    local={"P": 1.0, "Q": 0.5, "soc": 0.7},
    global_info={"bus_voltage": 1.05, "price": 50.0},
    messages=[],
    timestamp=10.0,
)

# Convert to flat vector for RL
vec = obs.as_vector()  # np.ndarray

# Message
msg = Message(
    sender="agent_1",
    content={"price": 75.0, "setpoint": 0.5},
    timestamp=10.0,
    priority=1,
)
```

### 5. Space Utilities (`spaces.py`)

Helpers for multi-agent action/observation spaces:

```python
from powergrid.agents import compose_action_spaces, ActionMask

# Compose heterogeneous action spaces
spaces = {
    "ess_1": Box(low=-0.5, high=0.5, shape=(1,)),
    "dg_1": Box(low=0, high=1, shape=(2,)),
}
combined = compose_action_spaces(spaces, mode="dict")

# Action masking for asynchronous execution
mask = ActionMask(
    agent_ids=["ess_1", "dg_1", "solar_1"],
    frequencies={"ess_1": 1, "dg_1": 4, "solar_1": 1},  # dg acts every 4 steps
)
active = mask.get_active_agents(timestep=0)  # ["ess_1", "solar_1"]
```

## Usage Examples

### Example 1: Single DeviceAgent

```python
from powergrid.agents import DeviceAgent
from powergrid.devices.storage import ESS

# Create and wrap device
ess = ESS(name="ess_1", bus=800, min_p_mw=-0.5, max_p_mw=0.5, capacity=1.0)
agent = DeviceAgent(device=ess)

# Reset
agent.reset(seed=42)

# Step
global_state = {"bus_vm": {800: 1.05}, "dataset": {"price": 50.0}}
obs = agent.observe(global_state)
action = agent.act(obs)

# Get reward
reward = agent.get_reward()  # -cost - safety
```

### Example 2: Hierarchical Coordination

```python
from powergrid.agents import DeviceAgent, GridCoordinatorAgent, PriceSignalProtocol
from powergrid.devices.storage import ESS
from powergrid.devices.generator import DG

# Create device agents
ess = ESS(name="ess_1", bus=800, min_p_mw=-0.5, max_p_mw=0.5, capacity=1.0)
dg = DG(name="dg_1", bus="806", min_p_mw=0, max_p_mw=0.5)

ess_agent = DeviceAgent(device=ess)
dg_agent = DeviceAgent(device=dg)

# Create grid coordinator
coordinator = GridCoordinatorAgent(
    agent_id="mg_controller",
    subordinates=[ess_agent, dg_agent],
    protocol=PriceSignalProtocol(initial_price=50.0),
)

# Reset
coordinator.reset(seed=42)

# Step
global_state = {"bus_vm": {800: 1.05, "806": 1.03}, "converged": True}
obs = coordinator.observe(global_state)
coordinator.act(obs)  # Broadcasts price to ess_agent and dg_agent

# Subordinates receive messages and act
for agent in [ess_agent, dg_agent]:
    agent_obs = agent.observe(global_state)
    agent.act(agent_obs)
```

### Example 3: Custom Policy

```python
from powergrid.agents import DeviceAgent, Policy

class MPCPolicy(Policy):
    """Model Predictive Control policy."""

    def forward(self, observation):
        # Solve optimization problem
        soc = observation.local["soc"]
        price = observation.global_info["price"]

        # Simple heuristic: charge when price < 40, discharge when > 60
        if price < 40:
            return np.array([0.5])  # Charge
        elif price > 60:
            return np.array([-0.5])  # Discharge
        else:
            return np.array([0.0])  # Hold

    def reset(self):
        pass

# Use with DeviceAgent
ess = ESS(name="ess_1", bus=800, min_p_mw=-0.5, max_p_mw=0.5, capacity=1.0)
agent = DeviceAgent(device=ess, policy=MPCPolicy())
```

## Testing

Run the test suite:

```bash
pytest powergrid/test/agents/ -v
```

Individual test files:
- `test_base_agent.py`: Tests for base Agent, Observation, Message
- `test_device_agent.py`: Tests for DeviceAgent wrapper
- `test_grid_agent.py`: Tests for GridCoordinatorAgent and protocols

## Design Principles

1. **Separation of Concerns**: Devices handle physics, agents handle decision-making
2. **Backward Compatibility**: DeviceAgent wraps existing Device objects
3. **Extensibility**: Easy to add new agent types, protocols, policies
4. **Composability**: Agents can be nested in hierarchies
5. **Standard Interfaces**: Uses Gymnasium spaces for RL compatibility

## Next Steps

For multi-agent environment integration, see:
- `powergrid/envs/multi_agent/` (Week 3-4 implementation)
- PettingZoo compatibility layer
- RLlib/SB3 integration examples

## References

- [Proposal Document](../../../docs/design/proposal.md)
- [Architecture Diagrams](../../../docs/design/architecture_diagrams.md)
- Week 1-2 Deliverables: Agent Abstraction Layer ✅
