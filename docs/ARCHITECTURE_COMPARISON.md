# Architecture Comparison: Old vs New Multi-Agent System

## Overview

This document compares the **OLD** architecture (`multiagent/base.py`, `NetworkedGridEnv`) with the **NEW** architecture (`multi_agent/pettingzoo_env.py`, `MultiAgentPowerGridEnv`) to verify feature parity before deprecation.

---

## OLD ARCHITECTURE

### Files
- `powergrid/envs/multiagent/base.py`
  - `GridEnv`: Microgrid wrapper (NOT an agent)
  - `NetworkedGridEnv`: RLlib MultiAgentEnv base class
- `powergrid/envs/multiagent/ieee34_ieee13.py`
  - `MultiAgentMicrogrids(NetworkedGridEnv)`: Old implementation (already deprecated)
  - `MultiAgentMicrogridsV2()`: Factory function using NEW architecture

### Key Classes

#### 1. `GridEnv` (OLD)
**Purpose**: Wrapper around a pandapower network with devices

**Key Features**:
- Wraps a pandapower network
- Manages devices (ESS, DG) via dictionaries
- Merges networks via `add_to()`
- Updates pandapower state from device states
- Computes cost/safety aggregation
- **NOT a reinforcement learning agent**

**Methods**:
```python
- __init__(net, **kwargs)
- add_dataset(dataset)
- add_to(ext_net, bus_name)          # Merge networks
- add_sgen(sgens)                     # Add generators
- add_storage(storages)               # Add storage
- step(net, action, t)                # Update devices
- reset(net, t, rng)                  # Reset devices
- _get_obs(net)                       # Get observation
- _set_action(action)                 # Dispatch action to devices
- _update_state(net, t)               # Apply dataset + update pandapower
- _update_cost_safety(net)            # Compute cost + safety violations
```

#### 2. `NetworkedGridEnv` (OLD)
**Purpose**: RLlib MultiAgentEnv for networked microgrids

**Key Features**:
- Inherits from `ray.rllib.env.multi_agent_env.MultiAgentEnv`
- Uses `GridEnv` instances as "agents" (but they're not real agents!)
- Central pandapower network simulation
- Reward sharing option
- Runs power flow centrally

**Methods**:
```python
- __init__(env_config)
- _build_net()                        # Abstract: build network + agents
- _reward_and_safety()                # Abstract: compute rewards
- step(action_n)                      # Run power flow + compute rewards
- reset(seed, options)                # Reset all GridEnvs
- _get_obs()                          # Collect observations from GridEnvs
- _init_space()                       # Build action/obs spaces from GridEnvs
```

**Example Usage** (OLD):
```python
class MultiAgentMicrogrids(NetworkedGridEnv):
    def _build_net(self):
        # Create GridEnv instances
        mg1 = GridEnv(IEEE13Bus('MG1'), ...)
        mg1.add_storage(ESS(...))
        mg1.add_sgen([DG(...), DG(...)])

        # Merge into single network
        net = mg1.add_to(main_net, 'Bus 822')

        self.net = net
        self.possible_agents = {'MG1': mg1, 'MG2': mg2, 'MG3': mg3}
```

---

## NEW ARCHITECTURE

### Files
- `powergrid/envs/multi_agent/pettingzoo_env.py`
  - `MultiAgentPowerGridEnv`: PettingZoo ParallelEnv
- `powergrid/agents/`
  - `DeviceAgent`: RL agent wrapping a device
  - `GridAgent`: Hierarchical coordinator managing DeviceAgents
  - Protocols: `VerticalProtocol`, `HorizontalProtocol`

### Key Classes

#### 1. `DeviceAgent` (NEW)
**Purpose**: RL agent wrapper around a single device

**Key Features**:
- Wraps a single device (ESS, DG, etc.)
- Has `action_space` and `observation_space`
- Receives messages (e.g., price signals)
- Acts based on policy or heuristic
- **Actual RL agent** with proper agent interface

**Methods**:
```python
- __init__(device, partial_obs, policy)
- observe(global_state)               # Get observation
- act(observation)                    # Compute action
- reset(seed)                         # Reset device
```

#### 2. `GridAgent` (NEW)
**Purpose**: Hierarchical coordinator managing multiple DeviceAgents

**Key Features**:
- Manages subordinate DeviceAgents
- Implements vertical coordination protocols
- Centralized action space (concatenated subordinates)
- Message-based coordination
- **Actual RL agent** (the primary controllable agent)

**Methods**:
```python
- __init__(agent_id, subordinates, vertical_protocol, centralized)
- observe(global_state)               # Aggregate subordinate observations
- act(observation)                    # Coordinate subordinates
- coordinate_subordinates()           # Run vertical protocol
- reset(seed)                         # Reset all subordinates
```

#### 3. `MultiAgentPowerGridEnv` (NEW)
**Purpose**: PettingZoo-based multi-agent environment

**Key Features**:
- Inherits from `pettingzoo.ParallelEnv`
- Uses `GridAgent` as primary agents
- Supports vertical protocols (agent-owned)
- Supports horizontal protocols (environment-owned)
- Modular, extensible architecture

**Methods**:
```python
- __init__(config)
- reset(seed, options)
- step(actions)                       # Multi-phase coordination
- _build_agents(microgrids)           # Create GridAgents from config
- _sync_to_pandapower()               # Update pandapower from device states
- _solve_power_flow()                 # Run power flow
- _update_cost_safety(converged)      # Compute costs + safety
- _compute_rewards(converged)         # Compute rewards with penalties
```

**Multi-Phase Step Execution**:
```python
1. Set actions on GridAgents
2. Vertical coordination (GridAgent → DeviceAgents)
3. Horizontal coordination (GridAgent ↔ GridAgent)
4. Update device states
5. Sync to pandapower
6. Solve power flow
7. Compute costs/safety/rewards
8. Get observations
```

**Example Usage** (NEW):
```python
config = {
    'microgrids': [
        {
            'name': 'MG1',
            'network': IEEE13Bus('MG1'),
            'devices': [ESS(...), DG(...), DG(...)],
            'vertical_protocol': 'price_signal',
            'dataset': {...}
        },
        ...
    ],
    'horizontal_protocol': 'p2p_trading',
    'episode_length': 24,
}

env = MultiAgentPowerGridEnv(config)
obs, info = env.reset()
obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## FEATURE COMPARISON

| Feature | OLD (GridEnv/NetworkedGridEnv) | NEW (GridAgent/MultiAgentPowerGridEnv) | Status |
|---------|-------------------------------|---------------------------------------|--------|
| **Device Management** | `GridEnv` dict of devices | `GridAgent` with `DeviceAgent` subordinates | ✅ Improved |
| **Network Merging** | `GridEnv.add_to()` merges networks | Merged in `MultiAgentPowerGridEnv.__init__()` | ✅ Equivalent |
| **Power Flow** | Central `pp.runpp()` in `NetworkedGridEnv.step()` | Central `pp.runpp()` in `MultiAgentPowerGridEnv._solve_power_flow()` | ✅ Equivalent |
| **Cost/Safety Computation** | `GridEnv._update_cost_safety()` | Device-level via `device.update_cost_safety()`, aggregated in env | ✅ Equivalent |
| **Reward Calculation** | Abstract `_reward_and_safety()` | `_compute_rewards()` with penalties | ✅ Improved |
| **Reward Sharing** | Supported via `share_reward` flag | Supported via `share_reward` flag | ✅ Equivalent |
| **Dataset Integration** | `GridEnv.add_dataset()` | Config dict `'dataset': {...}` | ✅ Equivalent |
| **Action Space** | Concatenated device actions in `GridEnv` | Concatenated `DeviceAgent` actions in `GridAgent` | ✅ Equivalent |
| **Observation Space** | `GridEnv._get_obs()` | `GridAgent.observe()` aggregates subordinates | ✅ Improved |
| **Coordination Protocols** | ❌ Not supported | ✅ Vertical + Horizontal protocols | ✅ NEW FEATURE |
| **Message Passing** | ❌ Not supported | ✅ Message-based coordination | ✅ NEW FEATURE |
| **Hierarchical Agents** | ❌ `GridEnv` is NOT an agent | ✅ `GridAgent` → `DeviceAgent` hierarchy | ✅ NEW FEATURE |
| **PettingZoo Compatibility** | ❌ RLlib only | ✅ PettingZoo ParallelEnv | ✅ Improved |
| **Modularity** | Tightly coupled | Highly modular (agents, protocols, env) | ✅ Improved |

---

## LOGIC MAPPING

### Network Building

**OLD**:
```python
class MultiAgentMicrogrids(NetworkedGridEnv):
    def _build_net(self):
        net = IEEE34Bus('DSO')
        dso = GridEnv(net, ...)

        mg1 = GridEnv(IEEE13Bus('MG1'), ...)
        mg1.add_storage(ESS(...))
        mg1.add_sgen([DG(...), DG(...)])
        net = mg1.add_to(net, 'DSO Bus 822')

        self.net = net
        self.possible_agents = {'MG1': mg1, 'MG2': mg2}
```

**NEW**:
```python
config = {
    'network': IEEE34Bus('DSO'),
    'microgrids': [
        {
            'name': 'MG1',
            'network': IEEE13Bus('MG1'),
            'devices': [ESS(...), DG(...), DG(...)],
            'connection_bus': 'DSO Bus 822',  # Implicit merge
        }
    ],
}
env = MultiAgentPowerGridEnv(config)
# Network merging happens in __init__ via pp.merge_nets()
```

**Equivalent**: Both merge networks and create agents. NEW version uses declarative config.

---

### Step Execution

**OLD**:
```python
def step(self, action_n):
    # 1. Set actions on GridEnv "agents"
    for name, action in action_n.items():
        self.actionable_agents[name].step(self.net, action, self._t)

    # 2. Run power flow
    pp.runpp(self.net)

    # 3. Update cost/safety
    for agent in self.agents.values():
        agent._update_cost_safety(self.net)

    # 4. Compute rewards
    rewards, safety = self._reward_and_safety()

    # 5. Get observations
    return self._get_obs(), rewards, ...
```

**NEW**:
```python
def step(self, actions):
    # 1. Set actions on GridAgents → DeviceAgents
    for aid, action in actions.items():
        self._set_subordinate_actions(self.agents[aid], action)

    # 2. Vertical coordination (optional)
    for agent in self.agents.values():
        agent.coordinate_subordinates(global_state)

    # 3. Horizontal coordination (optional)
    self.horizontal_protocol.coordinate(self.agents, global_state)

    # 4. Update device states
    self._update_device_states()

    # 5. Sync to pandapower
    self._sync_to_pandapower()

    # 6. Solve power flow
    converged = self._solve_power_flow()

    # 7. Update cost/safety
    self._update_cost_safety(converged)

    # 8. Compute rewards
    rewards = self._compute_rewards(converged)

    # 9. Get observations
    obs = self._get_observations()

    return obs, rewards, ...
```

**Equivalent**: Both execute the same core logic (set actions → power flow → cost → rewards → obs). NEW version adds coordination phases.

---

### Cost/Safety Computation

**OLD**:
```python
# In GridEnv
def _update_cost_safety(self, net):
    self.cost, self.safety = 0, 0
    for ess in self.storage.values():
        ess.update_cost_safety()
        self.cost += ess.cost
        self.safety += ess.safety

    for dg in self.sgen.values():
        dg.update_cost_safety()
        self.cost += dg.cost
        self.safety += dg.safety

    # Add voltage/line violations
    if net["converged"]:
        local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
        overvoltage = np.maximum(local_vm - 1.05, 0).sum()
        undervoltage = np.maximum(0.95 - local_vm, 0).sum()
        ...
        self.safety += overloading + overvoltage + undervoltage
```

**NEW**:
```python
# In MultiAgentPowerGridEnv
def _update_cost_safety(self, converged):
    for agent in self.agents.values():
        total_cost = 0
        total_safety = 0

        for sub_agent in agent.subordinates.values():
            device = sub_agent.device
            if hasattr(device, 'update_cost_safety'):
                device.update_cost_safety()

            total_cost += getattr(device, 'cost', 0)
            total_safety += getattr(device, 'safety', 0)

        agent.cost = total_cost
        agent.safety = total_safety
```

**Equivalent**: Both aggregate device costs/safety. OLD includes voltage violations (NEW could add this if needed).

---

### Reward Computation

**OLD**:
```python
def _reward_and_safety(self):
    if self.net["converged"]:
        rewards = {n: -a.cost for n, a in self.agents.items()}
        safety = {n: a.safety for n, a in self.agents.items()}
    else:
        rewards = {n: -200.0 for n in self.agents}
        safety = {n: 20 for n in self.agents}

    if self.env_config.get('penalty'):
        for name in self.agents:
            rewards[name] -= safety[name] * self.env_config.get('penalty')

    return rewards, safety
```

**NEW**:
```python
def _compute_rewards(self, converged):
    rewards = {}

    for aid, agent in self.agents.items():
        reward = -agent.cost
        reward -= self.penalty * agent.safety

        if not converged:
            reward -= self.penalty * 10

        rewards[aid] = reward

    if self.share_reward:
        avg_reward = sum(rewards.values()) / len(rewards)
        rewards = {aid: avg_reward for aid in rewards}

    return rewards
```

**Equivalent**: Both compute reward = -cost - penalty*safety, with convergence penalties.

---

## MISSING FEATURES IN NEW ARCHITECTURE

### ❌ Voltage/Line Constraint Safety Violations

**OLD** includes voltage and line loading violations in safety:
```python
overvoltage = np.maximum(local_vm - 1.05, 0).sum()
undervoltage = np.maximum(0.95 - local_vm, 0).sum()
overloading = np.maximum(local_line_loading - 100, 0).sum() * 0.01
self.safety += overloading + overvoltage + undervoltage
```

**NEW**: Only aggregates device-level safety (from `device.update_cost_safety()`).

**ACTION REQUIRED**: Add grid-level constraint violations to `MultiAgentPowerGridEnv._update_cost_safety()`.

---

## CONCLUSION

✅ **The NEW architecture implements all core functionality from the OLD architecture**:
- Network merging
- Device management
- Power flow simulation
- Cost/safety aggregation
- Reward computation
- Observation collection
- Action dispatching

✅ **The NEW architecture IMPROVES upon the OLD**:
- True RL agents (`GridAgent`, `DeviceAgent`) vs pseudo-agents (`GridEnv`)
- Hierarchical agent structure
- Coordination protocols (vertical + horizontal)
- Message passing
- PettingZoo compatibility
- Better modularity

❌ **Missing Feature**: Grid-level constraint violations (voltage, line loading) not included in safety.

**RECOMMENDATION**:
1. ✅ Deprecate `multiagent/base.py` and `NetworkedGridEnv`
2. ✅ Keep `MultiAgentMicrogrids` class with deprecation warning (already done)
3. ⚠️ Add grid-level safety violations to `MultiAgentPowerGridEnv._update_cost_safety()`
4. ✅ Update all code to use `MultiAgentMicrogridsV2()` factory (already done)

---

## IMPLEMENTATION STATUS

| Component | OLD Location | NEW Location | Status |
|-----------|-------------|-------------|---------|
| Network wrapper | `GridEnv` | N/A (direct pandapower) | ✅ Not needed |
| Microgrid agent | `GridEnv` (pseudo-agent) | `GridAgent` | ✅ Implemented |
| Device agent | N/A | `DeviceAgent` | ✅ Implemented |
| Multi-agent env | `NetworkedGridEnv` | `MultiAgentPowerGridEnv` | ✅ Implemented |
| 3-MG scenario | `MultiAgentMicrogrids` | `MultiAgentMicrogridsV2()` | ✅ Implemented |
| Voltage violations | `GridEnv._update_cost_safety()` | N/A | ❌ **TODO** |
| Line loading violations | `GridEnv._update_cost_safety()` | N/A | ❌ **TODO** |
