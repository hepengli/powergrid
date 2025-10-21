# DeviceAgent Cleanup: Removal of Global Info Handling

## Summary

Removed global information handling from `DeviceAgent` to enforce proper separation of concerns in the hierarchical agent architecture.

---

## Rationale

**DeviceAgent should only observe its local device state.**

In a hierarchical architecture:
- **DeviceAgent** (Level 1) = Device-level control, local state only
- **GridAgent** (Level 2) = Coordination, aggregates global info + subordinate states
- **Environment** = Power flow, network state, dataset

Global information (voltages, prices, convergence status) should be provided by:
1. Parent GridAgent through coordination protocols
2. Messages from GridAgent to DeviceAgent
3. Environment passing global context to GridAgent

**Problem**: DeviceAgent was directly accessing global_state and populating `obs.global_info`, violating the abstraction.

---

## Changes Made

### 1. Removed `partial_obs` Parameter

**Before**:
```python
def __init__(
    self,
    device: Device,
    policy: Optional[Policy] = None,
    partial_obs: bool = False,  # ❌ Removed
    agent_id: Optional[str] = None,
):
```

**After**:
```python
def __init__(
    self,
    device: Device,
    policy: Optional[Policy] = None,
    agent_id: Optional[str] = None,
):
```

**Reason**: No longer needed since DeviceAgent always observes local state only.

---

### 2. Removed Global Info Handling in `observe()`

**Before**:
```python
def observe(self, global_state: Dict[str, Any]) -> Observation:
    obs = Observation(timestamp=self._timestep)

    # Local device state
    obs.local = {
        "P": self.device.state.P,
        "Q": self.device.state.Q,
        "on": self.device.state.on,
        "soc": self.device.state.soc,
    }

    # ❌ Global information (if not partial obs)
    if not self.partial_obs:
        if self._bus_id is not None and "bus_vm" in global_state:
            obs.global_info["bus_voltage"] = global_state["bus_vm"][self._bus_id]
            obs.global_info["bus_angle"] = global_state["bus_va"][self._bus_id]

        if "dataset" in global_state:
            obs.global_info["price"] = global_state["dataset"]["price"]
            obs.global_info["load"] = global_state["dataset"]["load"]

        obs.global_info["converged"] = global_state.get("converged", True)

    obs.messages = self.mailbox.copy()
    return obs
```

**After**:
```python
def observe(self, global_state: Dict[str, Any]) -> Observation:
    """Extract observations from device state.

    DeviceAgent only observes its local device state. Global information
    (voltages, prices, etc.) should be provided by the parent GridAgent
    or through coordination messages.

    Args:
        global_state: Environment state (unused by device, provided for interface)

    Returns:
        Observation with only local device state
    """
    obs = Observation(timestamp=self._timestep)

    # ✅ Local device state only
    obs.local = {
        "P": self.device.state.P,
        "Q": getattr(self.device.state, "Q", 0.0),
        "on": self.device.state.on,
    }

    if hasattr(self.device.state, "soc"):
        obs.local["soc"] = self.device.state.soc

    if hasattr(self.device.state, "Pmax"):
        obs.local["Pmax"] = self.device.state.Pmax
        obs.local["Pmin"] = self.device.state.Pmin

    # ✅ Messages from coordinator/other agents (e.g., price signals)
    obs.messages = self.mailbox.copy()

    return obs
```

**Changes**:
- ✅ Only populates `obs.local` with device state
- ✅ No longer populates `obs.global_info`
- ✅ Updated docstring to clarify responsibility
- ✅ `global_state` parameter kept for interface compatibility but unused

---

### 3. Removed Internal Attributes

**Before**:
```python
self.device = device
self.policy = policy or RandomPolicy(action_space)
self.partial_obs = partial_obs  # ❌ Removed
self._bus_id = getattr(device, "bus", None)  # ❌ Removed
```

**After**:
```python
self.device = device
self.policy = policy or RandomPolicy(action_space)
```

**Reason**: `_bus_id` was only used for looking up bus voltages in global_state.

---

### 4. Updated `_build_observation_space()` Signature

**Before**:
```python
def _build_observation_space(
    self,
    device: Device,
    partial_obs: bool,  # ❌ Removed
) -> gym.Space:
```

**After**:
```python
def _build_observation_space(self, device: Device) -> gym.Space:
    """Build observation space for local device state.

    Args:
        device: Device object

    Returns:
        Gymnasium observation space for local state only
    """
```

---

## Files Updated

### Core Implementation
- ✅ `powergrid/agents/device_agent.py` - Removed global info handling

### Tests
- ✅ `tests/agents/test_device_agent.py` - Updated test expectations
  - Removed `test_device_agent_partial_obs()` test
  - Updated `test_device_agent_observe()` to expect empty `global_info`

### Examples
- ✅ `examples/agent_demo.py` - Removed `partial_obs=False` arguments

### Documentation
- ✅ `powergrid/agents/README.md` - Updated DeviceAgent examples
- ✅ `docs/ARCHITECTURE_COMPARISON.md` - Updated DeviceAgent signature

### Bug Fixes
- ✅ `powergrid/agents/grid_agent.py` - Removed invalid `priority` parameter from `send_message()`

---

## How to Provide Global Info to DeviceAgent

DeviceAgent can still receive global information through **messages**:

### Example: Price Signal Protocol

```python
# In GridAgent with PriceSignalProtocol
coordinator.coordinate_subordinates(global_state)
# ↓ Protocol computes price and sends message
# ↓ Message: {"price": 50.0}

# In DeviceAgent
obs = device_agent.observe(global_state)
# obs.messages contains [Message(content={"price": 50.0})]

# Policy can use message content
price = obs.messages[0].content.get("price", 0.0) if obs.messages else 0.0
```

### Example: Observation Augmentation by GridAgent

```python
# In GridAgent.observe()
sub_obs = device_agent.observe(global_state)
# ↓ GridAgent adds global context
sub_obs.global_info["bus_voltage"] = global_state["bus_vm"][bus_id]
sub_obs.global_info["price"] = global_state["dataset"]["price"]
```

---

## Benefits

### ✅ 1. Clear Separation of Concerns
- DeviceAgent = Local device state only
- GridAgent = Coordination + global context
- Environment = Network simulation + power flow

### ✅ 2. Proper Hierarchical Abstraction
- Devices don't need to know about buses, voltages, or system-wide state
- GridAgent mediates between environment and devices
- Coordination happens through protocols, not direct state access

### ✅ 3. Simpler DeviceAgent
- Fewer parameters (`partial_obs` removed)
- Smaller observation (only local state)
- Clearer API contract

### ✅ 4. Better Modularity
- DeviceAgent can be used in any context without global_state
- GridAgent controls what information flows to subordinates
- Easy to test DeviceAgent in isolation

---

## Migration Guide

### Old Code (BEFORE):
```python
# DeviceAgent directly accessed global info
agent = DeviceAgent(device=ess, partial_obs=False)
obs = agent.observe(global_state)
price = obs.global_info["price"]  # ❌ No longer available
```

### New Code (AFTER):
```python
# DeviceAgent only has local state
agent = DeviceAgent(device=ess)
obs = agent.observe(global_state)

# Global info comes from messages (sent by GridAgent)
if obs.messages:
    price = obs.messages[0].content.get("price", 0.0)
```

### Recommended Pattern:
```python
# In GridAgent hierarchy
class CustomGridAgent(GridAgent):
    def coordinate_subordinates(self, global_state):
        # Get price from environment
        price = global_state["dataset"]["price"]

        # Send to all subordinates
        for sub_id, sub_agent in self.subordinates.items():
            msg = self.send_message(
                content={"price": price, "voltage": ...},
                recipients=[sub_id]
            )
            sub_agent.receive_message(msg)
```

---

## Backward Compatibility

### Breaking Changes
⚠️ **API Changes**:
- Removed `partial_obs` parameter from `DeviceAgent.__init__()`
- `DeviceAgent.observe()` no longer populates `obs.global_info`

### Migration Path
✅ **Easy migration**:
1. Remove `partial_obs=...` from `DeviceAgent()` calls
2. If using `obs.global_info`, switch to message-based coordination
3. Update tests to expect empty `global_info`

---

## Testing

### Verified
✅ `tests/agents/test_device_agent.py::test_device_agent_observe` - PASSED
✅ `examples/agent_demo.py` - Runs successfully
✅ Observation vector shape reduced from (9,) to (4,) (local state only)
✅ Messages still received correctly from GridAgent

---

## Conclusion

DeviceAgent is now a **true local agent** that:
- ✅ Only observes its device state
- ✅ Receives coordination signals via messages
- ✅ Doesn't directly access global environment state
- ✅ Fits cleanly into hierarchical architecture

This change enforces proper abstraction boundaries and makes the agent hierarchy more modular and testable.
