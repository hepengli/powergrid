# Tasks for Week 3-4

## High Priority Tasks

### 1. Implement Mixed Action Space Support
**Location**: `powergrid/agents/grid_agent.py:394-396`

**Issue**: MultiAgentMicrogrids environment fails with:
```python
NotImplementedError: Mixed action spaces not yet supported
```

**Root Cause**: Microgrids have devices with both continuous (ESS, DG power) and discrete (DG unit commitment) actions. The current implementation doesn't support combining these.

**Proposed Solutions**:
- **Option A**: Use `gymnasium.spaces.Dict` to separate continuous and discrete actions
  ```python
  from gymnasium.spaces import Dict
  return Dict({
      'continuous': Box(low=low, high=high, dtype=np.float32),
      'discrete': MultiDiscrete(discrete_n)
  })
  ```
- **Option B**: Flatten to single continuous space (convert discrete to continuous with thresholding)
- **Option C**: Use `gymnasium.spaces.Tuple` for ordered action components

**Files to Modify**:
- `powergrid/agents/grid_agent.py` - `get_grid_action_space()` method
- `powergrid/agents/device_agent.py` - `_set_device_action()` to handle Dict/Tuple spaces
- `powergrid/envs/multi_agent/networked_grid_env.py` - Action space registration

**Tests to Update**:
- `tests/envs/test_multi_agent_microgrids.py` - Remove skip marker, enable all 9 tests
- `tests/agents/test_grid_agent.py` - Add tests for mixed action space handling

**Priority**: HIGH - Blocks full MultiAgentMicrogrids environment functionality

---

## Medium Priority Tasks (Code TODOs)

### 2. Implement Global Info Aggregation
**Locations**:
- `powergrid/agents/device_agent.py:169`
- `powergrid/agents/grid_agent.py:104`

**Current Code**:
```python
# TODO: aggregate global info if needed
obs.global_info = global_state
```

**Description**: Currently just passes through global state without aggregation. Need to implement proper aggregation strategy for:
- Price signals from DSO
- Network-wide voltage/frequency
- Renewable generation forecasts
- System-wide constraints

**Proposed Implementation**:
- Define global info schema (what info is shared across agents)
- Implement aggregation function (mean, max, weighted average?)
- Add filtering based on agent type/location
- Consider privacy/information hiding requirements

**Files to Modify**:
- `powergrid/agents/device_agent.py`
- `powergrid/agents/grid_agent.py`
- `powergrid/core/state.py` - Add global state dataclass

**Tests to Add**:
- Test global info propagation in hierarchy
- Test different aggregation strategies
- Test info filtering

**Priority**: MEDIUM - Affects coordination quality but not blocking

---

### 3. Implement Decentralized Coordination
**Location**: `powergrid/agents/grid_agent.py:143-147`

**Current Code**:
```python
# TODO: this is coordinator action computation
# Non-centralized GridAgent using coordination_policy to coordinate devices
# to compute their actions individually
# Afterwards, GridAgent can also send messages to devices if needed
raise NotImplementedError("Decentralized coordination not yet implemented")
```

**Description**: GridAgent currently only supports centralized control (parent computes all device actions). Need to implement decentralized mode where:
- Devices compute their own actions using local policies
- GridAgent coordinates via messages/signals (price, setpoints, etc.)
- Uses existing protocol system (PriceSignalProtocol, SetpointProtocol)

**Proposed Implementation**:
```python
if not self.centralized:
    # Use protocol to send coordination signals
    coord_signals = self.protocol.coordinate(
        subordinate_observations={d.agent_id: device_obs[d.agent_id] for d in self.devices},
        parent_action=None
    )
    # Devices act based on coordination signals
    for device in self.devices:
        device.act(device_obs[device.agent_id], coord_signals.get(device.agent_id))
    return None  # No centralized action
```

**Files to Modify**:
- `powergrid/agents/grid_agent.py` - `act()` method
- `powergrid/core/protocols.py` - Ensure protocols support this flow

**Tests to Add**:
- Test decentralized GridAgent with PriceSignalProtocol
- Test decentralized GridAgent with SetpointProtocol
- Compare centralized vs decentralized performance

**Priority**: MEDIUM - Important for hierarchical control research

---

### 4. Add Communication Logic to DeviceAgent
**Location**: `powergrid/agents/device_agent.py:191`

**Current Code**:
```python
# TODO: Add communication logic (send/receive message) if needed
```

**Description**: DeviceAgent can observe and act, but doesn't use the message passing system defined in `base.py`. Need to implement:
- Message sending to parent/peers
- Message receiving and processing
- Message queue management

**Proposed Implementation**:
```python
def act(self, observation: Observation, given_action: Any = None) -> Any:
    # Process incoming messages
    messages = self.receive_messages()
    for msg in messages:
        self._process_message(msg)

    # Compute action (existing logic)
    ...

    # Send messages if needed
    if self._should_communicate(observation):
        msg = self._create_message(observation, action)
        self.send_message(msg)

    return action

def _process_message(self, msg: Message) -> None:
    """Process incoming message (update state, adjust action, etc.)"""
    pass

def _should_communicate(self, obs: Observation) -> bool:
    """Decide whether to send message based on observation"""
    return False  # Default: no communication

def _create_message(self, obs: Observation, action: Any) -> Message:
    """Create message to send to parent/peers"""
    pass
```

**Files to Modify**:
- `powergrid/agents/device_agent.py`
- `powergrid/agents/base.py` - Add message queue/inbox

**Tests to Add**:
- Test message sending/receiving
- Test peer-to-peer communication
- Test broadcast messages

**Priority**: MEDIUM - Enables distributed coordination research

---

### 5. Verify Action Format Matching
**Location**: `powergrid/agents/device_agent.py:201`

**Current Code**:
```python
# TODO: verify action format matches policy forward output
assert action.size == self.action.dim_c + self.action.dim_d
```

**Description**: Current assertion checks size but doesn't verify:
- Action bounds (within action_space limits)
- Data type consistency (float32)
- Shape compatibility
- Discrete action validity (within categorical bounds)

**Proposed Implementation**:
```python
def _verify_action(self, action: Any) -> None:
    """Verify action format and bounds."""
    # Check size
    expected_size = self.action.dim_c + self.action.dim_d
    assert action.size == expected_size, f"Action size {action.size} != expected {expected_size}"

    # Check dtype
    assert action.dtype == np.float32, f"Action dtype {action.dtype} != float32"

    # Check continuous bounds
    if self.action.dim_c > 0:
        continuous_part = action[:self.action.dim_c]
        assert self.action_space.contains(continuous_part), "Continuous action out of bounds"

    # Check discrete validity
    if self.action.dim_d > 0:
        discrete_part = action[self.action.dim_c:]
        # Verify discrete actions are valid category indices
        for i, val in enumerate(discrete_part):
            assert 0 <= val < self.action.ncats, f"Discrete action {val} out of range"
```

**Files to Modify**:
- `powergrid/agents/device_agent.py`

**Tests to Add**:
- Test action validation with valid actions
- Test action validation with out-of-bounds actions
- Test action validation with wrong dtype

**Priority**: LOW - Nice to have for debugging, not critical

---

### 6. Enhance Message Attributes
**Location**: `powergrid/agents/base.py:87`

**Current Code**:
```python
# TODO: add more attributes like expiration, priority, etc.
```

**Description**: Message dataclass is minimal. Consider adding:
- `expiration`: Time when message expires (for time-critical coordination)
- `priority`: Message priority level (urgent, normal, low)
- `message_type`: Categorical type (request, response, broadcast, etc.)
- `requires_ack`: Whether message needs acknowledgment
- `in_reply_to`: Message ID this is replying to (for request-response)

**Proposed Implementation**:
```python
@dataclass
class Message:
    sender: AgentID
    content: Dict[str, Any]
    recipient: Optional[Union[AgentID, List[AgentID]]] = None
    timestamp: float = 0.0

    # Extended attributes
    expiration: Optional[float] = None  # Expiration timestamp
    priority: int = 0  # 0=normal, 1=high, -1=low
    message_type: str = "broadcast"  # request, response, broadcast, etc.
    requires_ack: bool = False  # Whether acknowledgment is required
    in_reply_to: Optional[str] = None  # Message ID being replied to
    message_id: str = None  # Unique message identifier

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = f"{self.sender}_{self.timestamp}_{id(self)}"

    def is_expired(self, current_time: float) -> bool:
        """Check if message has expired."""
        if self.expiration is None:
            return False
        return current_time > self.expiration
```

**Files to Modify**:
- `powergrid/agents/base.py`
- Documentation for message protocol

**Tests to Add**:
- Test message expiration
- Test message priority handling
- Test request-response pattern

**Priority**: LOW - Nice to have for advanced coordination

---

## Testing & Documentation Tasks

### 7. Increase Test Coverage for Edge Cases
**Files**: All test files

**Areas needing more coverage**:
- Error handling and failure modes
- Boundary conditions (min/max power limits, SOC limits)
- Concurrent device actions
- Network convergence failures
- Data loading with missing/corrupt data

**Priority**: MEDIUM

---

### 8. Add Integration Tests
**New test files needed**:
- `tests/integration/test_centralized_control.py` - Full centralized control flow
- `tests/integration/test_protocol_coordination.py` - Test protocol-based coordination
- `tests/integration/test_multi_timestep.py` - Test environment over multiple timesteps

**Priority**: MEDIUM

---

### 9. Documentation Improvements
**Files to create/update**:
- `docs/architecture.md` - Overall system architecture
- `docs/agent_hierarchy.md` - Agent types and relationships
- `docs/protocols.md` - Coordination protocol guide
- `docs/extending.md` - How to add new devices/agents/protocols
- Add more docstring examples with code snippets

**Priority**: LOW - Important but not blocking development

---

## Summary by Priority

### High Priority (Week 3)
1. ✅ Implement Mixed Action Space Support - **Blocks MultiAgentMicrogrids**

### Medium Priority (Week 3-4)
2. Implement Global Info Aggregation
3. Implement Decentralized Coordination
4. Add Communication Logic to DeviceAgent
7. Increase Test Coverage for Edge Cases
8. Add Integration Tests

### Low Priority (Week 4+)
5. Verify Action Format Matching
6. Enhance Message Attributes
9. Documentation Improvements

---

## Notes

- All TODOs extracted from codebase are documented above
- Mixed action space is the highest priority blocking issue
- Communication and coordination features are foundational for multi-agent research
- Testing and documentation can be done incrementally alongside development
