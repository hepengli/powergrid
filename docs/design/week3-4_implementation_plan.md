# Week 3-4 Implementation Plan: Multi-Agent Environment API

**Document Version**: 1.0
**Date**: 2025-10-19
**Owner**: Architect
**Status**: Ready for Implementation

---

## **Overview**

**Objective**: Implement PettingZoo-compatible `MultiAgentPowerGridEnv` with hierarchical control and protocol-based coordination.

**Key Architectural Decisions**:
- **GridAgents** are the primary RL-controllable agents (microgrid controllers)
- **DeviceAgents** are subordinates managed internally by GridAgents
- **Two protocol types**:
  - **Vertical** (agent-owned): Parent → child coordination
  - **Horizontal** (environment-owned): Peer ↔ peer coordination
- **SystemAgent** deferred to Week 11-12

**Timeline**: 10 working days
**Success Metrics**: PettingZoo environment + 3 examples + MAPPO training convergence

---

## **Week 3: Core Environment & Protocol System**

### **Day 1-2: Protocol System Refactoring**

#### **Task 3.1: Refactor Protocol Base Classes**

**File**: `powergrid/agents/protocols.py` 🔧 UPDATE

**Objective**: Split `Protocol` into `VerticalProtocol` and `HorizontalProtocol` base classes

**Implementation Steps**:

1. **Add base classes** (at top of file):
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from .base import Agent, Observation, AgentID
import numpy as np

class Protocol(ABC):
    """Base class for all coordination protocols."""
    pass

# =============================================================================
# VERTICAL PROTOCOLS (Agent-owned: Parent → Child)
# =============================================================================

class VerticalProtocol(Protocol):
    """
    Vertical coordination protocol for parent → subordinate communication.

    Each agent owns its own vertical protocol to coordinate its subordinates.
    This is decentralized - each agent independently manages its children.

    Example:
        GridAgent owns a PriceSignalProtocol to coordinate its DeviceAgents.
    """

    @abstractmethod
    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """
        Compute coordination signals for subordinates.

        Args:
            subordinate_observations: Observations from all subordinate agents
            parent_action: Optional action from parent's policy (e.g., price to broadcast)

        Returns:
            Dict mapping subordinate_id → coordination signal
            Example: {'ess1': {'price': 50.0}, 'dg1': {'price': 50.0}}
        """
        pass


# =============================================================================
# HORIZONTAL PROTOCOLS (Environment-owned: Peer ↔ Peer)
# =============================================================================

class HorizontalProtocol(Protocol):
    """
    Horizontal coordination protocol for peer-to-peer communication.

    The environment owns and runs horizontal protocols, as they require
    global view of all agents. Agents participate but don't run the protocol.

    Example:
        Environment runs PeerToPeerTradingProtocol to enable trading between
        GridAgents MG1, MG2, and MG3.
    """

    @abstractmethod
    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """
        Coordinate peer agents (requires global view).

        Args:
            agents: All participating agents
            observations: Observations from all agents
            topology: Optional network topology (e.g., adjacency matrix for trades)

        Returns:
            Dict mapping agent_id → coordination signal
            Example: {'MG1': {'trades': [...]}, 'MG2': {'trades': [...]}}
        """
        pass
```

2. **Update existing protocols to inherit from `VerticalProtocol`**:
```python
class NoProtocol(VerticalProtocol):
    """No coordination - subordinates act independently."""

    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Return empty coordination signals."""
        return {sub_id: {} for sub_id in subordinate_observations}


class PriceSignalProtocol(VerticalProtocol):
    """Price-based coordination via marginal price signals."""

    def __init__(self, initial_price: float = 50.0):
        """
        Args:
            initial_price: Initial electricity price ($/MWh)
        """
        self.price = initial_price

    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Broadcast price signal to all subordinates."""
        # Update price from parent action if provided
        if parent_action is not None:
            if isinstance(parent_action, dict):
                self.price = parent_action.get("price", self.price)
            else:
                self.price = float(parent_action)

        # Broadcast to all subordinates
        return {
            sub_id: {"price": self.price}
            for sub_id in subordinate_observations
        }


class SetpointProtocol(VerticalProtocol):
    """Setpoint-based coordination - parent assigns power setpoints."""

    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Distribute setpoints to subordinate agents."""
        if parent_action is None:
            return {sub_id: {} for sub_id in subordinate_observations}

        # Distribute setpoints (parent_action should be dict of {sub_id: setpoint})
        signals = {}
        for sub_id in subordinate_observations:
            if sub_id in parent_action:
                signals[sub_id] = {"setpoint": parent_action[sub_id]}
            else:
                signals[sub_id] = {}

        return signals
```

3. **Add horizontal protocol implementations**:
```python
class NoHorizontalProtocol(HorizontalProtocol):
    """No peer coordination - agents act independently."""

    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Return empty signals for all agents."""
        return {aid: {} for aid in agents}


class PeerToPeerTradingProtocol(HorizontalProtocol):
    """
    Peer-to-peer energy trading marketplace.

    Agents submit bids/offers based on their net demand and marginal cost.
    The environment (acting as market auctioneer) clears the market and
    sends trade confirmations back to agents.
    """

    def __init__(self, trading_fee: float = 0.01):
        """
        Args:
            trading_fee: Transaction fee as fraction of trade price
        """
        self.trading_fee = trading_fee

    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Run peer-to-peer market clearing."""
        # Step 1: Collect bids and offers from all agents
        bids = {}
        offers = {}

        for agent_id, obs in observations.items():
            # Agents should compute these in their observe() method
            net_demand = obs.local.get('net_demand', 0)
            marginal_cost = obs.local.get('marginal_cost', 50)

            if net_demand > 0:  # Need to buy
                bids[agent_id] = {
                    'quantity': net_demand,
                    'max_price': marginal_cost * 1.2  # Willing to pay 20% premium
                }
            elif net_demand < 0:  # Can sell
                offers[agent_id] = {
                    'quantity': -net_demand,
                    'min_price': marginal_cost * 0.8  # Willing to sell at 20% discount
                }

        # Step 2: Market clearing
        trades = self._clear_market(bids, offers)

        # Step 3: Generate trade confirmations
        signals = {}
        for buyer, seller, quantity, price in trades:
            if buyer not in signals:
                signals[buyer] = {'trades': []}
            if seller not in signals:
                signals[seller] = {'trades': []}

            signals[buyer]['trades'].append({
                'counterparty': seller,
                'quantity': quantity,  # Positive = buying
                'price': price
            })
            signals[seller]['trades'].append({
                'counterparty': buyer,
                'quantity': -quantity,  # Negative = selling
                'price': price
            })

        return signals

    def _clear_market(
        self,
        bids: Dict[AgentID, Dict],
        offers: Dict[AgentID, Dict]
    ) -> List[Tuple[AgentID, AgentID, float, float]]:
        """
        Simple market clearing algorithm.

        Returns:
            List of (buyer_id, seller_id, quantity, price)
        """
        trades = []

        # Sort bids (descending by price) and offers (ascending by price)
        sorted_bids = sorted(
            bids.items(),
            key=lambda x: x[1]['max_price'],
            reverse=True
        )
        sorted_offers = sorted(
            offers.items(),
            key=lambda x: x[1]['min_price']
        )

        bid_idx = 0
        offer_idx = 0

        while bid_idx < len(sorted_bids) and offer_idx < len(sorted_offers):
            buyer_id, bid = sorted_bids[bid_idx]
            seller_id, offer = sorted_offers[offer_idx]

            # Check if trade is feasible
            if bid['max_price'] >= offer['min_price']:
                # Trade price: midpoint of bid and offer
                trade_price = (bid['max_price'] + offer['min_price']) / 2

                # Trade quantity: min of bid and offer
                trade_qty = min(bid['quantity'], offer['quantity'])

                trades.append((buyer_id, seller_id, trade_qty, trade_price))

                # Update remaining quantities
                bid['quantity'] -= trade_qty
                offer['quantity'] -= trade_qty

                if bid['quantity'] == 0:
                    bid_idx += 1
                if offer['quantity'] == 0:
                    offer_idx += 1
            else:
                break  # No more feasible trades

        return trades


class ConsensusProtocol(HorizontalProtocol):
    """
    Distributed consensus via gossip algorithm.

    Agents iteratively average their values with neighbors until convergence.
    Useful for coordinated frequency regulation or voltage control.
    """

    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01):
        """
        Args:
            max_iterations: Maximum gossip iterations
            tolerance: Convergence threshold
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Run consensus algorithm (gossip)."""
        # Initialize with local values
        values = {
            agent_id: obs.local.get('control_value', 0)
            for agent_id, obs in observations.items()
        }

        # Build adjacency from topology or use fully connected
        if topology and 'adjacency' in topology:
            adjacency = topology['adjacency']
        else:
            # Fully connected graph
            adjacency = {
                aid: [other for other in agents if other != aid]
                for aid in agents
            }

        # Iterative consensus
        for iteration in range(self.max_iterations):
            new_values = {}

            for agent_id in agents:
                # Average with neighbors
                neighbors = adjacency.get(agent_id, [])
                neighbor_vals = [values[nid] for nid in neighbors if nid in values]

                if neighbor_vals:
                    new_values[agent_id] = (
                        values[agent_id] + sum(neighbor_vals)
                    ) / (len(neighbor_vals) + 1)
                else:
                    new_values[agent_id] = values[agent_id]

            # Check convergence
            max_change = max(
                abs(new_values[aid] - values[aid])
                for aid in agents
            )

            values = new_values

            if max_change < self.tolerance:
                break

        # Return consensus values
        return {
            agent_id: {'consensus_value': values[agent_id]}
            for agent_id in agents
        }
```

**Testing**:
- Create `tests/test_protocols.py`
- Test each protocol independently
- Verify vertical protocols work with mock agents
- Verify horizontal protocols clear markets correctly

**Acceptance Criteria**:
- [ ] `VerticalProtocol` and `HorizontalProtocol` base classes added
- [ ] All existing protocols updated to inherit from `VerticalProtocol`
- [ ] `PeerToPeerTradingProtocol` implemented and tested
- [ ] `ConsensusProtocol` implemented and tested
- [ ] Unit tests passing (>80% coverage)

**Time Estimate**: 1.5 days

---

#### **Task 3.2: Update GridAgent for Vertical Protocol**

**File**: `powergrid/agents/grid_agent.py` 🔧 UPDATE

**Objective**: Add `coordinate_subordinates()` method and clarify protocol ownership

**Implementation Steps**:

1. **Update constructor parameter name**:
```python
class GridAgent(Agent):
    def __init__(
        self,
        agent_id: AgentID,
        subordinates: List[DeviceAgent],
        vertical_protocol: Optional[VerticalProtocol] = None,  # Renamed from 'protocol'
        policy: Optional[Policy] = None,
        centralized: bool = False,
    ):
        """
        Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: List of device agents to coordinate
            vertical_protocol: Protocol for coordinating subordinate devices (agent-owned)
            policy: High-level policy (optional)
            centralized: If True, outputs single action for all subordinates
        """
        # ... existing code ...
        self.vertical_protocol = vertical_protocol or NoProtocol()
```

2. **Add `coordinate_subordinates()` method**:
```python
def coordinate_subordinates(self, global_state: Dict) -> None:
    """
    Coordinate subordinate devices using vertical protocol.

    This method is called by the environment during the coordination phase.
    The GridAgent runs its vertical protocol and sends messages to subordinates.

    Args:
        global_state: Global environment state for subordinates to observe
    """
    if not self.subordinates:
        return

    # Collect subordinate observations
    sub_obs = {
        sub_id: sub_agent.observe(global_state)
        for sub_id, sub_agent in self.subordinates.items()
    }

    # Run vertical protocol (coordinator_action could come from self.policy)
    coordinator_action = None  # Or self.policy.forward(...) if using learned coordination
    signals = self.vertical_protocol.coordinate(sub_obs, coordinator_action)

    # Send coordination signals to subordinates
    for sub_id, signal in signals.items():
        if signal:  # Only send non-empty signals
            msg = self.send_message(content=signal, recipients=[sub_id])
            self.subordinates[sub_id].receive_message(msg)
```

3. **Update docstrings** to clarify:
   - `vertical_protocol` is owned by this agent
   - Used for coordinating subordinates only
   - Separate from horizontal protocols (peer coordination)

**Testing**:
- Test `coordinate_subordinates()` with mock DeviceAgents
- Verify messages are sent to subordinates
- Test with different vertical protocols (price signal, setpoint)

**Acceptance Criteria**:
- [ ] `coordinate_subordinates()` method added
- [ ] Parameter renamed to `vertical_protocol`
- [ ] Docstrings updated
- [ ] Unit tests passing

**Time Estimate**: 0.5 days

---

### **Day 3-5: PettingZoo Environment Core**

#### **Task 3.3: Create MultiAgentPowerGridEnv**

**File**: `powergrid/envs/multi_agent/pettingzoo_env.py` ✨ NEW

**Objective**: Implement PettingZoo-compatible environment with protocol coordination

**Implementation Steps**:

1. **Create file structure**:
```python
"""
PettingZoo-compatible multi-agent environment for power grid control.

This environment supports:
- Hierarchical control with GridAgents (microgrid controllers)
- Vertical protocols (GridAgent → DeviceAgent coordination)
- Horizontal protocols (GridAgent ↔ GridAgent peer coordination)
- Flexible configuration via dict-based config
"""

from pettingzoo import ParallelEnv
import pandapower as pp
import numpy as np
from typing import Dict, Any, List, Optional
import gymnasium as gym

from powergrid.agents import GridAgent, DeviceAgent
from powergrid.agents.protocols import (
    VerticalProtocol, HorizontalProtocol,
    PriceSignalProtocol, SetpointProtocol, NoProtocol,
    PeerToPeerTradingProtocol, ConsensusProtocol, NoHorizontalProtocol
)
from powergrid.agents.base import Message
```

2. **Implement core class**:
```python
class MultiAgentPowerGridEnv(ParallelEnv):
    """
    PettingZoo-compatible environment for multi-agent microgrid control.

    Key Features:
    - GridAgents as primary RL-controllable agents (microgrid controllers)
    - DeviceAgents managed internally by GridAgents
    - Vertical protocols (agent-owned) for parent → child coordination
    - Horizontal protocols (environment-owned) for peer ↔ peer coordination
    - Extensible level-based architecture

    Example:
        config = {
            'microgrids': [
                {
                    'name': 'MG1',
                    'network': IEEE13Bus('MG1'),
                    'devices': [ESS(...), DG(...), RES(...)],
                    'vertical_protocol': 'price_signal',
                    'dataset': {...}
                },
                ...
            ],
            'horizontal_protocol': 'p2p_trading',
            'episode_length': 24,
            'train': True
        }

        env = MultiAgentPowerGridEnv(config)
        obs, info = env.reset()

        for _ in range(24):
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)
    """

    metadata = {"name": "multi_agent_powergrid"}

    def __init__(self, config: Dict):
        """
        Initialize multi-agent environment.

        Args:
            config: Configuration dictionary with structure:
                {
                    'network': Base network (e.g., IEEE34Bus('DSO')),
                    'microgrids': [
                        {
                            'name': str,  # Agent ID
                            'network': pandapower network,
                            'devices': [Device, ...],
                            'vertical_protocol': 'price_signal' | 'setpoint' | 'none',
                            'dataset': {'load': array, 'solar': array, ...}
                        },
                        ...
                    ],
                    'horizontal_protocol': 'p2p_trading' | 'consensus' | 'none',
                    'topology': {'adjacency': {aid: [neighbor_aids, ...]}},  # Optional
                    'episode_length': int,
                    'base_power': float,
                    'train': bool,
                    'penalty': float,
                    'share_reward': bool
                }
        """
        super().__init__()

        self.config = config
        self.episode_length = config.get('episode_length', 24)
        self.base_power = config.get('base_power', 1.0)
        self.train = config.get('train', True)
        self.timestep = 0

        # Build network and agents
        self.net = self._build_network(config)
        self.agents = self._build_grid_agents(config['microgrids'])

        # Build horizontal protocol (environment-owned)
        self.horizontal_protocol = self._build_horizontal_protocol(
            config.get('horizontal_protocol', 'none')
        )

        # PettingZoo API requirements
        self.possible_agents = list(self.agents.keys())
        self._agent_ids = self.possible_agents.copy()

        # Action/observation spaces per agent
        self.action_spaces = {
            aid: agent.action_space
            for aid, agent in self.agents.items()
        }
        self.observation_spaces = {
            aid: agent.observation_space
            for aid, agent in self.agents.items()
        }

    def _build_network(self, config: Dict) -> pp.pandapowerNet:
        """
        Build merged pandapower network from all microgrids.

        Args:
            config: Full config dict

        Returns:
            Merged pandapower network
        """
        # Get base network or create empty
        if 'network' in config and config['network'] is not None:
            main_net = config['network']
        else:
            # Create empty network
            main_net = pp.create_empty_network()

        # Merge each microgrid into main network
        for mg_cfg in config['microgrids']:
            mg_net = mg_cfg['network']

            # Add devices to microgrid network (create PP elements)
            self._add_devices_to_network(mg_net, mg_cfg['name'], mg_cfg['devices'])

            # Merge into main network
            if len(main_net.bus) > 0:
                # TODO: Implement merge logic (connect at PCC bus)
                main_net, _ = pp.merge_nets(main_net, mg_net, validate=False,
                                             return_net2_reindex_lookup=True)
            else:
                main_net = mg_net

        return main_net

    def _add_devices_to_network(
        self,
        net: pp.pandapowerNet,
        mg_name: str,
        devices: List
    ) -> None:
        """
        Add pandapower elements for each device.

        Args:
            net: Pandapower network
            mg_name: Microgrid name (prefix for element names)
            devices: List of Device objects
        """
        for device in devices:
            element_name = f"{mg_name}_{device.name}"
            bus_name = f"{mg_name} {device.bus}"
            bus_id = pp.get_element_index(net, 'bus', bus_name)

            if device.__class__.__name__ == 'ESS':
                pp.create_storage(
                    net, bus_id, device.state.P, device.max_e_mwh,
                    sn_mva=device.sn_mva, soc_percent=device.state.soc * 100,
                    min_e_mwh=device.min_e_mwh, name=element_name,
                    max_p_mw=device.max_p_mw, min_p_mw=device.min_p_mw,
                    max_q_mvar=getattr(device, 'max_q_mvar', 0),
                    min_q_mvar=getattr(device, 'min_q_mvar', 0)
                )

            elif device.__class__.__name__ in ['DG', 'RES']:
                pp.create_sgen(
                    net, bus_id, p_mw=device.state.P,
                    sn_mva=getattr(device, 'sn_mva', 1.0),
                    name=element_name,
                    max_p_mw=device.max_p_mw, min_p_mw=device.min_p_mw,
                    max_q_mvar=getattr(device, 'max_q_mvar', 0),
                    min_q_mvar=getattr(device, 'min_q_mvar', 0)
                )

    def _build_grid_agents(self, microgrids_config: List[Dict]) -> Dict[str, GridAgent]:
        """
        Build GridAgent objects from microgrid configs.

        Args:
            microgrids_config: List of microgrid config dicts

        Returns:
            Dict mapping agent_id → GridAgent
        """
        agents = {}

        for mg_cfg in microgrids_config:
            name = mg_cfg['name']

            # Create DeviceAgents for all devices
            device_agents = []
            for device in mg_cfg['devices']:
                # Set device dataset
                if 'dataset' in mg_cfg:
                    device.dataset = mg_cfg['dataset']

                dev_agent = DeviceAgent(
                    agent_id=f"{name}_{device.name}",
                    device=device,
                    level=1
                )
                device_agents.append(dev_agent)

            # Create vertical protocol
            v_protocol_name = mg_cfg.get('vertical_protocol', 'none')
            if v_protocol_name == 'price_signal':
                v_protocol = PriceSignalProtocol()
            elif v_protocol_name == 'setpoint':
                v_protocol = SetpointProtocol()
            else:
                v_protocol = NoProtocol()

            # Create GridAgent (microgrid controller)
            grid_agent = GridAgent(
                agent_id=name,
                subordinates=device_agents,
                vertical_protocol=v_protocol,
                centralized=True  # GridAgent outputs joint action for all devices
            )

            agents[name] = grid_agent

        return agents

    def _build_horizontal_protocol(self, protocol_name: str) -> HorizontalProtocol:
        """
        Build horizontal protocol owned by environment.

        Args:
            protocol_name: Protocol type ('p2p_trading', 'consensus', 'none')

        Returns:
            HorizontalProtocol instance
        """
        if protocol_name == 'p2p_trading':
            return PeerToPeerTradingProtocol()
        elif protocol_name == 'consensus':
            return ConsensusProtocol()
        else:
            return NoHorizontalProtocol()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment and all agents.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observations: Dict mapping agent_id → observation array
            infos: Dict mapping agent_id → info dict
        """
        super().reset(seed=seed)

        self.timestep = 0

        # Reset all GridAgents (which resets subordinate DeviceAgents)
        for agent in self.agents.values():
            agent.reset(seed=seed)

        # Reset dataset timestep
        if self.train:
            # Random starting day for training
            # TODO: Get dataset size from first agent
            first_agent = next(iter(self.agents.values()))
            first_device = next(iter(first_agent.subordinates.values())).device
            if hasattr(first_device, 'dataset') and 'load' in first_device.dataset:
                total_steps = len(first_device.dataset['load'])
                total_days = total_steps // self.episode_length
                if total_days > 1:
                    start_day = self.np_random.integers(0, total_days - 1)
                    self.timestep = start_day * self.episode_length

        # Solve initial power flow
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        # Get initial observations
        obs = self._get_observations()
        infos = {aid: {} for aid in self.possible_agents}

        return obs, infos

    def step(self, actions: Dict[str, Any]):
        """
        Execute one environment step with multi-protocol coordination.

        Execution order:
        1. Horizontal coordination (environment-level peer communication)
        2. Vertical coordination (agent-level subordinate coordination)
        3. Action execution (set device actions)
        4. Device state updates
        5. Power flow solve
        6. Cost/safety computation
        7. Reward calculation
        8. Observation collection

        Args:
            actions: Dict mapping agent_id → action

        Returns:
            observations: Dict mapping agent_id → observation array
            rewards: Dict mapping agent_id → reward
            dones: Dict mapping agent_id → done (plus '__all__')
            truncated: Dict mapping agent_id → truncated (plus '__all__')
            infos: Dict mapping agent_id → info dict
        """
        # PHASE 1: Horizontal coordination (environment-level)
        if not isinstance(self.horizontal_protocol, NoHorizontalProtocol):
            obs_dict = {
                aid: agent.observe(self._get_global_state())
                for aid, agent in self.agents.items()
            }

            signals = self.horizontal_protocol.coordinate(
                agents=self.agents,
                observations=obs_dict,
                topology=self.config.get('topology')
            )

            # Deliver signals to agents
            for agent_id, signal in signals.items():
                if signal:
                    self.agents[agent_id].receive_message(
                        Message(sender='MARKET', content=signal, timestamp=self.timestep)
                    )

        # PHASE 2: Vertical coordination (agent-level)
        global_state = self._get_global_state()
        for agent in self.agents.values():
            agent.coordinate_subordinates(global_state)

        # PHASE 3: Action execution
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if agent.centralized:
                self._set_subordinate_actions(agent, action)

        # PHASE 4: Update device states
        self._update_device_states()
        self._sync_to_pandapower()

        # PHASE 5: Power flow
        converged = self._solve_power_flow()

        # PHASE 6: Costs/safety
        self._update_cost_safety(converged)

        # PHASE 7: Rewards
        rewards = self._compute_rewards(converged)

        # PHASE 8: Observations
        obs = self._get_observations()

        # PHASE 9: Termination
        self.timestep += 1
        terminated = (self.timestep % self.episode_length == 0)
        dones = {aid: terminated for aid in self.possible_agents}
        dones['__all__'] = terminated
        truncated = {aid: False for aid in self.possible_agents}
        truncated['__all__'] = False

        # PHASE 10: Info
        infos = {
            aid: {
                'converged': converged,
                'cost': getattr(self.agents[aid], 'cost', 0),
                'safety': getattr(self.agents[aid], 'safety', 0),
                'timestep': self.timestep
            }
            for aid in self.possible_agents
        }

        return obs, rewards, dones, truncated, infos

    # Helper methods (implement these)
    def _set_subordinate_actions(self, agent: GridAgent, action: Any) -> None:
        """Distribute action vector to subordinate devices."""
        # TODO: Implement action distribution
        pass

    def _update_device_states(self) -> None:
        """Update all device states (dynamics + dataset scalers)."""
        # TODO: Implement device state updates
        pass

    def _sync_to_pandapower(self) -> None:
        """Push device states to pandapower network."""
        # TODO: Implement sync
        pass

    def _solve_power_flow(self) -> bool:
        """Run pandapower power flow."""
        try:
            pp.runpp(self.net)
            return self.net.get('converged', False)
        except:
            self.net['converged'] = False
            return False

    def _update_cost_safety(self, converged: bool) -> None:
        """Update cost/safety for all devices and aggregate to GridAgents."""
        # TODO: Implement cost/safety updates
        pass

    def _compute_rewards(self, converged: bool) -> Dict[str, float]:
        """Compute rewards for each GridAgent."""
        # TODO: Implement reward computation
        pass

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all GridAgents."""
        global_state = self._get_global_state()
        obs = {}

        for aid, agent in self.agents.items():
            agent_obs = agent.observe(global_state)
            obs[aid] = agent_obs.as_vector()

        return obs

    def _get_global_state(self) -> Dict[str, Any]:
        """Build global state dict for agent observations."""
        if self.net.get('converged', False):
            bus_vm = self.net.res_bus['vm_pu'].values
            bus_va = self.net.res_bus['va_degree'].values
        else:
            bus_vm = np.ones(len(self.net.bus))
            bus_va = np.zeros(len(self.net.bus))

        return {
            'bus_vm': bus_vm,
            'bus_va': bus_va,
            'timestep': self.timestep,
            'converged': self.net.get('converged', False)
        }
```

3. **Implement helper methods** (in same file, expand TODOs above)

4. **Add docstrings and type hints** to all methods

**Testing**:
- Create `tests/test_pettingzoo_env.py`
- Test initialization with 2 microgrids
- Test reset
- Test step for 24 timesteps
- Verify PettingZoo API compliance

**Acceptance Criteria**:
- [ ] `MultiAgentPowerGridEnv` class created
- [ ] All core methods implemented
- [ ] Helper methods implemented
- [ ] PettingZoo API compliance verified
- [ ] Unit tests passing

**Time Estimate**: 2 days

---

## **Week 4: Examples, Testing & Documentation**

### **Day 1-2: Example Environments**

#### **Task 4.1: Simple 2-Microgrid Example**

**File**: `examples/multi_agent/simple_2mg.py` ✨ NEW

**Objective**: Create minimal working example for tutorials

**Implementation** (full file content provided in plan above)

**Testing**: Run standalone, verify output

**Acceptance Criteria**:
- [ ] File created and runnable
- [ ] Example trains random policy for 24 steps
- [ ] Clear documentation in docstring

**Time Estimate**: 0.5 days

---

#### **Task 4.2: P2P Trading Example**

**File**: `examples/multi_agent/p2p_trading_3mg.py` ✨ NEW

**Objective**: Demonstrate horizontal protocol coordination

**Implementation** (full file content provided in plan above)

**Testing**: Verify trades occur, agents exchange energy

**Acceptance Criteria**:
- [ ] File created with 3 microgrids
- [ ] P2P trading protocol configured
- [ ] Trades logged and visible in output
- [ ] Documentation explains trading mechanism

**Time Estimate**: 0.5 days

---

#### **Task 4.3: Reimplement MultiAgentMicrogrids**

**File**: `powergrid/envs/multiagent/ieee34_ieee13.py` 🔄 UPDATE

**Objective**: Provide V2 implementation using new environment

**Implementation Steps**:

1. **Add new function** (don't remove old one):
```python
def MultiAgentMicrogridsV2(env_config):
    """
    New PettingZoo-based implementation of MultiAgentMicrogrids.

    3 GridAgents (MG1, MG2, MG3), each managing 4 devices (ESS, DG, PV, WT).
    Compatible with RLlib and other MARL libraries.

    Args:
        env_config: {
            'train': bool,
            'penalty': float,
            'share_reward': bool
        }

    Returns:
        MultiAgentPowerGridEnv instance
    """
    from powergrid.envs.multi_agent.pettingzoo_env import MultiAgentPowerGridEnv
    from powergrid.devices import ESS, DG
    from powergrid.networks.ieee13 import IEEE13Bus
    from powergrid.networks.ieee34 import IEEE34Bus

    config = {
        'network': IEEE34Bus('DSO'),
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        max_e_mwh=2, min_e_mwh=0.2),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.66,
                       cost_curve_coefs=[100, 72.4, 0.5011]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AVA', 'NP15')
            },
            {
                'name': 'MG2',
                'network': IEEE13Bus('MG2'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        max_e_mwh=2, min_e_mwh=0.2),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.60,
                       cost_curve_coefs=[100, 51.6, 0.4615]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'BANCMID', 'NP15')
            },
            {
                'name': 'MG3',
                'network': IEEE13Bus('MG3'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        max_e_mwh=2, min_e_mwh=0.2),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.50,
                       cost_curve_coefs=[100, 51.6, 0.4615]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AZPS', 'NP15')
            }
        ],
        'horizontal_protocol': 'none',
        'episode_length': 24,
        'base_power': 3.0,
        'train': env_config.get('train', True),
        'penalty': env_config.get('penalty', 10),
        'share_reward': env_config.get('share_reward', True)
    }

    return MultiAgentPowerGridEnv(config)
```

2. **Add deprecation notice to old implementation**:
```python
# At top of MultiAgentMicrogrids class
import warnings

class MultiAgentMicrogrids(NetworkedGridEnv):
    def __init__(self, env_config):
        warnings.warn(
            "MultiAgentMicrogrids will be deprecated in v3.0. "
            "Use MultiAgentMicrogridsV2 for new code.",
            FutureWarning,
            stacklevel=2
        )
        super().__init__(env_config)
        # ... existing code ...
```

**Testing**:
- Compare outputs between V1 and V2
- Verify rewards are similar
- Ensure old tests still pass

**Acceptance Criteria**:
- [ ] `MultiAgentMicrogridsV2` function added
- [ ] Old implementation marked deprecated
- [ ] Both versions work
- [ ] Documentation updated

**Time Estimate**: 1 day

---

### **Day 3: Unit Testing**

#### **Task 4.4: Protocol Unit Tests**

**File**: `tests/test_protocols.py` ✨ NEW

**Test Cases**:
```python
import pytest
import numpy as np
from powergrid.agents.protocols import (
    PriceSignalProtocol, SetpointProtocol,
    PeerToPeerTradingProtocol, ConsensusProtocol
)
from powergrid.agents.base import Observation

def test_price_signal_protocol():
    """Test price signal vertical protocol."""
    protocol = PriceSignalProtocol(initial_price=50.0)

    # Mock subordinate observations
    sub_obs = {
        'ess1': Observation(local={'P': 0.5}),
        'dg1': Observation(local={'P': 0.3})
    }

    # Coordinate without parent action
    signals = protocol.coordinate(sub_obs, parent_action=None)

    assert 'ess1' in signals
    assert 'dg1' in signals
    assert signals['ess1']['price'] == 50.0
    assert signals['dg1']['price'] == 50.0

    # Coordinate with parent action
    signals = protocol.coordinate(sub_obs, parent_action=80.0)
    assert signals['ess1']['price'] == 80.0

def test_setpoint_protocol():
    """Test setpoint vertical protocol."""
    protocol = SetpointProtocol()

    sub_obs = {
        'ess1': Observation(local={'P': 0.5}),
        'dg1': Observation(local={'P': 0.3})
    }

    # Without parent action
    signals = protocol.coordinate(sub_obs, parent_action=None)
    assert signals['ess1'] == {}

    # With parent action
    parent_action = {'ess1': 0.8, 'dg1': 0.4}
    signals = protocol.coordinate(sub_obs, parent_action)
    assert signals['ess1']['setpoint'] == 0.8
    assert signals['dg1']['setpoint'] == 0.4

def test_p2p_trading_protocol():
    """Test P2P trading horizontal protocol."""
    from powergrid.agents import GridAgent

    protocol = PeerToPeerTradingProtocol()

    # Mock agents
    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True)
    }

    # Mock observations with net demand
    observations = {
        'MG1': Observation(local={'net_demand': -0.5, 'marginal_cost': 40}),  # Seller
        'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 60})    # Buyer
    }

    # Coordinate
    signals = protocol.coordinate(agents, observations)

    # Verify trade occurred
    assert 'MG1' in signals
    assert 'MG2' in signals
    assert 'trades' in signals['MG1']
    assert 'trades' in signals['MG2']
    assert len(signals['MG1']['trades']) > 0

def test_consensus_protocol():
    """Test consensus horizontal protocol."""
    from powergrid.agents import GridAgent

    protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True),
        'MG3': GridAgent('MG3', subordinates=[], centralized=True)
    }

    # Different initial values
    observations = {
        'MG1': Observation(local={'control_value': 59.9}),
        'MG2': Observation(local={'control_value': 60.1}),
        'MG3': Observation(local={'control_value': 60.0})
    }

    # Coordinate
    signals = protocol.coordinate(agents, observations)

    # Verify consensus reached (values should be close)
    values = [signals[aid]['consensus_value'] for aid in agents]
    assert np.std(values) < 0.1  # Converged
    assert abs(np.mean(values) - 60.0) < 0.1  # Around average
```

**Acceptance Criteria**:
- [ ] All protocol tests passing
- [ ] Edge cases covered
- [ ] >80% coverage

**Time Estimate**: 0.5 days

---

#### **Task 4.5: Environment Unit Tests**

**File**: `tests/test_pettingzoo_env.py` ✨ NEW

**Test Cases**: (structure provided, implement each)
- `test_pettingzoo_api_compliance()`
- `test_multi_microgrid_control()`
- `test_backward_compatibility()`
- `test_reward_computation()`
- `test_convergence_penalty()`

**Acceptance Criteria**:
- [ ] All tests passing
- [ ] PettingZoo API verified
- [ ] >80% coverage

**Time Estimate**: 0.5 days

---

### **Day 4: Integration & Training**

#### **Task 4.6: RLlib Training Script**

**File**: `examples/train_mappo_microgrids.py` ✨ NEW

**Implementation** (full code in plan above)

**Testing**:
- Run for 10 iterations locally
- Verify no crashes
- Check reward improvement trend

**Acceptance Criteria**:
- [ ] Script runs without errors
- [ ] Training converges within 100 iterations
- [ ] Checkpointing works
- [ ] CLI arguments functional

**Time Estimate**: 0.5 days

---

#### **Task 4.7: RLlib Integration Tests**

**File**: `tests/integration/test_rllib.py` ✨ NEW

**Test Cases**:
```python
def test_mappo_training():
    """Test MAPPO training for 5 iterations."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2

    config = (
        PPOConfig()
        .environment(MultiAgentMicrogridsV2, env_config={'train': True, 'penalty': 10})
        .multi_agent(
            policies={'shared_policy'},
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy'
        )
        .training(train_batch_size=512)
        .rollouts(num_rollout_workers=0)
    )

    algo = config.build()

    for i in range(5):
        result = algo.train()
        assert 'episode_reward_mean' in result

    algo.stop()

def test_ippo_training():
    """Test IPPO with separate policies."""
    # Similar to above but with 3 separate policies
    pass
```

**Acceptance Criteria**:
- [ ] Both tests pass
- [ ] Training completes without errors
- [ ] Compatible with Ray 2.9.0

**Time Estimate**: 0.5 days

---

### **Day 5: Documentation**

#### **Task 4.8: Multi-Agent Quickstart Guide**

**File**: `docs/multi_agent_quickstart.md` ✨ NEW

**Content Outline**:
1. Introduction to multi-agent architecture
2. GridAgent vs DeviceAgent explained
3. Vertical vs Horizontal protocols explained
4. Tutorial: Simple 2-microgrid setup
5. Tutorial: Training with RLlib MAPPO
6. Tutorial: P2P trading example
7. FAQ and troubleshooting

**Acceptance Criteria**:
- [ ] All sections complete
- [ ] Code examples tested and working
- [ ] Clear diagrams/illustrations

**Time Estimate**: 0.5 days

---

#### **Task 4.9: Protocol Guide**

**File**: `docs/protocol_guide.md` ✨ NEW

**Content Outline**:
1. Protocol system overview
2. Vertical protocols in depth
3. Horizontal protocols in depth
4. Creating custom vertical protocols
5. Protocol comparison table
6. Advanced: Combining protocols

**Acceptance Criteria**:
- [ ] All sections complete
- [ ] Custom protocol tutorial tested
- [ ] Examples provided

**Time Estimate**: 0.5 days

---

## **Deliverables Checklist**

### **Code** (12 files)

**New Files**:
- [ ] `powergrid/envs/multi_agent/pettingzoo_env.py`
- [ ] `examples/multi_agent/simple_2mg.py`
- [ ] `examples/multi_agent/p2p_trading_3mg.py`
- [ ] `examples/train_mappo_microgrids.py`
- [ ] `tests/test_protocols.py`
- [ ] `tests/test_pettingzoo_env.py`
- [ ] `tests/integration/test_rllib.py`

**Updated Files**:
- [ ] `powergrid/agents/protocols.py`
- [ ] `powergrid/agents/grid_agent.py`
- [ ] `powergrid/envs/multiagent/ieee34_ieee13.py`

### **Documentation** (2 files)

- [ ] `docs/multi_agent_quickstart.md`
- [ ] `docs/protocol_guide.md`

### **Tests**

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] >80% code coverage on new files

### **Training**

- [ ] MAPPO trains for 100 iterations
- [ ] Reward convergence achieved (>-50)
- [ ] No crashes or errors

---

## **Success Criteria**

✅ **Functional**:
- PettingZoo environment with 2-3 GridAgents
- Vertical protocols working (GridAgent → DeviceAgent)
- Horizontal protocols working (GridAgent ↔ GridAgent)
- Backward compatibility maintained

✅ **Performance**:
- MAPPO training converges within 100 iterations
- Step time <1s per timestep
- No memory leaks

✅ **Testing**:
- >80% code coverage
- All tests passing
- Integration with RLlib verified

✅ **Documentation**:
- Quickstart guide complete with 2+ examples
- Protocol guide with custom protocol tutorial
- API docstrings for all public methods

---

## **Risk Mitigation**

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| PettingZoo API complexity | Medium | High | Start with ParallelEnv (simpler), use official examples as reference |
| Network merging issues | Medium | Medium | Test with simple networks first, add merge logic incrementally |
| Protocol coordination bugs | Medium | Medium | Unit test each protocol independently before integration |
| RLlib compatibility | Low | High | Pin Ray version (2.9.0), test early (Day 4) |
| Performance issues | Low | Medium | Profile if needed, optimize critical paths only |

---

## **Dependencies**

```bash
# Required
pip install pettingzoo>=1.24.0
pip install "ray[rllib]==2.9.0"
pip install gymnasium>=0.29.0
pip install pandapower>=2.14.0

# Testing
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0

# Optional (for examples)
pip install wandb  # For logging
```

---

## **Daily Standup Template**

**What did I complete yesterday?**
- [ ] Task X.Y completed
- [ ] Tests passing for feature Z

**What will I work on today?**
- [ ] Task X.Y+1
- [ ] Fix bug in module A

**Any blockers?**
- None / Blocked by: [description]

---

## **End of Week 4 Checklist**

Before marking weeks 3-4 complete, verify:

- [ ] All deliverables checked off
- [ ] All tests passing (unit + integration)
- [ ] >80% code coverage achieved
- [ ] Documentation complete and reviewed
- [ ] MAPPO training converges
- [ ] Backward compatibility verified
- [ ] Code reviewed by team
- [ ] Ready for Week 5 (YAML configuration)

---

**Document End**
