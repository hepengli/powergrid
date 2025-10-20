# Protocol Guide

This guide provides an in-depth look at the PowerGrid coordination protocol system, including vertical and horizontal protocols, implementation details, and how to create custom protocols.

---

## Table of Contents

1. [Protocol System Overview](#protocol-system-overview)
2. [Vertical Protocols](#vertical-protocols)
3. [Horizontal Protocols](#horizontal-protocols)
4. [Creating Custom Vertical Protocols](#creating-custom-vertical-protocols)
5. [Creating Custom Horizontal Protocols](#creating-custom-horizontal-protocols)
6. [Protocol Comparison Table](#protocol-comparison-table)
7. [Advanced: Combining Protocols](#advanced-combining-protocols)
8. [Best Practices](#best-practices)

---

## Protocol System Overview

### What are Protocols?

Protocols define **how agents coordinate** with each other. In PowerGrid, there are two types:

1. **Vertical Protocols**: Parent → subordinate coordination
2. **Horizontal Protocols**: Peer ↔ peer coordination

### Design Philosophy

**Separation of Concerns:**
- **Protocols**: Define *how* to coordinate (the mechanism)
- **Policies**: Define *what* to coordinate (the strategy)
- **Agents**: Execute coordination and actions

**Key Benefits:**
- Plug-and-play: Swap protocols without changing agent code
- Composable: Combine vertical and horizontal protocols
- Testable: Protocols are independent, unit-testable components

### Architecture

```
┌────────────────────────────────────────────────┐
│           Environment                          │
│  - Runs horizontal protocols                   │
│  - Coordinates peer agents                     │
└────────────────────────────────────────────────┘
         │
         │ Horizontal Coordination
         │ (P2P Trading, Consensus)
         ▼
    ┌────────┬────────┬────────┐
    │  MG1   │  MG2   │  MG3   │  GridAgents
    │        │        │        │  - Own vertical protocols
    └────────┴────────┴────────┘  - Coordinate subordinates
         │        │        │
         │ Vertical Coordination
         │ (Prices, Setpoints)
         ▼
    ┌────┬────┬────┐
    │ESS │ DG │ PV │  DeviceAgents
    └────┴────┴────┘  - Respond to signals
```

---

## Vertical Protocols

### Purpose

Vertical protocols enable **hierarchical control** where a parent agent (GridAgent) coordinates subordinate agents (DeviceAgents).

### Ownership

Each GridAgent **owns** its own vertical protocol. This enables:
- **Decentralized coordination**: Each microgrid independently manages its devices
- **Heterogeneous control**: Different microgrids can use different protocols
- **Privacy**: No need to share subordinate information with other agents

### Execution Flow

```python
# During environment step:
for grid_agent in agents:
    # 1. Collect subordinate observations
    sub_obs = {sub_id: sub.observe(global_state) for sub_id, sub in subordinates}

    # 2. Run vertical protocol
    signals = grid_agent.vertical_protocol.coordinate(sub_obs, parent_action)

    # 3. Send signals to subordinates
    for sub_id, signal in signals.items():
        subordinates[sub_id].receive_message(Message(content=signal))
```

### Built-in Vertical Protocols

#### 1. NoProtocol

**Use Case:** Independent device operation (baseline)

**How it works:** Returns empty signals for all subordinates

**Example:**
```python
from powergrid.agents.protocols import NoProtocol

protocol = NoProtocol()
signals = protocol.coordinate(subordinate_observations, parent_action=None)
# → {'ESS1': {}, 'DG1': {}, 'PV1': {}}
```

**When to use:**
- Benchmarking (compare against coordinated control)
- Fully learned control (policy directly outputs device actions)

---

#### 2. PriceSignalProtocol

**Use Case:** Economic dispatch via marginal price signals

**How it works:**
1. GridAgent broadcasts an electricity price ($/MWh)
2. Devices optimize locally to maximize profit at that price
3. ESS charges when price is low, discharges when high
4. DG generates more when price is high

**Parameters:**
- `initial_price`: Starting price (default: 50.0 $/MWh)

**Example:**
```python
from powergrid.agents.protocols import PriceSignalProtocol

protocol = PriceSignalProtocol(initial_price=50.0)

# Parent action: new price from learned policy
parent_action = 65.0  # $/MWh

signals = protocol.coordinate(subordinate_obs, parent_action)
# → {'ESS1': {'price': 65.0}, 'DG1': {'price': 65.0}, 'PV1': {'price': 65.0}}
```

**Implementation Details:**
```python
class PriceSignalProtocol(VerticalProtocol):
    def __init__(self, initial_price: float = 50.0):
        self.price = initial_price

    def coordinate(self, subordinate_observations, parent_action=None):
        # Update price from parent action
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
```

**Advantages:**
- Simple, interpretable
- Devices can use local optimization (fast)
- Aligns incentives (price = marginal value)

**Disadvantages:**
- Requires price-responsive device models
- May not satisfy hard constraints

---

#### 3. SetpointProtocol

**Use Case:** Direct centralized control with power setpoints

**How it works:**
1. GridAgent computes power setpoints for each device
2. Devices track their assigned setpoints
3. Useful for model-predictive control (MPC)

**Example:**
```python
from powergrid.agents.protocols import SetpointProtocol

protocol = SetpointProtocol()

# Parent action: dict of setpoints
parent_action = {
    'ESS1': 0.3,   # Charge at 0.3 MW
    'DG1': 0.5,    # Generate 0.5 MW
    'PV1': 0.1     # Solar at 0.1 MW
}

signals = protocol.coordinate(subordinate_obs, parent_action)
# → {'ESS1': {'setpoint': 0.3}, 'DG1': {'setpoint': 0.5}, 'PV1': {'setpoint': 0.1}}
```

**Implementation Details:**
```python
class SetpointProtocol(VerticalProtocol):
    def coordinate(self, subordinate_observations, parent_action=None):
        if parent_action is None:
            return {sub_id: {} for sub_id in subordinate_observations}

        # Distribute setpoints
        signals = {}
        for sub_id in subordinate_observations:
            if sub_id in parent_action:
                signals[sub_id] = {"setpoint": parent_action[sub_id]}
            else:
                signals[sub_id] = {}

        return signals
```

**Advantages:**
- Full control over device outputs
- Can enforce hard constraints
- Compatible with MPC, optimization-based control

**Disadvantages:**
- Requires accurate device models
- High-dimensional action space (one setpoint per device)

---

## Horizontal Protocols

### Purpose

Horizontal protocols enable **peer-to-peer coordination** where multiple GridAgents interact as equals.

### Ownership

The **environment** owns and runs horizontal protocols. This is because:
- Requires **global view** of all agents
- Implements **market mechanisms** (auctioneer)
- Ensures **fairness** and **truthfulness**

### Execution Flow

```python
# During environment step (before vertical coordination):

# 1. Collect observations from all agents
observations = {aid: agent.observe(global_state) for aid, agent in agents.items()}

# 2. Run horizontal protocol
signals = horizontal_protocol.coordinate(agents, observations, topology)

# 3. Deliver signals to agents (via messages)
for agent_id, signal in signals.items():
    agents[agent_id].receive_message(Message(sender='MARKET', content=signal))
```

### Built-in Horizontal Protocols

#### 1. NoHorizontalProtocol

**Use Case:** No peer coordination (baseline)

**How it works:** Returns empty signals for all agents

**Example:**
```python
from powergrid.agents.protocols import NoHorizontalProtocol

protocol = NoHorizontalProtocol()
signals = protocol.coordinate(agents, observations)
# → {'MG1': {}, 'MG2': {}, 'MG3': {}}
```

---

#### 2. PeerToPeerTradingProtocol

**Use Case:** Decentralized energy marketplace

**How it works:**
1. **Agents compute** net demand and marginal cost
2. **Environment collects** bids (buyers) and offers (sellers)
3. **Market clears**: Match bids and offers by price
4. **Trade confirmations** sent back to agents

**Parameters:**
- `trading_fee`: Transaction fee as fraction of price (default: 0.01)

**Example:**
```python
from powergrid.agents.protocols import PeerToPeerTradingProtocol

protocol = PeerToPeerTradingProtocol(trading_fee=0.01)

# Agents include 'net_demand' and 'marginal_cost' in observations
observations = {
    'MG1': Observation(local={'net_demand': -0.5, 'marginal_cost': 40}),  # Seller
    'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 60}),   # Buyer
    'MG3': Observation(local={'net_demand': 0.2, 'marginal_cost': 55}),   # Buyer
}

signals = protocol.coordinate(agents, observations)
# → {
#   'MG1': {'trades': [{'counterparty': 'MG2', 'quantity': -0.3, 'price': 50}]},
#   'MG2': {'trades': [{'counterparty': 'MG1', 'quantity': 0.3, 'price': 50}]},
#   'MG3': {'trades': [{'counterparty': 'MG1', 'quantity': 0.2, 'price': 48}]}
# }
```

**Market Clearing Algorithm:**
```python
def _clear_market(self, bids, offers):
    """Match highest bids with lowest offers."""
    # Sort bids descending, offers ascending
    sorted_bids = sorted(bids, key=lambda x: x['max_price'], reverse=True)
    sorted_offers = sorted(offers, key=lambda x: x['min_price'])

    trades = []
    for bid in sorted_bids:
        for offer in sorted_offers:
            if bid['max_price'] >= offer['min_price']:
                # Trade feasible!
                trade_price = (bid['max_price'] + offer['min_price']) / 2
                trade_qty = min(bid['quantity'], offer['quantity'])
                trades.append((buyer, seller, trade_qty, trade_price))

                # Update remaining quantities
                bid['quantity'] -= trade_qty
                offer['quantity'] -= trade_qty

    return trades
```

**Advantages:**
- Enables local energy markets
- Price discovery through supply/demand
- Reduces reliance on main grid

**Disadvantages:**
- Requires agents to estimate net demand accurately
- No guarantee of market equilibrium

---

#### 3. ConsensusProtocol

**Use Case:** Distributed frequency or voltage regulation

**How it works:**
1. Agents start with different control values
2. Iteratively **average with neighbors** (gossip)
3. Converge to **global average** (consensus)

**Parameters:**
- `max_iterations`: Max gossip rounds (default: 10)
- `tolerance`: Convergence threshold (default: 0.01)

**Example:**
```python
from powergrid.agents.protocols import ConsensusProtocol

protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

# Agents have different initial control values
observations = {
    'MG1': Observation(local={'control_value': 59.9}),  # Hz
    'MG2': Observation(local={'control_value': 60.1}),
    'MG3': Observation(local={'control_value': 60.0}),
}

# Optional: custom topology (default is fully connected)
topology = {
    'adjacency': {
        'MG1': ['MG2'],
        'MG2': ['MG1', 'MG3'],
        'MG3': ['MG2']
    }
}

signals = protocol.coordinate(agents, observations, topology)
# → After convergence:
# {'MG1': {'consensus_value': 60.0},
#  'MG2': {'consensus_value': 60.0},
#  'MG3': {'consensus_value': 60.0}}
```

**Gossip Algorithm:**
```python
for iteration in range(max_iterations):
    new_values = {}
    for agent_id in agents:
        # Average with neighbors
        neighbors = adjacency[agent_id]
        neighbor_vals = [values[nid] for nid in neighbors]

        new_values[agent_id] = (values[agent_id] + sum(neighbor_vals)) / (len(neighbors) + 1)

    # Check convergence
    max_change = max(abs(new_values[aid] - values[aid]) for aid in agents)
    if max_change < tolerance:
        break  # Converged

    values = new_values
```

**Advantages:**
- Fully distributed (no central coordinator)
- Robust to communication delays
- Proven convergence guarantees

**Disadvantages:**
- Requires multiple iterations (slow)
- Sensitive to network topology

---

## Creating Custom Vertical Protocols

### Template

```python
from powergrid.agents.protocols import VerticalProtocol
from powergrid.agents.base import Observation, AgentID
from typing import Dict, Any, Optional

class MyVerticalProtocol(VerticalProtocol):
    """My custom vertical protocol."""

    def __init__(self, param1, param2):
        """Initialize with custom parameters."""
        self.param1 = param1
        self.param2 = param2
        # Add any stateful variables

    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """
        Compute coordination signals for subordinates.

        Args:
            subordinate_observations: Obs from all subordinates
            parent_action: Optional action from parent's policy

        Returns:
            Dict mapping subordinate_id → signal dict
        """
        signals = {}

        for sub_id, obs in subordinate_observations.items():
            # Custom logic here
            signal = self._compute_signal(obs, parent_action)
            signals[sub_id] = signal

        return signals

    def _compute_signal(self, obs, parent_action):
        """Helper method to compute signal for one subordinate."""
        # Your logic here
        return {'custom_field': value}
```

### Example: ADMM Protocol

Alternating Direction Method of Multipliers (ADMM) for distributed optimization.

```python
class ADMMProtocol(VerticalProtocol):
    """ADMM-based coordination for optimal dispatch."""

    def __init__(self, rho=1.0, max_iter=10, tol=1e-3):
        self.rho = rho  # Penalty parameter
        self.max_iter = max_iter
        self.tol = tol

        # ADMM variables (dual variables)
        self.lambdas = {}  # Lagrange multipliers
        self.z = None      # Consensus variable

    def coordinate(self, subordinate_observations, parent_action=None):
        # Extract device states
        device_states = {
            sub_id: obs.local for sub_id, obs in subordinate_observations.items()
        }

        # ADMM iterations
        for iteration in range(self.max_iter):
            # Update primal variables (device setpoints)
            x_new = self._update_primal(device_states)

            # Update consensus variable
            self.z = self._update_consensus(x_new)

            # Update dual variables
            self._update_dual(x_new)

            # Check convergence
            if self._check_convergence(x_new):
                break

        # Return setpoints to devices
        signals = {
            sub_id: {'setpoint': x_new[sub_id], 'lambda': self.lambdas[sub_id]}
            for sub_id in subordinate_observations
        }

        return signals

    def _update_primal(self, device_states):
        """Update device setpoints (primal variables)."""
        # Solve local subproblem for each device
        x_new = {}
        for sub_id, state in device_states.items():
            # x_i = argmin f_i(x_i) + lambda^T x_i + (rho/2) ||x_i - z||^2
            x_new[sub_id] = self._solve_device_subproblem(
                state, self.lambdas.get(sub_id, 0), self.z
            )
        return x_new

    def _update_consensus(self, x):
        """Update consensus variable (average of primal variables)."""
        return sum(x.values()) / len(x)

    def _update_dual(self, x):
        """Update dual variables (Lagrange multipliers)."""
        for sub_id, x_i in x.items():
            self.lambdas[sub_id] = self.lambdas.get(sub_id, 0) + self.rho * (x_i - self.z)

    def _check_convergence(self, x):
        """Check if ADMM has converged."""
        if self.z is None:
            return False
        primal_residual = sum((x[sid] - self.z)**2 for sid in x)**0.5
        return primal_residual < self.tol

    def _solve_device_subproblem(self, state, lambda_val, z):
        """Solve local optimization for one device."""
        # Simplified: return a heuristic setpoint
        # In practice, solve a convex optimization problem
        P_max = state.get('max_p_mw', 1.0)
        P_min = state.get('min_p_mw', 0.0)

        # Optimal setpoint balances cost and consensus penalty
        P_star = z - lambda_val / self.rho
        return np.clip(P_star, P_min, P_max)
```

**Usage:**
```python
from powergrid.agents import GridAgent

grid_agent = GridAgent(
    agent_id='MG1',
    subordinates=[...],
    vertical_protocol=ADMMProtocol(rho=1.0, max_iter=10),
    centralized=True
)
```

---

## Creating Custom Horizontal Protocols

### Template

```python
from powergrid.agents.protocols import HorizontalProtocol
from powergrid.agents.base import Agent, Observation, AgentID
from typing import Dict, Any, Optional

class MyHorizontalProtocol(HorizontalProtocol):
    """My custom horizontal protocol."""

    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """
        Coordinate peer agents.

        Args:
            agents: All participating agents
            observations: Observations from all agents
            topology: Optional network topology

        Returns:
            Dict mapping agent_id → coordination signal
        """
        signals = {}

        # Step 1: Collect information from all agents
        agent_data = self._collect_data(observations)

        # Step 2: Run global coordination algorithm
        results = self._run_coordination(agent_data, topology)

        # Step 3: Generate signals for each agent
        for agent_id in agents:
            signals[agent_id] = self._generate_signal(agent_id, results)

        return signals
```

### Example: Double Auction Protocol

A two-sided auction for energy trading.

```python
class DoubleAuctionProtocol(HorizontalProtocol):
    """
    Double auction for energy trading.

    Buyers submit demand curves, sellers submit supply curves.
    Market clears at intersection (equilibrium price).
    """

    def __init__(self, reserve_price=100.0):
        self.reserve_price = reserve_price  # Max price

    def coordinate(self, agents, observations, topology=None):
        # Collect demand and supply curves
        demand_curves = {}
        supply_curves = {}

        for agent_id, obs in observations.items():
            if 'demand_curve' in obs.local:
                demand_curves[agent_id] = obs.local['demand_curve']
            if 'supply_curve' in obs.local:
                supply_curves[agent_id] = obs.local['supply_curve']

        # Find equilibrium price
        equilibrium_price = self._find_equilibrium(demand_curves, supply_curves)

        # Compute quantities at equilibrium
        signals = {}
        for agent_id in agents:
            if agent_id in demand_curves:
                quantity = self._evaluate_curve(demand_curves[agent_id], equilibrium_price)
                signals[agent_id] = {
                    'trade': 'buy',
                    'quantity': quantity,
                    'price': equilibrium_price
                }
            elif agent_id in supply_curves:
                quantity = self._evaluate_curve(supply_curves[agent_id], equilibrium_price)
                signals[agent_id] = {
                    'trade': 'sell',
                    'quantity': quantity,
                    'price': equilibrium_price
                }
            else:
                signals[agent_id] = {}

        return signals

    def _find_equilibrium(self, demand_curves, supply_curves):
        """Find price where total demand = total supply."""
        # Binary search for equilibrium price
        low, high = 0, self.reserve_price

        for _ in range(20):  # Max iterations
            mid = (low + high) / 2

            total_demand = sum(
                self._evaluate_curve(curve, mid)
                for curve in demand_curves.values()
            )
            total_supply = sum(
                self._evaluate_curve(curve, mid)
                for curve in supply_curves.values()
            )

            if abs(total_demand - total_supply) < 0.01:
                return mid

            if total_demand > total_supply:
                high = mid  # Lower price to reduce demand
            else:
                low = mid   # Raise price to increase supply

        return (low + high) / 2

    def _evaluate_curve(self, curve, price):
        """Evaluate demand/supply curve at given price."""
        # Assume curve is list of (price, quantity) pairs
        # Interpolate to find quantity at given price
        for i in range(len(curve) - 1):
            p1, q1 = curve[i]
            p2, q2 = curve[i + 1]

            if p1 <= price <= p2 or p2 <= price <= p1:
                # Linear interpolation
                return q1 + (q2 - q1) * (price - p1) / (p2 - p1)

        return 0  # Price outside curve range
```

**Usage:**
```python
config = {
    'microgrids': [...],
    'horizontal_protocol_obj': DoubleAuctionProtocol(reserve_price=150.0),
}

env = MultiAgentPowerGridEnv(config)
```

---

## Protocol Comparison Table

| Protocol | Type | Ownership | Use Case | Pros | Cons |
|----------|------|-----------|----------|------|------|
| **NoProtocol** | Vertical | Agent | Baseline | Simple | No coordination |
| **PriceSignalProtocol** | Vertical | Agent | Economic dispatch | Interpretable, fast | Needs price-responsive devices |
| **SetpointProtocol** | Vertical | Agent | Centralized control | Full control | High-dimensional action space |
| **NoHorizontalProtocol** | Horizontal | Environment | Baseline | Simple | No peer coordination |
| **PeerToPeerTradingProtocol** | Horizontal | Environment | Local energy market | Distributed, fair | Requires accurate demand estimates |
| **ConsensusProtocol** | Horizontal | Environment | Frequency/voltage regulation | Robust, distributed | Slow convergence |

---

## Advanced: Combining Protocols

You can combine vertical and horizontal protocols to create sophisticated control schemes.

### Example 1: P2P Trading + Price Signals

```python
config = {
    'microgrids': [
        {
            'name': 'MG1',
            'devices': [...],
            'vertical_protocol': 'price_signal',  # GridAgent → devices
        },
        {
            'name': 'MG2',
            'devices': [...],
            'vertical_protocol': 'price_signal',
        },
    ],
    'horizontal_protocol': 'p2p_trading',  # GridAgent ↔ GridAgent
}

# Execution flow:
# 1. P2P trading: MG1 and MG2 trade energy
# 2. Price signals: Each GridAgent broadcasts local price to its devices
```

### Example 2: Consensus + Setpoints

```python
config = {
    'microgrids': [...],
    'horizontal_protocol': 'consensus',  # Agree on frequency setpoint
    'vertical_protocol': 'setpoint',     # Enforce setpoints on devices
}

# Execution flow:
# 1. Consensus: GridAgents agree on target frequency (e.g., 60.0 Hz)
# 2. Setpoints: Each GridAgent computes device setpoints to achieve target
```

---

## Best Practices

### 1. **Start Simple**
- Begin with `NoProtocol` or `NoHorizontalProtocol`
- Add coordination only when needed

### 2. **Test Protocols Independently**
- Unit test protocols with mock agents
- Verify market clearing logic before integration

### 3. **Profile Performance**
- Horizontal protocols run every timestep (keep fast!)
- Cache expensive computations

### 4. **Document Assumptions**
- Specify what agents must include in observations
- Example: P2P trading needs `net_demand`, `marginal_cost`

### 5. **Handle Edge Cases**
- Empty bid/offer lists (no trades)
- Infeasible setpoints (clip to device limits)
- Non-convergence (timeout after max iterations)

### 6. **Validate Results**
- Check energy balance: sum of trades = 0
- Verify feasibility: all trades respect device limits
- Monitor market metrics: price, trade volume, welfare

---

## Further Reading

- **Multi-Agent RL**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
- **ADMM**: Boyd et al., "Distributed Optimization and Statistical Learning via ADMM" (2011)
- **P2P Trading**: Mengelkamp et al., "Designing microgrid energy markets" (2018)
- **Consensus**: Olfati-Saber et al., "Consensus and Cooperation in Networked Multi-Agent Systems" (2007)

---

## Get Involved

Have you built a custom protocol? Share it with the community!

- **Submit a PR**: Add your protocol to `powergrid/agents/protocols.py`
- **Publish results**: Cite PowerGrid in your papers
- **Ask questions**: Open an issue on GitHub

Happy coordinating! ⚡️🤝
