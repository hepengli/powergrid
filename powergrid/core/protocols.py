"""Coordination protocols for hierarchical multi-agent control.

This module defines vertical and horizontal coordination protocols:
- Vertical protocols: Parent → subordinate coordination (agent-owned)
- Horizontal protocols: Peer ↔ peer coordination (environment-owned)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from builtins import float

from typing import Dict, Any, Optional, List, Tuple
from ..agents.base import Agent, Observation, AgentID
import numpy as np

from powergrid.agents.base import Agent, AgentID, Observation, Message


class Protocol(ABC):
    """Base class for all coordination protocols."""

    def no_op(self) -> bool:
        """Check if this is a no-operation protocol.

        Returns:
            True if this protocol performs no coordination (NoProtocol or NoHorizontalProtocol)
        """
        return False


# =============================================================================
# VERTICAL PROTOCOLS (Agent-owned: Parent → Child)
# =============================================================================

class VerticalProtocol(Protocol):
    """Vertical coordination protocol for parent → subordinate communication.

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
        """Compute coordination signals for subordinates.

        Args:
            subordinate_observations: Observations from all subordinate agents
            parent_action: Optional action from parent's policy (e.g., price to broadcast)

        Returns:
            Dictionary mapping subordinate_id to coordination signal
            Example: {'ess1': {'price': 50.0}, 'dg1': {'price': 50.0}}
        """
        pass

    def coordinate_action(
        self,
        devices: Dict[AgentID, Agent],
        observation: Observation,
        action: Optional[Any] = None
    ) -> None:
        """Coordinate device actions (default: no-op).

        This method can be overridden to distribute actions to subordinate devices.

        Args:
            devices: Dictionary of subordinate device agents
            observation: Current observation from parent
            action: Action computed by parent agent
        """
        pass

    def coordinate_message(
        self,
        devices: Dict[AgentID, Agent],
        observation: Observation,
        action: Optional[Any] = None
    ) -> None:
        """Send coordination messages to devices (default: no-op).

        This method can be overridden to send messages based on coordination protocol.

        Args:
            devices: Dictionary of subordinate device agents
            observation: Current observation from parent
            action: Action computed by parent agent
        """
        pass


class NoProtocol(VerticalProtocol):
    """No coordination - subordinates act independently."""

    def no_op(self) -> bool:
        """NoProtocol is a no-operation protocol."""
        return True

    def coordinate(
        self,
        subordinate_observations: Dict[AgentID, Observation],
        parent_action: Optional[Any] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Return empty coordination signals."""
        return {sub_id: {} for sub_id in subordinate_observations}


class PriceSignalProtocol(VerticalProtocol):
    """Price-based coordination via marginal price signals.

    Attributes:
        price: Current electricity price ($/MWh)
    """

    def __init__(self, initial_price: float = 50.0):
        """Initialize price signal protocol.

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

    def coordinate_message(
        self,
        devices: Dict[AgentID, Agent],
        observation: Observation,
        action: Optional[Any] = None
    ) -> None:
        """Send price signal as message to all devices via mailbox.

        Broadcasts price to devices using the Agent mailbox system. Devices can read
        the price from their mailbox and use it in their local optimization.

        Args:
            devices: Dictionary of subordinate device agents
            observation: Current observation from parent
            action: Action computed by parent (can contain price update)
        """
        # Update price from action if provided
        if action is not None:
            if isinstance(action, dict):
                self.price = action.get("price", self.price)
            elif hasattr(action, "price"):
                self.price = action.price
            else:
                # If action is a scalar, treat it as price
                try:
                    self.price = float(action)
                except (TypeError, ValueError):
                    pass  # Keep current price if conversion fails

        # Send price message to all devices via mailbox

        if not devices:
            return

        # Create parent/coordinator ID for message sender
        parent_id = f"price_coordinator"

        # Create and send message to each device
        for device_id, device in devices.items():
            message = Message(
                sender=parent_id,
                content={"price": self.price, "type": "price_signal"},
                recipient=device_id,
                timestamp=observation.timestamp if observation else 0.0
            )
            device.receive_message(message)


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

    def coordinate_action(
        self,
        devices: Dict[AgentID, Agent],
        observation: Observation,
        action: Optional[Any] = None
    ) -> None:
        """Distribute actions (setpoints) directly to devices.

        In centralized setpoint control, the parent computes an action vector
        that contains setpoints for each device. This method distributes those
        setpoints to the devices.

        Args:
            devices: Dictionary of subordinate device agents
            observation: Current observation from parent (unused)
            action: Action vector from parent (numpy array or dict)
        """
        if action is None:
            return

        # Convert action to device setpoints
        device_list = list(devices.values())

        if isinstance(action, dict):
            # Action is already a dict of {device_id: setpoint}
            for device_id, setpoint in action.items():
                if device_id in devices:
                    devices[device_id].act(
                        devices[device_id].observation,
                        given_action=setpoint
                    )
        else:
            # Action is a numpy array - split among devices
            import numpy as np
            action = np.asarray(action)
            offset = 0
            for device in device_list:
                # Get device action size
                action_size = device.action.dim_c + device.action.dim_d
                device_action = action[offset:offset + action_size]
                device.act(device.observation, given_action=device_action)
                offset += action_size

    def coordinate_message(
        self,
        devices: Dict[AgentID, Agent],
        observation: Observation,
        action: Optional[Any] = None
    ) -> None:
        """Send setpoint assignments as messages to devices via mailbox.

        In setpoint control, the coordinator sends explicit power setpoint commands
        to each device. This is complementary to coordinate_action which directly
        applies the actions.

        Args:
            devices: Dictionary of subordinate device agents
            observation: Current observation from parent
            action: Action/setpoints computed by parent
        """
        if action is None or not devices:
            return

        parent_id = "setpoint_coordinator"

        if isinstance(action, dict):
            # Action is dict of {device_id: setpoint}
            for device_id, setpoint in action.items():
                if device_id in devices:
                    message = Message(
                        sender=parent_id,
                        content={"setpoint": setpoint, "type": "setpoint_command"},
                        recipient=device_id,
                        timestamp=observation.timestamp if observation else 0.0
                    )
                    devices[device_id].receive_message(message)
        else:
            # Action is numpy array - send portion to each device
            action = np.asarray(action)
            offset = 0
            device_list = list(devices.values())

            for device in device_list:
                action_size = device.action.dim_c + device.action.dim_d
                device_setpoint = action[offset:offset + action_size]

                message = Message(
                    sender=parent_id,
                    content={"setpoint": device_setpoint.tolist(), "type": "setpoint_command"},
                    recipient=device.agent_id,
                    timestamp=observation.timestamp if observation else 0.0
                )
                device.receive_message(message)
                offset += action_size


# =============================================================================
# HORIZONTAL PROTOCOLS (Environment-owned: Peer ↔ Peer)
# =============================================================================

class HorizontalProtocol(Protocol):
    """Horizontal coordination protocol for peer-to-peer communication.

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
        """Coordinate peer agents (requires global view).

        Args:
            agents: All participating agents
            observations: Observations from all agents
            topology: Optional network topology (e.g., adjacency matrix for trades)

        Returns:
            Dictionary mapping agent_id to coordination signal
            Example: {'MG1': {'trades': [...]}, 'MG2': {'trades': [...]}}
        """
        pass

    def coordinate_actions(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        actions: Dict[AgentID, Any],
        net: Optional[Any] = None
    ) -> None:
        """Coordinate peer agents' actions (default: no-op).

        Horizontal protocols can modify or coordinate actions across peer agents.
        For example, P2P trading might adjust power setpoints based on trades.

        Args:
            agents: Dictionary of peer agents
            observations: Observations from all agents
            actions: Actions computed by each agent
            net: Optional network object (e.g., PandaPower network)
        """
        pass


class NoHorizontalProtocol(HorizontalProtocol):
    """No peer coordination - agents act independently."""

    def no_op(self) -> bool:
        """NoHorizontalProtocol is a no-operation protocol."""
        return True

    def coordinate(
        self,
        agents: Dict[AgentID, Agent],
        observations: Dict[AgentID, Observation],
        topology: Optional[Dict] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Return empty signals for all agents."""
        return {aid: {} for aid in agents}


class PeerToPeerTradingProtocol(HorizontalProtocol):
    """Peer-to-peer energy trading marketplace.

    Agents submit bids/offers based on their net demand and marginal cost.
    The environment (acting as market auctioneer) clears the market and
    sends trade confirmations back to agents.

    Attributes:
        trading_fee: Transaction fee as fraction of trade price
    """

    def __init__(self, trading_fee: float = 0.01):
        """Initialize P2P trading protocol.

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
        """Simple market clearing algorithm.

        Args:
            bids: Dictionary of buyer bids {agent_id: {'quantity', 'max_price'}}
            offers: Dictionary of seller offers {agent_id: {'quantity', 'min_price'}}

        Returns:
            List of trades as (buyer_id, seller_id, quantity, price) tuples
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
    """Distributed consensus via gossip algorithm.

    Agents iteratively average their values with neighbors until convergence.
    Useful for coordinated frequency regulation or voltage control.

    Attributes:
        max_iterations: Maximum gossip iterations
        tolerance: Convergence threshold
    """

    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01):
        """Initialize consensus protocol.

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
