"""Tests for core.protocols module."""

import pytest
import numpy as np
from powergrid.core.protocols import (
    Protocol,
    VerticalProtocol,
    NoProtocol,
    PriceSignalProtocol,
    SetpointProtocol,
    HorizontalProtocol,
    NoHorizontalProtocol,
    PeerToPeerTradingProtocol,
    ConsensusProtocol,
)
from powergrid.agents.base import Agent, Observation


class MockAgent(Agent):
    """Mock agent for testing."""

    def observe(self, global_state=None, *args, **kwargs):
        return Observation(local={"value": 1.0})

    def act(self, observation, *args, **kwargs):
        return np.array([1.0])


class TestProtocol:
    """Test Protocol base class."""

    def test_protocol_is_abstract(self):
        """Test Protocol is a base class."""
        # Protocol itself has no abstract methods, so it can be instantiated
        protocol = Protocol()
        assert isinstance(protocol, Protocol)


class TestVerticalProtocol:
    """Test VerticalProtocol base class."""

    def test_vertical_protocol_is_abstract(self):
        """Test VerticalProtocol cannot be instantiated."""
        with pytest.raises(TypeError):
            VerticalProtocol()


class TestNoProtocol:
    """Test NoProtocol implementation."""

    def test_no_protocol_coordinate(self):
        """Test NoProtocol returns empty signals."""
        protocol = NoProtocol()

        obs_dict = {
            "agent1": Observation(),
            "agent2": Observation(),
        }

        signals = protocol.coordinate(obs_dict)

        assert len(signals) == 2
        assert signals["agent1"] == {}
        assert signals["agent2"] == {}


class TestPriceSignalProtocol:
    """Test PriceSignalProtocol implementation."""

    def test_price_protocol_initialization(self):
        """Test price protocol initialization."""
        protocol = PriceSignalProtocol(initial_price=60.0)

        assert protocol.price == 60.0

    def test_price_protocol_default_price(self):
        """Test default price."""
        protocol = PriceSignalProtocol()

        assert protocol.price == 50.0

    def test_price_protocol_broadcast(self):
        """Test price broadcast to all subordinates."""
        protocol = PriceSignalProtocol(initial_price=55.0)

        obs_dict = {
            "device1": Observation(),
            "device2": Observation(),
        }

        signals = protocol.coordinate(obs_dict)

        assert len(signals) == 2
        assert signals["device1"] == {"price": 55.0}
        assert signals["device2"] == {"price": 55.0}

    def test_price_protocol_update_from_action(self):
        """Test price update from parent action."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        obs_dict = {"device1": Observation()}

        # Update price via action (scalar)
        signals = protocol.coordinate(obs_dict, parent_action=75.0)

        assert protocol.price == 75.0
        assert signals["device1"] == {"price": 75.0}

    def test_price_protocol_update_from_dict_action(self):
        """Test price update from dict action."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        obs_dict = {"device1": Observation()}

        # Update price via dict action
        signals = protocol.coordinate(obs_dict, parent_action={"price": 80.0})

        assert protocol.price == 80.0
        assert signals["device1"] == {"price": 80.0}


class TestSetpointProtocol:
    """Test SetpointProtocol implementation."""

    def test_setpoint_protocol_no_action(self):
        """Test setpoint protocol with no action."""
        protocol = SetpointProtocol()

        obs_dict = {
            "device1": Observation(),
            "device2": Observation(),
        }

        signals = protocol.coordinate(obs_dict)

        assert signals["device1"] == {}
        assert signals["device2"] == {}

    def test_setpoint_protocol_with_setpoints(self):
        """Test setpoint distribution."""
        protocol = SetpointProtocol()

        obs_dict = {
            "device1": Observation(),
            "device2": Observation(),
        }

        parent_action = {
            "device1": 100.0,
            "device2": 150.0,
        }

        signals = protocol.coordinate(obs_dict, parent_action=parent_action)

        assert signals["device1"] == {"setpoint": 100.0}
        assert signals["device2"] == {"setpoint": 150.0}

    def test_setpoint_protocol_partial_setpoints(self):
        """Test setpoint protocol with partial assignments."""
        protocol = SetpointProtocol()

        obs_dict = {
            "device1": Observation(),
            "device2": Observation(),
        }

        parent_action = {"device1": 100.0}  # Only device1 gets setpoint

        signals = protocol.coordinate(obs_dict, parent_action=parent_action)

        assert signals["device1"] == {"setpoint": 100.0}
        assert signals["device2"] == {}


class TestHorizontalProtocol:
    """Test HorizontalProtocol base class."""

    def test_horizontal_protocol_is_abstract(self):
        """Test HorizontalProtocol cannot be instantiated."""
        with pytest.raises(TypeError):
            HorizontalProtocol()


class TestNoHorizontalProtocol:
    """Test NoHorizontalProtocol implementation."""

    def test_no_horizontal_protocol(self):
        """Test no horizontal coordination."""
        protocol = NoHorizontalProtocol()

        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        obs_dict = {
            "agent1": Observation(),
            "agent2": Observation(),
        }

        signals = protocol.coordinate(agents, obs_dict)

        assert len(signals) == 2
        assert signals["agent1"] == {}
        assert signals["agent2"] == {}


class TestPeerToPeerTradingProtocol:
    """Test P2P trading protocol."""

    def test_p2p_protocol_initialization(self):
        """Test P2P protocol initialization."""
        protocol = PeerToPeerTradingProtocol(trading_fee=0.02)

        assert protocol.trading_fee == 0.02

    def test_p2p_protocol_no_trades(self):
        """Test P2P with no feasible trades."""
        protocol = PeerToPeerTradingProtocol()

        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        # Both agents have no net demand
        obs_dict = {
            "agent1": Observation(local={"net_demand": 0, "marginal_cost": 50}),
            "agent2": Observation(local={"net_demand": 0, "marginal_cost": 50}),
        }

        signals = protocol.coordinate(agents, obs_dict)

        # No trades expected
        assert len(signals) == 0

    def test_p2p_protocol_simple_trade(self):
        """Test simple P2P trade between buyer and seller."""
        protocol = PeerToPeerTradingProtocol()

        agents = {
            "buyer": MockAgent("buyer"),
            "seller": MockAgent("seller"),
        }

        obs_dict = {
            "buyer": Observation(local={"net_demand": 10, "marginal_cost": 60}),
            "seller": Observation(local={"net_demand": -10, "marginal_cost": 40}),
        }

        signals = protocol.coordinate(agents, obs_dict)

        # Should have one trade
        assert "buyer" in signals
        assert "seller" in signals
        assert len(signals["buyer"]["trades"]) == 1
        assert len(signals["seller"]["trades"]) == 1

        # Check trade details
        buyer_trade = signals["buyer"]["trades"][0]
        assert buyer_trade["counterparty"] == "seller"
        assert buyer_trade["quantity"] == 10
        assert 40 < buyer_trade["price"] < 60  # Price between marginal costs

        seller_trade = signals["seller"]["trades"][0]
        assert seller_trade["counterparty"] == "buyer"
        assert seller_trade["quantity"] == -10  # Negative for seller


class TestConsensusProtocol:
    """Test consensus protocol."""

    def test_consensus_protocol_initialization(self):
        """Test consensus protocol initialization."""
        protocol = ConsensusProtocol(max_iterations=20, tolerance=0.001)

        assert protocol.max_iterations == 20
        assert protocol.tolerance == 0.001

    def test_consensus_protocol_convergence(self):
        """Test consensus reaches agreement."""
        protocol = ConsensusProtocol(max_iterations=100, tolerance=0.01)

        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
            "agent3": MockAgent("agent3"),
        }

        obs_dict = {
            "agent1": Observation(local={"control_value": 10.0}),
            "agent2": Observation(local={"control_value": 20.0}),
            "agent3": Observation(local={"control_value": 30.0}),
        }

        signals = protocol.coordinate(agents, obs_dict)

        # All agents should converge to similar values
        values = [signals[aid]["consensus_value"] for aid in agents]

        # Check all values are close (should converge to average ~20)
        assert all(15 < v < 25 for v in values)

        # Check values are close to each other
        max_diff = max(values) - min(values)
        assert max_diff < 1.0

    def test_consensus_protocol_with_topology(self):
        """Test consensus with specific network topology."""
        protocol = ConsensusProtocol(max_iterations=100)

        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
            "agent3": MockAgent("agent3"),
        }

        obs_dict = {
            "agent1": Observation(local={"control_value": 10.0}),
            "agent2": Observation(local={"control_value": 20.0}),
            "agent3": Observation(local={"control_value": 30.0}),
        }

        # Line topology: agent1 - agent2 - agent3
        topology = {
            "adjacency": {
                "agent1": ["agent2"],
                "agent2": ["agent1", "agent3"],
                "agent3": ["agent2"],
            }
        }

        signals = protocol.coordinate(agents, obs_dict, topology=topology)

        # Should still converge
        values = [signals[aid]["consensus_value"] for aid in agents]
        max_diff = max(values) - min(values)
        assert max_diff < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
