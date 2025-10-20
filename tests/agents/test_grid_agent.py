"""Unit tests for GridAgent."""

import pytest
import numpy as np

from powergrid.agents.grid_agent import (
    GridAgent,
    NoProtocol,
    PriceSignalProtocol,
    SetpointProtocol,
)
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.base import Observation
from powergrid.devices.storage import ESS
from powergrid.devices.generator import DG


class TestProtocols:
    """Tests for coordination protocols."""

    def test_no_protocol(self):
        """Test NoProtocol (independent agents)."""
        protocol = NoProtocol()

        obs = {
            "agent_1": Observation(local={"P": 1.0}),
            "agent_2": Observation(local={"P": 2.0}),
        }

        signals = protocol.coordinate(obs)

        assert "agent_1" in signals
        assert "agent_2" in signals
        assert signals["agent_1"] == {}
        assert signals["agent_2"] == {}

    def test_price_signal_protocol(self):
        """Test price signal protocol."""
        protocol = PriceSignalProtocol(initial_price=50.0)

        obs = {
            "agent_1": Observation(local={"P": 1.0}),
            "agent_2": Observation(local={"P": 2.0}),
        }

        # No coordinator action - use initial price
        signals = protocol.coordinate(obs)

        assert signals["agent_1"]["price"] == 50.0
        assert signals["agent_2"]["price"] == 50.0

        # With coordinator action - update price
        signals = protocol.coordinate(obs, coordinator_action={"price": 75.0})

        assert signals["agent_1"]["price"] == 75.0
        assert signals["agent_2"]["price"] == 75.0

    def test_price_signal_scalar_action(self):
        """Test price signal with scalar action."""
        protocol = PriceSignalProtocol()

        obs = {"agent_1": Observation()}

        signals = protocol.coordinate(obs, coordinator_action=100.0)

        assert signals["agent_1"]["price"] == 100.0

    def test_setpoint_protocol(self):
        """Test setpoint protocol."""
        protocol = SetpointProtocol()

        obs = {
            "agent_1": Observation(),
            "agent_2": Observation(),
        }

        # No coordinator action
        signals = protocol.coordinate(obs)
        assert signals["agent_1"] == {}
        assert signals["agent_2"] == {}

        # With setpoints
        coordinator_action = {
            "agent_1": 0.5,  # P setpoint
            "agent_2": -0.3,
        }
        signals = protocol.coordinate(obs, coordinator_action)

        assert signals["agent_1"]["setpoint"] == 0.5
        assert signals["agent_2"]["setpoint"] == -0.3


class TestGridAgent:
    """Tests for GridAgent."""

    @pytest.fixture
    def subordinates(self):
        """Create subordinate device agents."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
        )

        dg = DG(
            name="dg_1",
            bus="806",
            min_p_mw=0.0,
            max_p_mw=0.5,
        )

        return [
            DeviceAgent(device=ess),
            DeviceAgent(device=dg),
        ]

    def test_coordinator_initialization(self, subordinates):
        """Test coordinator initialization."""
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
            protocol=NoProtocol(),
        )

        assert coordinator.agent_id == "mg_controller"
        assert coordinator.level == 2  # Grid level
        assert len(coordinator.subordinates) == 2
        assert "ess_1" in coordinator.subordinates
        assert "dg_1" in coordinator.subordinates

    def test_coordinator_observe(self, subordinates):
        """Test coordinator observation collection."""
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
        )

        global_state = {
            "bus_vm": {800: 1.05, "806": 1.03},
            "bus_va": {800: 0.0, "806": 0.1},
            "converged": True,
            "dataset": {"price": 50.0},
        }

        obs = coordinator.observe(global_state)

        # Should contain subordinate states
        assert "subordinate_states" in obs.local
        assert "ess_1" in obs.local["subordinate_states"]
        assert "dg_1" in obs.local["subordinate_states"]

        # Should have global info from subordinates
        assert "converged" in obs.global_info

    def test_coordinator_act_with_protocol(self, subordinates):
        """Test coordinator action with protocol."""
        protocol = PriceSignalProtocol(initial_price=50.0)
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
            protocol=protocol,
        )

        global_state = {
            "bus_vm": {800: 1.05},
            "converged": True,
        }

        obs = coordinator.observe(global_state)
        coordinator.act(obs)

        # Check that subordinates received price signal
        for agent in coordinator.subordinates.values():
            assert len(agent.mailbox) == 1
            msg = agent.mailbox[0]
            assert msg.sender == "mg_controller"
            assert "price" in msg.content
            assert msg.content["price"] == 50.0

    def test_coordinator_reset(self, subordinates):
        """Test coordinator reset."""
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
        )

        # Add some state
        coordinator.update_timestep(10.0)

        # Reset
        coordinator.reset(seed=42)

        # Check reset
        assert coordinator._timestep == 0.0
        assert len(coordinator.mailbox) == 0

        # Subordinates should also be reset
        for agent in coordinator.subordinates.values():
            assert agent._timestep == 0.0

    def test_coordinator_get_subordinate_actions(self, subordinates):
        """Test getting actions from subordinates."""
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
        )

        observations = {
            "ess_1": Observation(local={"soc": 0.5}),
            "dg_1": Observation(local={"P": 0.3}),
        }

        actions = coordinator.get_subordinate_actions(observations)

        assert "ess_1" in actions
        assert "dg_1" in actions
        # Actions should be valid (from random policy)
        assert actions["ess_1"] is not None
        assert actions["dg_1"] is not None

    def test_coordinator_centralized_mode(self, subordinates):
        """Test centralized action space."""
        coordinator = GridAgent(
            agent_id="mg_controller",
            subordinates=subordinates,
            centralized=True,
        )

        # In centralized mode, action space should be flattened
        from gymnasium.spaces import Box
        assert isinstance(coordinator.action_space, Box)
        # Should have dimension > 1 (aggregated from subordinates)
        assert coordinator.action_space.shape[0] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
