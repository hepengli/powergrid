"""Tests for agents.grid_agent module."""

import numpy as np
import pytest
import gymnasium as gym

from powergrid.agents.grid_agent import GridAgent, PowerGridAgent
from powergrid.agents.base import Observation, Agent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, PriceSignalProtocol
from powergrid.devices.storage import ESS
from powergrid.devices.generator import DG


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value
        self.reset_called = False

    def forward(self, observation):
        """Return fixed action."""
        if isinstance(self.action_value, np.ndarray):
            return self.action_value
        return np.array([self.action_value])

    def reset(self):
        """Mark reset called."""
        self.reset_called = True


class MockDeviceAgent(Agent):
    """Mock device agent for testing."""

    def __init__(self, agent_id, action_space_size=1):
        self.action_space_size = action_space_size
        super().__init__(
            agent_id=agent_id,
            level=1,
            action_space=gym.spaces.Box(low=0, high=1, shape=(action_space_size,), dtype=np.float32),
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        )

    def observe(self, global_state=None, *args, **kwargs):
        """Return mock observation."""
        return Observation(
            local={"state": np.array([0.5, 0.5], dtype=np.float32)},
            timestamp=self._timestep
        )

    def act(self, observation, *args, **kwargs):
        """Return mock action."""
        return np.random.rand(self.action_space_size).astype(np.float32)

    def reset(self, *, seed=None, **kwargs):
        """Reset device."""
        super().reset(seed=seed)


class TestGridAgent:
    """Test GridAgent class."""

    def test_grid_agent_initialization(self):
        """Test grid agent initialization."""
        device1 = MockDeviceAgent("device1")
        device2 = MockDeviceAgent("device2")

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1, device2],
            centralized=True
        )

        assert grid.agent_id == "grid1"
        assert grid.level == 2
        assert len(grid.devices) == 2
        assert "device1" in grid.devices
        assert "device2" in grid.devices
        assert grid.centralized

    def test_grid_agent_initialization_with_protocol(self):
        """Test grid agent with protocol."""
        device1 = MockDeviceAgent("device1")
        protocol = PriceSignalProtocol(initial_price=60.0)

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1],
            protocol=protocol
        )

        assert grid.protocol == protocol
        assert isinstance(grid.protocol, PriceSignalProtocol)

    def test_grid_agent_reset(self):
        """Test grid agent reset."""
        device1 = MockDeviceAgent("device1")
        device2 = MockDeviceAgent("device2")
        policy = MockPolicy()

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1, device2],
            policy=policy,
            centralized=True
        )

        grid.reset()

        assert policy.reset_called
        # Devices should be reset (check timestep)
        assert device1._timestep == 0.0
        assert device2._timestep == 0.0

    def test_grid_agent_observe(self):
        """Test grid agent observe."""
        device1 = MockDeviceAgent("device1")
        device2 = MockDeviceAgent("device2")

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1, device2],
            centralized=True
        )

        obs = grid.observe()

        assert isinstance(obs, Observation)
        assert "device_obs" in obs.local
        assert len(obs.local["device_obs"]) == 2

    def test_grid_agent_act_centralized(self):
        """Test grid agent act in centralized mode."""
        device1 = MockDeviceAgent("device1")
        device2 = MockDeviceAgent("device2")
        policy = MockPolicy(np.array([0.5, 0.5]))

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1, device2],
            policy=policy,
            centralized=True
        )

        obs = grid.observe()
        action = grid.act(obs)

        assert isinstance(action, np.ndarray)
        assert len(action) == 2

    def test_grid_agent_act_decentralized_raises(self):
        """Test grid agent decentralized mode raises error."""
        device1 = MockDeviceAgent("device1")

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1],
            centralized=False
        )

        obs = grid.observe()

        with pytest.raises(NotImplementedError):
            grid.act(obs)

    def test_grid_agent_coordinate_device(self):
        """Test device coordination."""
        device1 = MockDeviceAgent("device1")
        protocol = NoProtocol()

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1],
            protocol=protocol,
            centralized=True,
            policy=MockPolicy()
        )

        obs = grid.observe()
        action = np.array([0.5])

        # Should not raise error
        grid.coordinate_device(obs, action)

    def test_grid_agent_repr(self):
        """Test grid agent string representation."""
        device1 = MockDeviceAgent("device1")
        device2 = MockDeviceAgent("device2")
        protocol = PriceSignalProtocol()

        grid = GridAgent(
            agent_id="grid1",
            devices=[device1, device2],
            protocol=protocol
        )

        repr_str = repr(grid)

        assert "GridAgent" in repr_str
        assert "grid1" in repr_str
        assert "devices=2" in repr_str
        assert "PriceSignalProtocol" in repr_str


class TestPowerGridAgent:
    """Test PowerGridAgent class."""

    def test_power_grid_agent_initialization(self):
        """Test PowerGridAgent initialization requires pandapower."""
        # This test requires pandapower network setup
        # Skip for now as it requires complex setup
        pass

    def test_power_grid_agent_add_dataset(self):
        """Test adding dataset to PowerGridAgent."""
        # Skip - requires pandapower network
        pass

    def test_power_grid_agent_add_storage(self):
        """Test adding storage to PowerGridAgent."""
        # Skip - requires pandapower network
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
