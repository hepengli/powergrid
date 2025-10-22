"""Tests for agents.device_agent module."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.base import Observation
from powergrid.core.actions import Action
from powergrid.core.state import DeviceState
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value

    def forward(self, observation):
        """Return fixed action."""
        return np.array([self.action_value])


class ConcreteDeviceAgent(DeviceAgent):
    """Concrete device agent for testing."""

    def set_action_space(self):
        """Define simple continuous action space."""
        self.action.range = np.array([[0.0], [1.0]], dtype=np.float32)
        self.action.dim_c = 1
        self.action.sample()

    def reset_device(self, *args, **kwargs):
        """Reset device state."""
        self.state.P = 0.0
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, *args, **kwargs):
        """Update device state from action."""
        self.state.P = float(self.action.c[0])

    def update_cost_safety(self, *args, **kwargs):
        """Update cost and safety metrics."""
        self.cost = abs(self.state.P) * 10.0
        self.safety = 0.0

    def get_reward(self):
        """Calculate reward."""
        return -self.cost - self.safety

    def __repr__(self):
        """String representation."""
        return f"ConcreteDeviceAgent(id={self.agent_id})"


class TestDeviceAgent:
    """Test DeviceAgent class."""

    def test_device_agent_initialization(self):
        """Test device agent initialization."""
        policy = MockPolicy()
        agent = ConcreteDeviceAgent(
            agent_id="test_device",
            policy=policy,
            device_config={"name": "test_device"}
        )

        assert agent.agent_id == "test_device"
        assert agent.level == 1
        assert agent.policy == policy
        assert isinstance(agent.state, DeviceState)
        assert isinstance(agent.action, Action)

    def test_device_agent_initialization_without_agent_id(self):
        """Test device agent uses name from config."""
        agent = ConcreteDeviceAgent(
            policy=MockPolicy(),
            device_config={"name": "device_from_config"}
        )

        assert agent.agent_id == "device_from_config"

    def test_device_agent_requires_id_or_name(self):
        """Test device agent requires either agent_id or name in config."""
        with pytest.raises(AssertionError):
            ConcreteDeviceAgent(policy=MockPolicy())

    def test_get_action_space_continuous(self):
        """Test action space construction for continuous actions."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        action_space = agent.action_space

        assert isinstance(action_space, Box)
        assert action_space.shape == (1,)
        np.testing.assert_array_equal(action_space.low, [0.0])
        np.testing.assert_array_equal(action_space.high, [1.0])

    def test_get_action_space_discrete(self):
        """Test discrete action space."""
        class DiscreteDeviceAgent(DeviceAgent):
            def set_action_space(self):
                self.action.ncats = 5
                self.action.dim_d = 1
                self.action.sample()

            def reset_device(self, **kwargs):
                pass

            def update_state(self, **kwargs):
                pass

            def update_cost_safety(self, **kwargs):
                pass

            def get_reward(self):
                return 0.0

            def __repr__(self):
                return "DiscreteDeviceAgent"

        agent = DiscreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        action_space = agent.action_space

        assert isinstance(action_space, Discrete)
        assert action_space.n == 5

    def test_get_action_space_multidimensional_continuous(self):
        """Test multi-dimensional continuous action space."""
        class MultiDimAgent(DeviceAgent):
            def set_action_space(self):
                self.action.range = np.array(
                    [[0.0, -1.0], [1.0, 1.0]], dtype=np.float32
                )
                self.action.dim_c = 2
                self.action.sample()

            def reset_device(self, **kwargs):
                pass

            def update_state(self, **kwargs):
                pass

            def update_cost_safety(self, **kwargs):
                pass

            def get_reward(self):
                return 0.0

            def __repr__(self):
                return "MultiDimAgent"

        agent = MultiDimAgent(agent_id="test", policy=MockPolicy())

        action_space = agent.action_space

        assert isinstance(action_space, Box)
        assert action_space.shape == (2,)

    def test_get_observation_space(self):
        """Test observation space construction."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        obs_space = agent.observation_space

        assert isinstance(obs_space, Box)
        assert obs_space.dtype == np.float32

    def test_reset(self):
        """Test agent reset."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        # Modify state
        agent.state.P = 10.0
        agent.cost = 100.0
        agent._timestep = 5.0
        agent.mailbox.append("message")

        # Reset
        agent.reset()

        assert agent.state.P == 0.0
        assert agent.cost == 0.0
        assert agent._timestep == 0.0
        assert len(agent.mailbox) == 0

    def test_observe(self):
        """Test observe method."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )
        agent._timestep = 5.0
        agent.state.P = 0.5

        obs = agent.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "state" in obs.local
        assert isinstance(obs.local["state"], np.ndarray)

    def test_observe_with_global_state(self):
        """Test observe with global state information."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        global_state = {"voltage": 1.05, "frequency": 60.0}
        obs = agent.observe(global_state=global_state)

        assert obs.global_info == global_state

    def test_act_with_policy(self):
        """Test act method with policy."""
        policy = MockPolicy(action_value=0.7)
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=policy
        )

        obs = Observation(local={"state": np.array([0.5])})
        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        np.testing.assert_array_almost_equal(action, [0.7])
        np.testing.assert_array_almost_equal(agent.action.c, [0.7])

    def test_act_with_given_action(self):
        """Test act with externally provided action."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        obs = Observation()
        given_action = np.array([0.3])
        action = agent.act(obs, given_action=given_action)

        assert action is given_action
        np.testing.assert_array_almost_equal(agent.action.c, [0.3])

    def test_act_without_policy_raises_error(self):
        """Test act without policy raises assertion error."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=None
        )

        obs = Observation()

        with pytest.raises(AssertionError):
            agent.act(obs)

    def test_set_device_action(self):
        """Test _set_device_action method."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        action = np.array([0.8])
        agent._set_device_action(action)

        np.testing.assert_array_almost_equal(agent.action.c, [0.8])

    def test_set_device_action_with_discrete_action_config(self):
        """Test _set_device_action with discrete action discretization."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy(),
            device_config={"discrete_action": True, "discrete_action_cats": 3}
        )

        # Should discretize continuous range [0, 1] into 3 categories
        action = np.array([1])  # Select category 1
        agent._set_device_action(action)

        # Category 1 should be middle value: 0.5
        assert 0.4 < agent.action.c[0] < 0.6

    def test_feasible_action_default(self):
        """Test feasible_action default implementation."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        # Should return None by default
        result = agent.feasible_action()
        assert result is None

    def test_repr(self):
        """Test string representation."""
        agent = ConcreteDeviceAgent(
            agent_id="test_device",
            policy=MockPolicy()
        )

        repr_str = repr(agent)

        assert "ConcreteDeviceAgent" in repr_str
        assert "test_device" in repr_str


class TestDeviceAgentIntegration:
    """Integration tests for DeviceAgent."""

    def test_full_lifecycle(self):
        """Test full agent lifecycle: reset -> observe -> act -> update."""
        policy = MockPolicy(action_value=0.6)
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=policy
        )

        # Reset
        agent.reset()
        assert agent.state.P == 0.0

        # Observe
        obs = agent.observe()
        assert isinstance(obs, Observation)

        # Act
        action = agent.act(obs)
        np.testing.assert_array_almost_equal(action, [0.6])

        # Update state
        agent.update_state()
        np.testing.assert_almost_equal(agent.state.P, 0.6, decimal=5)

        # Update cost/safety
        agent.update_cost_safety()
        assert agent.cost > 0

    def test_timestep_update(self):
        """Test timestep tracking through lifecycle."""
        agent = ConcreteDeviceAgent(
            agent_id="test",
            policy=MockPolicy()
        )

        agent.reset()
        assert agent._timestep == 0.0

        agent.update_timestep(1.0)
        obs = agent.observe()
        assert obs.timestamp == 1.0

        agent.update_timestep(2.0)
        obs = agent.observe()
        assert obs.timestamp == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
