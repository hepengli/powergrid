"""Tests for core.policies module."""

import pytest
import numpy as np

from powergrid.core.policies import Policy
from powergrid.agents.base import Observation


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value
        self.reset_called = False

    def forward(self, observation):
        """Return fixed action."""
        return np.array([self.action_value])

    def reset(self):
        """Mark that reset was called."""
        self.reset_called = True


class TestPolicy:
    """Test Policy abstract base class."""

    def test_policy_forward(self):
        """Test policy forward method."""
        policy = MockPolicy(action_value=0.7)
        obs = Observation(local={"P": 1.0})

        action = policy.forward(obs)

        assert isinstance(action, np.ndarray)
        np.testing.assert_array_almost_equal(action, [0.7])

    def test_policy_reset(self):
        """Test policy reset method."""
        policy = MockPolicy()

        assert not policy.reset_called

        policy.reset()

        assert policy.reset_called

    def test_policy_multiple_forward_calls(self):
        """Test multiple forward calls."""
        policy = MockPolicy(action_value=0.5)
        obs = Observation()

        action1 = policy.forward(obs)
        action2 = policy.forward(obs)

        np.testing.assert_array_equal(action1, action2)

    def test_policy_with_different_observations(self):
        """Test policy with different observations."""
        policy = MockPolicy(action_value=0.5)

        obs1 = Observation(local={"P": 1.0})
        obs2 = Observation(local={"P": 2.0})

        action1 = policy.forward(obs1)
        action2 = policy.forward(obs2)

        # Mock policy returns same action regardless of observation
        np.testing.assert_array_equal(action1, action2)

    def test_policy_cannot_instantiate_directly(self):
        """Test that Policy abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            Policy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
