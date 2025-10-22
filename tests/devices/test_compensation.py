"""Tests for devices.compensation module (Shunt)."""

import numpy as np
import pytest

from powergrid.devices.compensation import Shunt
from powergrid.core.policies import Policy
from powergrid.agents.base import Observation


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0):
        self.action_value = action_value

    def forward(self, observation):
        """Return fixed discrete action."""
        return np.array([self.action_value])


class TestShunt:
    """Test Shunt device."""

    def test_shunt_initialization(self):
        """Test shunt initialization."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=3,
            policy=MockPolicy()
        )

        assert shunt.name == "SCB1"
        assert shunt.bus == 1
        assert shunt.q_mvar == 1.0
        assert shunt.max_step == 3
        assert shunt.type == "SCB"
        assert shunt.agent_id == "SCB1"

    def test_shunt_action_space(self):
        """Test shunt has discrete action space."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            policy=MockPolicy()
        )

        assert shunt.action.ncats == 3  # 0, 1, 2
        assert shunt.action.dim_d == 1

    def test_shunt_state_initialization(self):
        """Test shunt state is initialized correctly."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            policy=MockPolicy()
        )

        assert shunt.state.max_step == 2
        assert shunt.state.step.shape == (3,)  # max_step + 1
        assert shunt.state.step.dtype == np.float32

    def test_shunt_update_state(self):
        """Test shunt state update from action."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            policy=MockPolicy(action_value=1)
        )

        # Set action
        shunt.action.d = np.array([1], dtype=np.int32)

        # Update state
        shunt.update_state()

        # Check one-hot encoding
        expected_step = np.array([0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(shunt.state.step, expected_step)
        assert shunt._current_step == 1

    def test_shunt_update_cost_safety_no_change(self):
        """Test cost when step doesn't change."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            switching_cost=10.0,
            policy=MockPolicy()
        )

        # Initialize
        shunt.reset_device()
        shunt.action.d = np.array([0], dtype=np.int32)
        shunt.update_state()
        shunt.update_cost_safety()

        # No change from initial state
        assert shunt.cost == 0.0
        assert shunt.safety == 0.0

    def test_shunt_update_cost_safety_with_change(self):
        """Test cost when step changes."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            switching_cost=10.0,
            policy=MockPolicy()
        )

        # Initialize at step 0
        shunt.reset_device()
        shunt._last_step = 0

        # Change to step 1
        shunt.action.d = np.array([1], dtype=np.int32)
        shunt.update_state()
        shunt.update_cost_safety()

        # Should incur switching cost
        assert shunt.cost == 10.0
        assert shunt.safety == 0.0

    def test_shunt_reset(self):
        """Test shunt reset."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            policy=MockPolicy()
        )

        # Modify state
        shunt.action.d = np.array([2], dtype=np.int32)
        shunt.update_state()
        shunt.cost = 100.0

        # Reset
        shunt.reset_device()

        # Check reset state
        expected_step = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(shunt.state.step, expected_step)
        assert shunt._last_step == 0
        assert shunt.cost == 0.0
        assert shunt.safety == 0.0

    def test_shunt_repr(self):
        """Test shunt string representation."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            policy=MockPolicy()
        )

        repr_str = repr(shunt)

        assert "Shunt" in repr_str
        assert "SCB1" in repr_str
        assert "1.0" in repr_str

    def test_shunt_full_lifecycle(self):
        """Test full shunt lifecycle."""
        policy = MockPolicy(action_value=1)
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=2,
            switching_cost=5.0,
            policy=policy
        )

        # Reset
        shunt.reset()
        assert shunt.cost == 0.0

        # Observe
        obs = shunt.observe()
        assert isinstance(obs, Observation)

        # Act - select step 1
        action = shunt.act(obs)
        assert action[0] == 1

        # Update state
        shunt.update_state()
        assert shunt._current_step == 1

        # Update cost - should have switching cost
        shunt.update_cost_safety()
        assert shunt.cost == 5.0

    def test_shunt_multiple_step_changes(self):
        """Test cost tracking across multiple step changes."""
        shunt = Shunt(
            name="SCB1",
            bus=1,
            q_mvar=1.0,
            max_step=3,
            switching_cost=10.0,
            policy=MockPolicy()
        )

        shunt.reset_device()

        # Step 0 -> 1: should cost
        shunt.action.d = np.array([1])
        shunt.update_state()
        shunt.update_cost_safety()
        assert shunt.cost == 10.0

        # Step 1 -> 1: no change, no cost
        shunt.action.d = np.array([1])
        shunt.update_state()
        shunt.update_cost_safety()
        assert shunt.cost == 0.0

        # Step 1 -> 3: should cost
        shunt.action.d = np.array([3])
        shunt.update_state()
        shunt.update_cost_safety()
        assert shunt.cost == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
