"""Tests for devices.storage module (ESS)."""

import numpy as np
import pytest

from powergrid.devices.storage import ESS
from powergrid.core.policies import Policy
from powergrid.agents.base import Observation


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, action_value=0.5):
        self.action_value = action_value

    def forward(self, observation):
        """Return fixed action."""
        if isinstance(self.action_value, (list, np.ndarray)):
            return np.array(self.action_value, dtype=np.float32)
        return np.array([self.action_value], dtype=np.float32)


class TestESS:
    """Test ESS (Energy Storage System) device."""

    def test_ess_initialization(self):
        """Test ESS initialization."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            policy=MockPolicy()
        )

        assert ess.name == "ESS1"
        assert ess.bus == 1
        assert ess.min_p_mw == -2.0
        assert ess.max_p_mw == 2.0
        assert ess.capacity == 10.0
        assert ess.type == "ESS"

    def test_ess_with_q_control(self):
        """Test ESS with reactive power control."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            sn_mva=3.0,
            policy=MockPolicy()
        )

        # Should compute Q limits from sn_mva
        assert not np.isnan(ess.min_q_mvar)
        assert not np.isnan(ess.max_q_mvar)
        assert ess.action.dim_c == 2  # P and Q

    def test_ess_action_space_p_only(self):
        """Test action space with P control only."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            policy=MockPolicy()
        )

        assert ess.action.dim_c == 1
        assert ess.action.range.shape == (2, 1)
        np.testing.assert_array_almost_equal(ess.action.range[0], [-2.0])
        np.testing.assert_array_almost_equal(ess.action.range[1], [2.0])

    def test_ess_soc_initialization(self):
        """Test SOC initialization."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            init_soc=0.7,
            policy=MockPolicy()
        )

        assert ess.state.soc == 0.7

    def test_ess_update_state_charging(self):
        """Test state update during charging (P > 0)."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            init_soc=0.5,
            ch_eff=0.95,
            dt=1.0,
            policy=MockPolicy()
        )

        # Charge at 1 MW for 1 hour
        ess.action.c = np.array([1.0], dtype=np.float32)
        ess.update_state()

        # SOC should increase: 0.5 + (1.0 * 0.95 * 1.0 / 10.0) = 0.595
        assert ess.state.P == 1.0
        np.testing.assert_almost_equal(ess.state.soc, 0.595)

    def test_ess_update_state_discharging(self):
        """Test state update during discharging (P < 0)."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            init_soc=0.5,
            dsc_eff=0.95,
            dt=1.0,
            policy=MockPolicy()
        )

        # Discharge at -1 MW for 1 hour
        ess.action.c = np.array([-1.0], dtype=np.float32)
        ess.update_state()

        # SOC should decrease: 0.5 + (-1.0 / 0.95 * 1.0 / 10.0) = 0.3947
        assert ess.state.P == -1.0
        np.testing.assert_almost_equal(ess.state.soc, 0.3947, decimal=4)

    def test_ess_update_state_with_q(self):
        """Test state update with P and Q control."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            sn_mva=3.0,
            policy=MockPolicy([1.0, 0.5])
        )

        ess.action.c = np.array([1.0, 0.5], dtype=np.float32)
        ess.update_state()

        assert ess.state.P == 1.0
        assert ess.state.Q == 0.5

    def test_ess_update_cost_safety(self):
        """Test cost and safety updates."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            cost_curve_coefs=(1.0, 2.0, 3.0),
            dt=1.0,
            policy=MockPolicy()
        )

        ess.state.P = 1.0
        ess.state.soc = 0.5
        ess.update_cost_safety()

        # Cost should be calculated from curve
        assert ess.cost >= 0
        assert ess.safety >= 0

    def test_ess_feasible_action(self):
        """Test feasible action clamping based on SOC."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            min_e_mwh=1.0,
            max_e_mwh=9.0,
            ch_eff=1.0,
            dsc_eff=1.0,
            dt=1.0,
            policy=MockPolicy()
        )

        # Set SOC near minimum
        ess.state.soc = 0.15  # 1.5 MWh out of 10 MWh capacity

        # Try to discharge at max power
        ess.action.c = np.array([-2.0], dtype=np.float32)

        # Apply feasibility constraint
        ess.feasible_action()

        # Should be clamped to prevent going below min_e_mwh
        # Available discharge: (0.15 - 0.1) * 10 = 0.5 MWh over 1 hour = 0.5 MW
        assert ess.action.c[0] >= -0.5

    def test_ess_feasible_action_charging(self):
        """Test feasible action clamping during charging."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            max_e_mwh=9.0,
            ch_eff=1.0,
            dt=1.0,
            policy=MockPolicy()
        )

        # Set SOC near maximum
        ess.state.soc = 0.85  # 8.5 MWh out of 10 MWh capacity

        # Try to charge at max power
        ess.action.c = np.array([2.0], dtype=np.float32)

        # Apply feasibility constraint
        ess.feasible_action()

        # Should be clamped to prevent exceeding max_e_mwh
        # Available charge: (0.9 - 0.85) * 10 = 0.5 MWh over 1 hour = 0.5 MW
        assert ess.action.c[0] <= 0.5

    def test_ess_reset(self):
        """Test ESS reset."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            min_e_mwh=2.0,
            max_e_mwh=8.0,
            policy=MockPolicy()
        )

        # Modify state
        ess.state.P = 1.5
        ess.state.soc = 0.3
        ess.cost = 100.0

        # Reset with specific SOC
        ess.reset_device(init_soc=0.6)

        assert ess.state.P == 0.0
        assert ess.state.soc == 0.6
        assert ess.cost == 0.0
        assert ess.safety == 0.0

    def test_ess_reset_random_soc(self):
        """Test ESS reset with random SOC."""
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            min_e_mwh=2.0,
            max_e_mwh=8.0,
            policy=MockPolicy()
        )

        rnd = np.random.RandomState(42)
        ess.reset_device(rnd=rnd)

        # SOC should be between min and max
        assert ess.min_soc <= ess.state.soc <= ess.max_soc

    def test_ess_full_lifecycle(self):
        """Test full ESS lifecycle."""
        policy = MockPolicy(action_value=1.0)
        ess = ESS(
            name="ESS1",
            bus=1,
            min_p_mw=-2.0,
            max_p_mw=2.0,
            capacity=10.0,
            init_soc=0.5,
            policy=policy
        )

        # Reset
        ess.reset()
        initial_soc = ess.state.soc

        # Observe
        obs = ess.observe()
        assert isinstance(obs, Observation)

        # Act - charge
        action = ess.act(obs)
        assert action[0] == 1.0

        # Update
        ess.update_state()
        assert ess.state.soc > initial_soc

        # Update cost/safety
        ess.update_cost_safety()
        assert ess.cost >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
