"""Tests for devices.generator module (DG and RES)."""

import numpy as np
import pytest

from powergrid.devices.generator import DG, RES
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


class TestDG:
    """Test DG (Distributed Generator) device."""

    def test_dg_initialization(self):
        """Test DG initialization."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            policy=MockPolicy()
        )

        assert dg.name == "DG1"
        assert dg.bus == "bus1"
        assert dg.min_p_mw == 0.0
        assert dg.max_p_mw == 2.0
        assert dg.type == "fossil"

    def test_dg_with_custom_type(self):
        """Test DG with custom generator type."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            type="hydro",
            policy=MockPolicy()
        )

        assert dg.type == "hydro"

    def test_dg_with_q_control(self):
        """Test DG with reactive power control."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            sn_mva=3.0,
            policy=MockPolicy()
        )

        # Should compute Q limits from sn_mva
        assert not np.isnan(dg.min_q_mvar)
        assert not np.isnan(dg.max_q_mvar)
        assert dg.action.dim_c == 2  # P and Q

    def test_dg_with_unit_commitment(self):
        """Test DG with unit commitment enabled."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            startup_time=2,
            shutdown_time=1,
            startup_cost=100.0,
            shutdown_cost=50.0,
            policy=MockPolicy()
        )

        assert dg.startup_time == 2
        assert dg.shutdown_time == 1
        assert dg.startup_cost == 100.0
        assert dg.shutdown_cost == 50.0
        assert hasattr(dg.state, "shutting")
        assert hasattr(dg.state, "starting")

    def test_dg_action_space_p_only(self):
        """Test action space with P control only."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            policy=MockPolicy()
        )

        assert dg.action.dim_c == 1
        assert dg.action.dim_d == 0  # No UC

    def test_dg_action_space_with_uc(self):
        """Test action space with unit commitment."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            startup_time=2,
            policy=MockPolicy()
        )

        assert dg.action.dim_c == 1  # P
        assert dg.action.dim_d == 1  # UC
        assert dg.action.ncats == 2  # On/off

    def test_dg_update_state_simple(self):
        """Test simple state update without UC."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            policy=MockPolicy()
        )

        dg.action.c = np.array([1.5], dtype=np.float32)
        dg.update_state()

        assert dg.state.P == 1.5

    def test_dg_update_state_with_pq(self):
        """Test state update with P and Q."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            sn_mva=3.0,
            policy=MockPolicy([1.5, 0.5])
        )

        dg.action.c = np.array([1.5, 0.5], dtype=np.float32)
        dg.update_state()

        assert dg.state.P == 1.5
        assert dg.state.Q == 0.5

    def test_dg_uc_startup(self):
        """Test unit commitment startup sequence."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            startup_time=2,
            startup_cost=100.0,
            policy=MockPolicy()
        )

        # Initialize off
        dg.reset_device()
        dg.state.on = 0
        dg.state.starting = 0

        # Command to start
        dg.action.d = np.array([1], dtype=np.int32)  # Start command

        # First update - starting
        dg.update_state()
        assert dg.state.on == 0
        assert dg.state.starting == 1

        # Second update - still starting
        dg.update_state()
        assert dg.state.on == 0
        assert dg.state.starting == 2

        # Third update - should be on
        dg.update_state()
        assert dg.state.on == 1
        assert dg.state.starting == 0
        assert dg.uc_cost == 100.0

    def test_dg_uc_shutdown(self):
        """Test unit commitment shutdown sequence."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            startup_time=2,
            shutdown_time=1,
            shutdown_cost=50.0,
            policy=MockPolicy()
        )

        # Initialize on
        dg.reset_device()
        assert dg.state.on == 1

        # Command to shutdown
        dg.action.d = np.array([0], dtype=np.int32)

        # First update - shutting
        dg.update_state()
        assert dg.state.on == 1
        assert dg.state.shutting == 1

        # Second update - should be off
        dg.update_state()
        assert dg.state.on == 0
        assert dg.state.shutting == 0
        assert dg.uc_cost == 50.0

    def test_dg_update_cost_safety(self):
        """Test cost and safety calculations."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            cost_curve_coefs=(1.0, 2.0, 3.0),
            dt=1.0,
            policy=MockPolicy()
        )

        dg.state.P = 1.0
        dg.state.on = 1
        dg.update_cost_safety()

        # Cost should be positive
        assert dg.cost > 0
        assert dg.safety >= 0

    def test_dg_reset(self):
        """Test DG reset."""
        dg = DG(
            name="DG1",
            bus="bus1",
            min_p_mw=0.0,
            max_p_mw=2.0,
            startup_time=2,
            policy=MockPolicy()
        )

        # Modify state
        dg.state.P = 1.5
        dg.state.on = 0
        dg.cost = 100.0

        # Reset
        dg.reset_device()

        assert dg.state.P == 0.0
        assert dg.state.on == 1  # Should reset to on
        assert dg.state.shutting == 0
        assert dg.state.starting == 0
        assert dg.cost == 0.0


class TestRES:
    """Test RES (Renewable Energy Source) device."""

    def test_res_initialization_solar(self):
        """Test RES initialization for solar."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            policy=MockPolicy()
        )

        assert res.name == "Solar1"
        assert res.bus == "bus1"
        assert res.type == "solar"
        assert res.sn_mva == 2.0
        assert res.max_p_mw == 2.0

    def test_res_initialization_wind(self):
        """Test RES initialization for wind."""
        res = RES(
            name="Wind1",
            bus="bus1",
            sn_mva=3.0,
            source="wind",
            policy=MockPolicy()
        )

        assert res.type == "wind"

    def test_res_invalid_source(self):
        """Test RES rejects invalid source type."""
        with pytest.raises(AssertionError):
            RES(
                name="Invalid",
                bus="bus1",
                sn_mva=2.0,
                source="nuclear",
                policy=MockPolicy()
            )

    def test_res_with_q_control(self):
        """Test RES with reactive power control."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            max_q_mvar=1.0,
            min_q_mvar=-1.0,
            policy=MockPolicy()
        )

        assert res.max_q_mvar == 1.0
        assert res.min_q_mvar == -1.0
        assert res.action.dim_c == 1  # Q only

    def test_res_update_state_with_scaling(self):
        """Test RES state update with scaling factor."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            policy=MockPolicy()
        )

        # 70% solar irradiance
        res.update_state(scaling=0.7)

        assert res.state.P == 1.4  # 70% of 2.0 MW

    def test_res_update_state_with_q(self):
        """Test RES state update with Q control."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            max_q_mvar=1.0,
            min_q_mvar=-1.0,
            policy=MockPolicy(0.3)
        )

        res.action.c = np.array([0.3], dtype=np.float32)
        res.update_state(scaling=0.8)

        assert res.state.P == 1.6  # 80% of 2.0 MW
        np.testing.assert_almost_equal(res.state.Q, 0.3, decimal=5)

    def test_res_update_cost_safety(self):
        """Test RES safety calculation."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            max_q_mvar=1.0,
            min_q_mvar=-1.0,
            dt=1.0,
            policy=MockPolicy()
        )

        # Set state within limits
        res.state.P = 1.5
        res.state.Q = 0.5
        res.action.c = np.array([0.5], dtype=np.float32)
        res.update_cost_safety()

        # No safety violation
        assert res.safety >= 0

    def test_res_update_cost_safety_exceeding_rating(self):
        """Test RES safety penalty when exceeding rating."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            max_q_mvar=2.0,
            min_q_mvar=-2.0,
            dt=1.0,
            policy=MockPolicy()
        )

        # Set state exceeding sn_mva rating
        res.state.P = 1.8
        res.state.Q = 1.8  # S = sqrt(1.8^2 + 1.8^2) = 2.55 > 2.0
        res.action.c = np.array([1.8], dtype=np.float32)
        res.update_cost_safety()

        # Should have safety penalty
        assert res.safety > 0

    def test_res_reset(self):
        """Test RES reset."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            max_q_mvar=1.0,
            min_q_mvar=-1.0,
            policy=MockPolicy()
        )

        # Modify state
        res.state.P = 1.5
        res.state.Q = 0.5
        res.cost = 100.0

        # Reset
        res.reset_device()

        assert res.state.P == 0.0
        assert res.state.Q == 0.0
        assert res.cost == 0.0
        assert res.safety == 0.0

    def test_res_without_q_control(self):
        """Test RES without Q control (action_callback mode)."""
        res = RES(
            name="Solar1",
            bus="bus1",
            sn_mva=2.0,
            source="solar",
            policy=MockPolicy()
        )

        # Should be in action_callback mode with dummy discrete action
        assert res.action_callback
        assert res.action.dim_c == 0
        assert res.action.dim_d == 1
        assert res.action.ncats == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
