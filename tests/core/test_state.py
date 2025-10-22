"""Tests for core.state module."""

import numpy as np
import pytest

from powergrid.core.state import DeviceState


class TestDeviceState:
    """Test DeviceState dataclass."""

    def test_device_state_initialization_default(self):
        """Test device state initialization with default values."""
        state = DeviceState()

        assert state.P == 0.0
        assert state.Q == 0.0
        assert state.on == 1
        assert state.Pmax is None
        assert state.Pmin is None
        assert state.soc is None

    def test_device_state_custom_values(self):
        """Test device state with custom values."""
        state = DeviceState(
            P=1.5,
            Q=0.5,
            on=0,
            Pmax=2.0,
            Pmin=0.0,
            soc=0.8
        )

        assert state.P == 1.5
        assert state.Q == 0.5
        assert state.on == 0
        assert state.Pmax == 2.0
        assert state.Pmin == 0.0
        assert state.soc == 0.8

    def test_as_vector_empty(self):
        """Test as_vector with minimal state."""
        state = DeviceState()

        vec = state.as_vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        # Should be empty since no optional fields are set
        assert len(vec) == 0

    def test_as_vector_with_power_limits(self):
        """Test as_vector includes P and Q when limits are set."""
        state = DeviceState(
            P=1.5,
            Q=0.5,
            Pmax=2.0,
            Qmax=1.0
        )

        vec = state.as_vector()

        # Should include P and Q
        assert len(vec) == 2
        assert vec[0] == 1.5  # P
        assert vec[1] == 0.5  # Q

    def test_as_vector_with_soc(self):
        """Test as_vector includes SOC for storage devices."""
        state = DeviceState(
            P=1.0,
            Q=0.5,
            Pmax=2.0,
            Qmax=1.0,
            soc=0.8
        )

        vec = state.as_vector()

        # P, Q, soc
        assert len(vec) == 3
        assert vec[2] == 0.8  # SOC

    def test_as_vector_with_uc_state(self):
        """Test as_vector includes unit commitment state."""
        state = DeviceState(
            P=1.0,
            Q=0.5,
            on=1,
            Pmax=2.0,
            Qmax=1.0,
            shutting=0,
            starting=0
        )

        vec = state.as_vector()

        # P, Q, on_state (2 elements), shutting, starting
        assert len(vec) == 6
        assert vec[2] == 0.0  # on_state[0] = off
        assert vec[3] == 1.0  # on_state[1] = on
        assert vec[4] == 0.0  # shutting
        assert vec[5] == 0.0  # starting

    def test_as_vector_uc_state_off(self):
        """Test as_vector with unit off."""
        state = DeviceState(
            P=0.0,
            Q=0.0,
            on=0,
            Pmax=2.0,
            Qmax=1.0,
            shutting=0,
            starting=0
        )

        vec = state.as_vector()

        # Check on_state one-hot encoding
        assert vec[2] == 1.0  # on_state[0] = off
        assert vec[3] == 0.0  # on_state[1] = on

    def test_as_vector_with_shunt_state(self):
        """Test as_vector with shunt (stepped) device."""
        state = DeviceState(
            max_step=3,
            step=np.array([0, 0, 1, 0], dtype=np.float32)  # Step 2 active
        )

        vec = state.as_vector()

        # Should include the step one-hot vector
        assert len(vec) == 4
        np.testing.assert_array_equal(vec, [0, 0, 1, 0])

    def test_as_vector_with_shunt_none_step(self):
        """Test as_vector with shunt when step is None."""
        state = DeviceState(max_step=2)
        # step is None

        vec = state.as_vector()

        # Should create zero vector
        assert len(vec) == 3
        np.testing.assert_array_equal(vec, [0, 0, 0])

    def test_as_vector_with_transformer_state(self):
        """Test as_vector with transformer tap position."""
        state = DeviceState(
            tap_min=-2,
            tap_max=2,
            tap_position=1
        )

        vec = state.as_vector()

        # tap positions: -2, -1, 0, 1, 2 (5 positions)
        # Position 1 is at index 3
        assert len(vec) == 5
        expected = np.array([0, 0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_as_vector_with_transformer_loading(self):
        """Test as_vector includes transformer loading."""
        state = DeviceState(
            tap_min=0,
            tap_max=2,
            tap_position=1,
            loading_percentage=75.0
        )

        vec = state.as_vector()

        # 3 tap positions + loading
        assert len(vec) == 4
        assert vec[-1] == 0.75  # Loading normalized to [0, 1]

    def test_as_vector_with_price(self):
        """Test as_vector includes price information."""
        state = DeviceState(
            P=1.0,
            Pmax=2.0,
            price=50.0
        )

        vec = state.as_vector()

        # P, price
        assert len(vec) == 2
        assert vec[0] == 1.0  # P
        assert vec[1] == 0.5  # price / 100

    def test_as_vector_comprehensive(self):
        """Test as_vector with multiple state components."""
        state = DeviceState(
            P=1.5,
            Q=0.5,
            on=1,
            Pmax=2.0,
            Qmax=1.0,
            price=75.0,
            shutting=0,
            starting=0,
            soc=0.6
        )

        vec = state.as_vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        # P, Q, price, on_state(2), shutting, starting, soc
        assert len(vec) == 8

    def test_as_vector_tap_position_clamping(self):
        """Test tap position is clamped to valid range."""
        state = DeviceState(
            tap_min=-1,
            tap_max=1,
            tap_position=10  # Out of range
        )

        vec = state.as_vector()

        # Should clamp to max position (1)
        # Positions: -1, 0, 1 (index 2 for position 1)
        assert len(vec) == 3
        assert vec[2] == 1.0

    def test_as_vector_tap_position_none(self):
        """Test tap position defaults to tap_min when None."""
        state = DeviceState(
            tap_min=-1,
            tap_max=1,
            tap_position=None
        )

        vec = state.as_vector()

        # Should default to tap_min (-1), which is index 0
        assert vec[0] == 1.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0

    def test_as_vector_consistency(self):
        """Test as_vector returns consistent results."""
        state = DeviceState(
            P=1.0,
            Q=0.5,
            Pmax=2.0,
            Qmax=1.0,
            soc=0.7
        )

        vec1 = state.as_vector()
        vec2 = state.as_vector()

        np.testing.assert_array_equal(vec1, vec2)

    def test_state_modification(self):
        """Test modifying state values."""
        state = DeviceState(P=1.0, Q=0.5)

        state.P = 2.0
        state.Q = 1.0

        assert state.P == 2.0
        assert state.Q == 1.0

    def test_state_optional_fields(self):
        """Test optional fields remain None when not set."""
        state = DeviceState()

        assert state.Pmax is None
        assert state.Pmin is None
        assert state.Qmax is None
        assert state.Qmin is None
        assert state.shutting is None
        assert state.starting is None
        assert state.soc is None
        assert state.max_step is None
        assert state.step is None
        assert state.tap_max is None
        assert state.tap_min is None
        assert state.tap_position is None
        assert state.loading_percentage is None
        assert state.price is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
