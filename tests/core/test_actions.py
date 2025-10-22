"""Tests for core.actions module."""

import numpy as np
import pytest

from powergrid.core.actions import Action


class TestAction:
    """Test Action dataclass."""

    def test_action_initialization_default(self):
        """Test action initialization with default values."""
        action = Action()

        assert isinstance(action.c, np.ndarray)
        assert action.c.dtype == np.float32
        assert len(action.c) == 0

        assert isinstance(action.d, np.ndarray)
        assert action.d.dtype == np.int32
        assert len(action.d) == 0

        assert action.dim_c == 0
        assert action.dim_d == 0
        assert action.ncats == 0
        assert action.range is None

    def test_action_continuous_setup(self):
        """Test action setup for continuous control."""
        action = Action()
        action.dim_c = 2
        action.range = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)

        assert action.dim_c == 2
        assert action.range.shape == (2, 2)

    def test_action_discrete_setup(self):
        """Test action setup for discrete control."""
        action = Action()
        action.dim_d = 1
        action.ncats = 5

        assert action.dim_d == 1
        assert action.ncats == 5

    def test_sample_continuous(self):
        """Test sampling continuous actions."""
        action = Action()
        action.dim_c = 2
        action.range = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)

        action.sample()

        assert action.c.shape == (2,)
        assert action.c.dtype == np.float32
        assert 0.0 <= action.c[0] <= 1.0
        assert -1.0 <= action.c[1] <= 1.0

    def test_sample_discrete(self):
        """Test sampling discrete actions."""
        action = Action()
        action.dim_d = 1
        action.ncats = 5

        action.sample()

        assert action.d.shape == (1,)
        assert action.d.dtype == np.int32
        assert 0 <= action.d[0] < 5

    def test_sample_mixed_actions(self):
        """Test sampling both continuous and discrete actions."""
        action = Action()
        action.dim_c = 1
        action.range = np.array([[0.0], [1.0]], dtype=np.float32)
        action.dim_d = 1
        action.ncats = 3

        action.sample()

        assert action.c.shape == (1,)
        assert 0.0 <= action.c[0] <= 1.0
        assert action.d.shape == (1,)
        assert 0 <= action.d[0] < 3

    def test_sample_without_range_does_nothing(self):
        """Test sampling continuous without range specified."""
        action = Action()
        action.dim_c = 2
        # Don't set range

        action.sample()

        # Should remain empty since no range specified
        assert len(action.c) == 0

    def test_sample_without_ncats_does_nothing(self):
        """Test sampling discrete without ncats specified."""
        action = Action()
        action.dim_d = 1
        # Don't set ncats

        action.sample()

        # Should remain empty since no ncats specified
        assert len(action.d) == 0

    def test_continuous_action_bounds(self):
        """Test continuous action respects specified bounds."""
        action = Action()
        action.dim_c = 3
        action.range = np.array(
            [[0.0, 5.0, -10.0], [10.0, 15.0, 10.0]], dtype=np.float32
        )

        # Sample multiple times to check consistency
        for _ in range(10):
            action.sample()
            assert 0.0 <= action.c[0] <= 10.0
            assert 5.0 <= action.c[1] <= 15.0
            assert -10.0 <= action.c[2] <= 10.0

    def test_action_modification(self):
        """Test modifying action values."""
        action = Action()
        action.dim_c = 2
        action.range = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        action.sample()

        # Modify continuous action
        action.c[0] = 0.5
        action.c[1] = 0.8

        np.testing.assert_array_almost_equal(action.c, [0.5, 0.8])

    def test_action_discrete_modification(self):
        """Test modifying discrete action values."""
        action = Action()
        action.dim_d = 1
        action.ncats = 5
        action.sample()

        # Modify discrete action
        action.d[0] = 3

        assert action.d[0] == 3

    def test_action_range_shape(self):
        """Test action range has correct shape."""
        action = Action()
        action.dim_c = 3
        action.range = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32
        )

        assert action.range.shape == (2, 3)
        assert action.range[0, 0] == 0.0  # Lower bound dim 0
        assert action.range[1, 0] == 3.0  # Upper bound dim 0

    def test_action_dtype_enforcement(self):
        """Test action maintains correct dtypes."""
        action = Action()
        action.dim_c = 1
        action.range = np.array([[0.0], [1.0]], dtype=np.float32)
        action.dim_d = 1
        action.ncats = 3

        action.sample()

        assert action.c.dtype == np.float32
        assert action.d.dtype == np.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
