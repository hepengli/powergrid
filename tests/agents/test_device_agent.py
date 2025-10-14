"""Unit tests for DeviceAgent."""

import pytest
import numpy as np
from gymnasium.spaces import Box

from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.base import Observation, RandomPolicy
from powergrid.devices.storage import ESS
from powergrid.devices.generator import DG


class TestDeviceAgent:
    """Tests for DeviceAgent wrapper."""

    def test_device_agent_from_ess(self):
        """Test creating DeviceAgent from ESS device."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
            init_soc=0.5,
        )

        agent = DeviceAgent(device=ess)

        assert agent.agent_id == "ess_1"
        assert agent.level == 1
        assert agent.device == ess
        assert agent.action_space is not None
        assert agent.observation_space is not None

    def test_device_agent_action_space(self):
        """Test action space construction from device."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
        )

        agent = DeviceAgent(device=ess)

        # ESS has continuous action (P)
        assert isinstance(agent.action_space, Box)
        assert agent.action_space.shape == (1,)  # P only
        np.testing.assert_array_almost_equal(agent.action_space.low, [-0.5])
        np.testing.assert_array_almost_equal(agent.action_space.high, [0.5])

    def test_device_agent_observe(self):
        """Test observation extraction."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
            init_soc=0.5,
        )
        ess.state.P = 0.2
        ess.state.Q = 0.1

        agent = DeviceAgent(device=ess)

        global_state = {
            "bus_vm": {800: 1.05},
            "bus_va": {800: 0.5},
            "converged": True,
            "dataset": {"price": 50.0, "load": 1.0},
        }

        obs = agent.observe(global_state)

        # Check local state
        assert obs.local["P"] == 0.2
        assert obs.local["Q"] == 0.1
        assert obs.local["on"] == 1
        assert obs.local["soc"] == 0.5

        # Check global info (not partial obs)
        assert obs.global_info["bus_voltage"] == 1.05
        assert obs.global_info["bus_angle"] == 0.5
        assert obs.global_info["price"] == 50.0
        assert obs.global_info["converged"] is True

    def test_device_agent_partial_obs(self):
        """Test partial observability."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
        )

        agent = DeviceAgent(device=ess, partial_obs=True)

        global_state = {
            "bus_vm": {800: 1.05},
            "dataset": {"price": 50.0},
        }

        obs = agent.observe(global_state)

        # Local state should be present
        assert "P" in obs.local
        assert "soc" in obs.local

        # Global info should be empty (partial obs)
        assert len(obs.global_info) == 0

    def test_device_agent_act(self):
        """Test action computation."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
        )

        policy = RandomPolicy(Box(low=-0.5, high=0.5, shape=(1,)), seed=42)
        agent = DeviceAgent(device=ess, policy=policy)

        obs = Observation(local={"soc": 0.5})
        action = agent.act(obs)

        # Action should be in valid range
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert -0.5 <= action[0] <= 0.5

        # Device action should be updated
        assert ess.action.c.shape == (1,)

    def test_device_agent_reset(self):
        """Test agent reset."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
            init_soc=0.7,
        )

        agent = DeviceAgent(device=ess)

        # Reset should initialize SOC
        agent.reset(seed=42)

        # SOC should be reset (randomly within bounds)
        assert 0.0 <= ess.state.soc <= 1.0

    def test_device_agent_get_reward(self):
        """Test reward computation from device cost/safety."""
        ess = ESS(
            name="ess_1",
            bus=800,
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=1.0,
        )
        ess.cost = 10.0
        ess.safety = 2.0

        agent = DeviceAgent(device=ess)
        reward = agent.get_reward()

        # Reward = -cost - safety
        assert reward == -12.0

    def test_device_agent_with_dg(self):
        """Test DeviceAgent with DG (discrete + continuous action)."""
        dg = DG(
            name="dg_1",
            bus="806",
            min_p_mw=0.0,
            max_p_mw=0.5,
            startup_time=2,
            shutdown_time=1,
        )

        agent = DeviceAgent(device=dg)

        # DG has both continuous (P) and discrete (UC) actions
        assert agent.action_space is not None
        # Should be a Dict space with continuous and discrete
        from gymnasium.spaces import Dict as SpaceDict
        assert isinstance(agent.action_space, SpaceDict)
        assert "continuous" in agent.action_space.spaces
        assert "discrete" in agent.action_space.spaces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
