"""Comprehensive tests for envs.multi_agent.networked_grid_env module."""

import pytest
import numpy as np
import pandapower as pp
from unittest.mock import Mock, MagicMock, patch
import gymnasium as gym

from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.agents.base import Observation
from pettingzoo import ParallelEnv


class MockPowerGridAgent(PowerGridAgent):
    """Mock PowerGridAgent for testing."""

    def __init__(self, name, net=None, has_devices=True):
        self.agent_id = name
        self.name = name
        self.level = 2
        self.devices = {} if not has_devices else {"device1": Mock()}
        self.cost = 10.0
        self.safety = 0.5
        self._reset_called = False
        self._update_state_called = False
        self._update_cost_safety_called = False

    def get_device_action_spaces(self):
        """Return mock action spaces."""
        if self.devices:
            return {"device1": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)}
        return {}

    def get_grid_action_space(self):
        """Return mock action space."""
        if self.devices:
            return gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        return gym.spaces.Discrete(1)

    def get_grid_observation_space(self, net):
        """Return mock observation space."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def observe(self, net=None):
        """Return mock observation."""
        return Observation(
            local={"state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
            timestamp=0.0
        )

    def act(self, obs, given_action=None):
        """Mock act method."""
        return given_action

    def update_state(self, net, t):
        """Mock update state."""
        self._update_state_called = True

    def update_cost_safety(self, net):
        """Mock update cost/safety."""
        self._update_cost_safety_called = True

    def reset(self, seed=None):
        """Mock reset."""
        self._reset_called = True


class ConcreteNetworkedGridEnv(NetworkedGridEnv):
    """Concrete implementation for testing."""

    def __init__(self, env_config):
        self._mock_agents_created = False
        super().__init__(env_config)

    def _build_net(self):
        """Build mock network."""
        # Create simple pandapower network
        self.net = pp.create_empty_network()
        pp.create_bus(self.net, vn_kv=20., name="Bus1")
        pp.create_bus(self.net, vn_kv=20., name="Bus2")
        pp.create_ext_grid(self.net, bus=0, vm_pu=1.0, name="Grid Connection")
        pp.create_line(self.net, from_bus=0, to_bus=1, length_km=1.0,
                      std_type="NAYY 4x50 SE", name="Line1")
        self.net['converged'] = True

        # Create mock agents
        self.agent_dict = {
            "MG1": MockPowerGridAgent("MG1", self.net, has_devices=True),
            "MG2": MockPowerGridAgent("MG2", self.net, has_devices=True),
            "MG3": MockPowerGridAgent("MG3", self.net, has_devices=False),
        }
        self.possible_agents = list(self.agent_dict.keys())
        self.agents = self.possible_agents
        self.data_size = 100
        self.total_days = 4
        self._mock_agents_created = True

    def _reward_and_safety(self):
        """Mock reward computation."""
        rewards = {name: -agent.cost for name, agent in self.agent_dict.items()}
        safety = {name: agent.safety for name, agent in self.agent_dict.items()}
        return rewards, safety


class TestNetworkedGridEnv:
    """Test NetworkedGridEnv class."""

    def test_networked_grid_env_initialization(self):
        """Test environment initialization."""
        env_config = {
            "max_episode_steps": 24,
            "train": True,
            "type": "AC"
        }

        env = ConcreteNetworkedGridEnv(env_config)

        assert env.max_episode_steps == 24
        assert env.train == True
        assert env.type == "AC"
        assert len(env.agent_dict) == 3
        assert env._t == 0
        # _day is initialized in reset(), not __init__

    def test_networked_grid_env_extends_parallel_env(self):
        """Test that NetworkedGridEnv extends ParallelEnv."""
        assert issubclass(NetworkedGridEnv, ParallelEnv)

    def test_networked_grid_env_metadata(self):
        """Test NetworkedGridEnv metadata."""
        assert "name" in NetworkedGridEnv.metadata
        assert NetworkedGridEnv.metadata["name"] == "networked_grid_env"

    def test_actionable_agents_property(self):
        """Test actionable_agents property filters agents correctly."""
        env_config = {"train": True}
        env = ConcreteNetworkedGridEnv(env_config)

        actionable = env.actionable_agents

        assert len(actionable) == 2  # MG1 and MG2 have devices
        assert "MG1" in actionable
        assert "MG2" in actionable
        assert "MG3" not in actionable

    def test_reset_training_mode(self):
        """Test reset in training mode."""
        env_config = {"train": True, "max_episode_steps": 24}
        env = ConcreteNetworkedGridEnv(env_config)

        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert len(obs) == 3
        assert all(agent_id in obs for agent_id in env.agent_dict.keys())
        assert isinstance(info, dict)
        # Check agents were reset
        for agent in env.agent_dict.values():
            assert agent._reset_called

    def test_reset_test_mode(self):
        """Test reset in test mode."""
        env_config = {"train": False, "max_episode_steps": 24}
        env = ConcreteNetworkedGridEnv(env_config)

        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert env._t == 0
        assert env._day == 0

    def test_step_basic(self):
        """Test basic step functionality."""
        env_config = {"train": True}
        env = ConcreteNetworkedGridEnv(env_config)
        env.reset()

        actions = {
            "MG1": np.array([0.5, 0.5]),
            "MG2": np.array([0.3, 0.7]),
        }

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert "__all__" in terminateds
        assert "__all__" in truncateds

    def test_step_increments_timestep(self):
        """Test step increments timestep correctly."""
        env_config = {"train": True, "max_episode_steps": 24}
        env = ConcreteNetworkedGridEnv(env_config)
        env.reset()

        initial_t = env._t
        actions = {"MG1": np.array([0.5, 0.5]), "MG2": np.array([0.3, 0.7])}
        env.step(actions)

        assert env._t == initial_t + 1

    def test_step_with_shared_rewards(self):
        """Test step with reward sharing enabled."""
        env_config = {"train": True, "share_reward": True}
        env = ConcreteNetworkedGridEnv(env_config)
        env.reset()

        actions = {"MG1": np.array([0.5, 0.5]), "MG2": np.array([0.3, 0.7])}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # All agents should have the same reward (mean)
        reward_values = list(rewards.values())
        assert all(r == reward_values[0] for r in reward_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
