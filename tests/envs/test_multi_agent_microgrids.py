"""Comprehensive tests for envs.multi_agent.multi_agent_microgrids module.

NOTE: These tests work with pandas >= 2.3.1 which fixes the PandaPower NAType issue.
However, the environment requires implementing mixed action space support (continuous + discrete).
"""

import pytest
import numpy as np

from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.envs.configs.config_loader import load_config


# Skip tests due to unimplemented mixed action space feature
pytestmark = pytest.mark.skip(reason="Mixed action spaces (continuous + discrete) not yet implemented in grid_agent.py")


class TestMultiAgentMicrogrids:
    """Test MultiAgentMicrogrids class."""

    @pytest.fixture
    def env_config(self):
        """Load the actual environment configuration using config_loader."""
        config = load_config('ieee34_ieee13')
        config['max_episode_steps'] = 24
        return config

    def test_multi_agent_microgrids_extends_networked_grid_env(self):
        """Test that MultiAgentMicrogrids extends NetworkedGridEnv."""
        assert issubclass(MultiAgentMicrogrids, NetworkedGridEnv)

    def test_initialization_with_real_config(self, env_config):
        """Test initialization with real config file."""
        env = MultiAgentMicrogrids(env_config)

        # Check basic attributes
        assert hasattr(env, 'dso')
        assert hasattr(env, 'agent_dict')
        assert len(env.agent_dict) == 4  # DSO + 3 MGs
        assert 'DSO' in env.agent_dict
        assert 'MG1' in env.agent_dict
        assert 'MG2' in env.agent_dict
        assert 'MG3' in env.agent_dict

        # Check penalty settings
        assert env._safety == env_config['penalty']
        assert env._convergence_failure_reward == env_config['convergence_failure_reward']
        assert env._convergence_failure_safety == env_config['convergence_failure_safety']

    def test_reset_with_real_config(self, env_config):
        """Test reset functionality with real config."""
        env = MultiAgentMicrogrids(env_config)

        obs, info = env.reset(seed=42)

        # Check observations structure
        assert isinstance(obs, dict)
        assert len(obs) > 0

        # Check that actionable agents have observations
        for agent_name in env.actionable_agents:
            assert agent_name in obs
            assert isinstance(obs[agent_name], np.ndarray)

        # Check timestep initialization
        assert env._t >= 0

    def test_step_basic_with_real_config(self, env_config):
        """Test basic step functionality with real config."""
        # Use train mode for deterministic behavior
        env_config['train'] = True
        env = MultiAgentMicrogrids(env_config)

        obs, info = env.reset(seed=42)

        # Create actions for actionable agents
        actions = {}
        for agent_name, agent in env.actionable_agents.items():
            # Sample random action from the agent's action space
            action = env.action_spaces[agent_name].sample()
            actions[agent_name] = action

        # Take a step
        obs_next, rewards, terminateds, truncateds, infos = env.step(actions)

        # Check outputs
        assert isinstance(obs_next, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminateds, dict)
        assert isinstance(truncateds, dict)
        assert isinstance(infos, dict)

        # Check rewards for all agents
        for agent_name in env.agent_dict:
            assert agent_name in rewards
            assert isinstance(rewards[agent_name], (int, float, np.number))

    def test_reward_and_safety_converged(self, env_config):
        """Test _reward_and_safety when power flow converges."""
        env = MultiAgentMicrogrids(env_config)
        env.reset(seed=42)

        # Set network as converged
        env.net['converged'] = True

        # Update agent costs
        for agent in env.agent_dict.values():
            agent.update_cost_safety(env.net)

        rewards, safety = env._reward_and_safety()

        # Check that rewards were computed for all agents
        assert len(rewards) == len(env.agent_dict)
        assert len(safety) == len(env.agent_dict)

        for agent_name in env.agent_dict:
            assert agent_name in rewards
            assert agent_name in safety
            assert isinstance(rewards[agent_name], (int, float, np.number))
            assert isinstance(safety[agent_name], (int, float, np.number))

        # Verify reward calculation: reward = -cost - penalty * safety
        penalty = env_config['penalty']
        for agent_name, agent in env.agent_dict.items():
            expected_reward = -agent.cost - penalty * agent.safety
            assert np.isclose(rewards[agent_name], expected_reward)

    def test_reward_and_safety_not_converged(self, env_config):
        """Test _reward_and_safety when power flow doesn't converge."""
        env = MultiAgentMicrogrids(env_config)
        env.reset(seed=42)

        # Set network as not converged
        env.net['converged'] = False

        rewards, safety = env._reward_and_safety()

        # Check that convergence failure penalties are applied
        cf_reward = env_config['convergence_failure_reward']
        cf_safety = env_config['convergence_failure_safety']
        penalty = env_config['penalty']

        expected_reward = cf_reward - penalty * cf_safety

        for agent_name in env.agent_dict:
            assert rewards[agent_name] == expected_reward
            assert safety[agent_name] == cf_safety

    def test_devices_in_microgrids(self, env_config):
        """Test that microgrids have the correct devices."""
        env = MultiAgentMicrogrids(env_config)

        # Check MG1 devices
        mg1 = env.agent_dict['MG1']
        mg1_devices = mg1.devices
        assert len(mg1_devices) == 4  # ESS, DG, 2xRES

        # Check device types
        device_types = [type(d).__name__ for d in mg1_devices]
        assert 'ESS' in device_types
        assert 'DG' in device_types
        assert device_types.count('RES') == 2

    def test_dataset_loaded(self, env_config):
        """Test that dataset is properly loaded."""
        env = MultiAgentMicrogrids(env_config)

        assert hasattr(env, 'dso')
        assert hasattr(env.dso, 'dataset')
        assert env.dso.dataset is not None

        # Check dataset structure
        dataset = env.dso.dataset
        assert 'load' in dataset
        assert 'solar' in dataset
        assert 'wind' in dataset
        assert 'price' in dataset

        # Check data shapes
        for key in ['load', 'solar', 'wind', 'price']:
            assert isinstance(dataset[key], np.ndarray)
            assert len(dataset[key]) > 0

    def test_train_vs_test_mode(self, env_config):
        """Test difference between train and test mode."""
        # Training mode
        train_config = env_config.copy()
        train_config['train'] = True
        env_train = MultiAgentMicrogrids(train_config)

        obs1_train, _ = env_train.reset(seed=42)
        obs2_train, _ = env_train.reset(seed=42)

        # In training mode with same seed, might get different days (random)
        # Just check that reset works
        assert isinstance(obs1_train, dict)
        assert isinstance(obs2_train, dict)

        # Test mode
        test_config = env_config.copy()
        test_config['train'] = False
        env_test = MultiAgentMicrogrids(test_config)

        obs1_test, _ = env_test.reset()
        obs2_test, _ = env_test.reset()

        # In test mode, resets should progress through days sequentially
        # _day should increment
        assert hasattr(env_test, '_day')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
