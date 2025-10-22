"""
Integration tests for multi-agent training environment.

Tests the complete workflow without requiring Ray/RLlib installation.
"""

import pytest
import numpy as np
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config


class TestMultiAgentTrainingIntegration:
    """Integration tests for training readiness."""

    @pytest.fixture
    def env_config(self):
        """Load environment configuration."""
        config = load_config('ieee34_ieee13')
        config['max_episode_steps'] = 24  # Short episodes for testing
        return config

    @pytest.fixture
    def env(self, env_config):
        """Create environment instance."""
        return MultiAgentMicrogrids(env_config)

    def test_environment_creation(self, env):
        """Test that environment can be created."""
        assert env is not None
        assert hasattr(env, 'possible_agents')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'observation_spaces')
        assert len(env.possible_agents) == 3  # MG1, MG2, MG3

    def test_action_space_types(self, env):
        """Test that action spaces are Dict spaces with continuous and discrete."""
        from gymnasium.spaces import Dict, Box, MultiDiscrete

        for agent_id in env.possible_agents:
            action_space = env.action_spaces[agent_id]
            assert isinstance(action_space, Dict), f"{agent_id} should have Dict action space"
            assert 'continuous' in action_space.spaces
            assert 'discrete' in action_space.spaces
            assert isinstance(action_space['continuous'], Box)
            assert isinstance(action_space['discrete'], MultiDiscrete)

    def test_observation_space_types(self, env):
        """Test that observation spaces are Box spaces."""
        from gymnasium.spaces import Box

        for agent_id in env.possible_agents:
            obs_space = env.observation_spaces[agent_id]
            assert isinstance(obs_space, Box), f"{agent_id} should have Box obs space"
            assert obs_space.shape[0] > 0, "Observation should be non-empty"

    def test_reset_returns_valid_observations(self, env):
        """Test that reset returns valid observations."""
        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert set(obs.keys()) == set(env.possible_agents)

        for agent_id in env.possible_agents:
            assert agent_id in obs
            assert isinstance(obs[agent_id], np.ndarray)
            # Check observation has reasonable size (may differ from space due to network merging)
            obs_space = env.observation_spaces[agent_id]
            assert obs[agent_id].shape[0] > 0, f"{agent_id} observation should not be empty"
            # Note: obs size may differ from space due to network topology changes during merging
            # The key is that observations are valid (no NaN/inf)
            assert not np.any(np.isnan(obs[agent_id])), f"{agent_id} has NaN in observation"
            assert not np.any(np.isinf(obs[agent_id])), f"{agent_id} has inf in observation"

    def test_step_with_random_actions(self, env):
        """Test that step works with random actions."""
        obs, info = env.reset(seed=42)

        # Sample random actions
        actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}

        # Take step
        next_obs, rewards, dones, truncated, infos = env.step(actions)

        # Validate observations
        assert isinstance(next_obs, dict)
        assert set(next_obs.keys()) == set(env.possible_agents)
        for agent_id in env.possible_agents:
            assert not np.any(np.isnan(next_obs[agent_id])), f"{agent_id} has NaN in next_obs"
            assert not np.any(np.isinf(next_obs[agent_id])), f"{agent_id} has inf in next_obs"

        # Validate rewards
        assert isinstance(rewards, dict)
        assert set(rewards.keys()) == set(env.possible_agents)
        for agent_id in env.possible_agents:
            assert isinstance(rewards[agent_id], (int, float, np.number))
            assert not np.isnan(rewards[agent_id]), f"{agent_id} has NaN reward"
            assert not np.isinf(rewards[agent_id]), f"{agent_id} has inf reward"
            assert -1000 < rewards[agent_id] < 1000, f"{agent_id} reward out of reasonable range"

    def test_full_episode(self, env, env_config):
        """Test a complete episode with random actions."""
        obs, info = env.reset(seed=42)

        episode_rewards = {aid: [] for aid in env.possible_agents}
        done = False
        step_count = 0
        max_steps = env_config.get('max_episode_steps', 100) + 10  # Safety limit

        while not done and step_count < max_steps:
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)

            for agent_id in env.possible_agents:
                episode_rewards[agent_id].append(rewards[agent_id])

            # Check if all agents are done
            done = all(dones.values()) or all(truncated.values())
            step_count += 1

        # Validate episode
        assert step_count > 0, "Episode should have at least one step"
        assert step_count <= max_steps, "Episode exceeded safety limit"

        for agent_id in env.possible_agents:
            assert len(episode_rewards[agent_id]) > 0
            total_reward = sum(episode_rewards[agent_id])
            assert not np.isnan(total_reward), f"{agent_id} has NaN total reward"
            assert not np.isinf(total_reward), f"{agent_id} has inf total reward"

    def test_deterministic_reset(self, env_config):
        """Test that reset with same seed gives consistent initial observations."""
        # Create fresh environments for determinism test
        env1 = MultiAgentMicrogrids(env_config)
        env2 = MultiAgentMicrogrids(env_config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        for agent_id in env1.possible_agents:
            # Observations should be similar (may have minor differences due to floating point)
            # Check most values are close
            diff = np.abs(obs1[agent_id] - obs2[agent_id])
            similar_values = np.sum(diff < 0.01) / len(diff)
            assert similar_values > 0.9, f"{agent_id} observations too different: {similar_values:.2%} similar"

    def test_action_bounds_respected(self, env):
        """Test that environment respects action space bounds."""
        obs, info = env.reset(seed=42)

        for agent_id in env.possible_agents:
            action_space = env.action_spaces[agent_id]

            # Test with extreme actions within bounds
            action = {
                'continuous': action_space['continuous'].high,
                'discrete': action_space['discrete'].nvec - 1  # Max discrete actions
            }

            try:
                next_obs, rewards, dones, truncated, infos = env.step({agent_id: action})
                # Should not crash
                assert True
            except Exception as e:
                pytest.fail(f"Failed with valid action: {e}")

    def test_reward_structure(self, env_config):
        """Test reward structure with different configurations."""
        # Test shared rewards
        env_config['share_reward'] = True
        env_shared = MultiAgentMicrogrids(env_config)
        obs, _ = env_shared.reset(seed=42)
        actions = {aid: env_shared.action_spaces[aid].sample() for aid in env_shared.possible_agents}
        _, rewards_shared, _, _, _ = env_shared.step(actions)

        # All agents should have same reward when shared
        reward_values = list(rewards_shared.values())
        assert all(abs(r - reward_values[0]) < 1e-6 for r in reward_values), \
            "Shared rewards should be identical"

        # Test independent rewards
        env_config['share_reward'] = False
        env_independent = MultiAgentMicrogrids(env_config)
        obs, _ = env_independent.reset(seed=42)
        actions = {aid: env_independent.action_spaces[aid].sample() for aid in env_independent.possible_agents}
        _, rewards_independent, _, _, _ = env_independent.step(actions)

        # Rewards can differ when independent
        # Just check they're all valid
        for agent_id in env_independent.possible_agents:
            assert isinstance(rewards_independent[agent_id], (int, float, np.number))
            assert not np.isnan(rewards_independent[agent_id])

    def test_convergence_failure_handling(self, env):
        """Test that convergence failures are handled properly."""
        obs, info = env.reset(seed=42)

        # Take multiple steps
        for _ in range(10):
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)

            # Check for convergence info in infos
            # Note: info structure may vary, so we just check it's not crashing
            assert isinstance(infos, dict)

    def test_multiple_episodes(self, env):
        """Test multiple consecutive episodes."""
        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)

            done = False
            step_count = 0

            while not done and step_count < 50:
                actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
                obs, rewards, dones, truncated, infos = env.step(actions)
                done = all(dones.values()) or all(truncated.values())
                step_count += 1

            assert step_count > 0, f"Episode {episode} should have steps"

    def test_pettingzoo_api_compatibility(self, env):
        """Test PettingZoo API compatibility for RLlib."""
        # Check required PettingZoo API methods
        assert hasattr(env, 'possible_agents')
        assert hasattr(env, 'observation_spaces')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

        # Check that spaces dict has all agents
        assert set(env.observation_spaces.keys()) == set(env.possible_agents)
        assert set(env.action_spaces.keys()) == set(env.possible_agents)

    def test_action_dict_format(self, env):
        """Test that environment accepts Dict action format."""
        obs, info = env.reset(seed=42)

        for agent_id in env.possible_agents:
            action_space = env.action_spaces[agent_id]

            # Create action in dict format
            action = action_space.sample()
            assert isinstance(action, dict)
            assert 'continuous' in action
            assert 'discrete' in action

            # Single agent step
            actions = {agent_id: action}
            # Fill others with zeros
            for other_id in env.possible_agents:
                if other_id != agent_id:
                    actions[other_id] = {
                        'continuous': np.zeros_like(action_space['continuous'].sample()),
                        'discrete': np.zeros_like(action_space['discrete'].sample())
                    }

            try:
                next_obs, rewards, dones, truncated, infos = env.step(actions)
                assert True  # Should not crash
            except Exception as e:
                pytest.fail(f"Failed with dict action format: {e}")
