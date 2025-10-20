"""Integration tests for RLlib compatibility."""

import pytest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray once for all tests."""
    ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
    ray.shutdown()


def env_creator(env_config):
    """Create environment for RLlib."""
    env = MultiAgentMicrogridsV2(env_config)
    return ParallelPettingZooEnv(env)


def test_mappo_training_basic(ray_init):
    """Test MAPPO training for 5 iterations with shared policy."""
    # Register environment
    register_env("test_mg_mappo", env_creator)

    env_config = {
        'train': True,
        'penalty': 10,
        'share_reward': True
    }

    # Create temporary environment to get spaces
    temp_env = env_creator(env_config)

    # Configure PPO with shared policy
    config = (
        PPOConfig()
        .environment(
            env="test_mg_mappo",
            env_config=env_config,
        )
        .framework("torch")
        .training(
            train_batch_size=512,
            sgd_minibatch_size=64,
            num_sgd_iter=5,
            lr=5e-4,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies={'shared_policy': (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy',
        )
        .rollouts(
            num_rollout_workers=0,  # No parallelization for test
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=0,
        )
        .debugging(
            seed=42,
        )
    )

    # Build and train
    algo = config.build()

    # Train for 5 iterations
    for i in range(5):
        result = algo.train()

        # Check that training returns expected metrics
        assert 'episode_reward_mean' in result
        assert 'episodes_this_iter' in result
        assert 'timesteps_total' in result

        # Check rewards are finite
        assert result['episode_reward_mean'] is not None
        import math
        assert math.isfinite(result['episode_reward_mean'])

    # Cleanup
    algo.stop()


def test_ippo_training_basic(ray_init):
    """Test IPPO training with independent policies."""
    # Register environment
    register_env("test_mg_ippo", env_creator)

    env_config = {
        'train': True,
        'penalty': 10,
        'share_reward': False
    }

    # Create temporary environment to get agent list
    temp_env = env_creator(env_config)
    agent_ids = temp_env.possible_agents

    # Configure PPO with independent policies
    policies = {
        agent_id: (None, temp_env.observation_space(agent_id),
                   temp_env.action_space(agent_id), {})
        for agent_id in agent_ids
    }

    config = (
        PPOConfig()
        .environment(
            env="test_mg_ippo",
            env_config=env_config,
        )
        .framework("torch")
        .training(
            train_batch_size=512,
            sgd_minibatch_size=64,
            num_sgd_iter=5,
            lr=5e-4,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args: agent_id,
        )
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=0,
        )
        .debugging(
            seed=42,
        )
    )

    # Build and train
    algo = config.build()

    # Train for 5 iterations
    for i in range(5):
        result = algo.train()

        # Verify training progresses
        assert 'episode_reward_mean' in result
        assert result['timesteps_total'] > 0

    # Cleanup
    algo.stop()


def test_rllib_checkpoint_restore(ray_init, tmp_path):
    """Test checkpoint saving and restoration."""
    register_env("test_mg_checkpoint", env_creator)

    env_config = {'train': True, 'penalty': 10}

    config = (
        PPOConfig()
        .environment(env="test_mg_checkpoint", env_config=env_config)
        .framework("torch")
        .training(
            train_batch_size=256,
            model={"fcnet_hiddens": [32, 32]},
        )
        .multi_agent(
            policies={'shared_policy': (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy',
        )
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Train for 2 iterations
    result1 = algo.train()
    result2 = algo.train()

    # Save checkpoint
    checkpoint_path = algo.save(str(tmp_path))
    assert checkpoint_path is not None

    # Get current timesteps
    timesteps_before = result2['timesteps_total']

    # Stop algo
    algo.stop()

    # Restore from checkpoint
    algo_restored = config.build()
    algo_restored.restore(checkpoint_path)

    # Train one more iteration
    result3 = algo_restored.train()

    # Timesteps should continue from checkpoint
    assert result3['timesteps_total'] > timesteps_before

    # Cleanup
    algo_restored.stop()


def test_rllib_policy_inference(ray_init):
    """Test that trained policy can compute actions."""
    register_env("test_mg_inference", env_creator)

    env_config = {'train': False, 'penalty': 10}

    config = (
        PPOConfig()
        .environment(env="test_mg_inference", env_config=env_config)
        .framework("torch")
        .training(
            train_batch_size=256,
            model={"fcnet_hiddens": [32, 32]},
        )
        .multi_agent(
            policies={'shared_policy': (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy',
        )
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Train briefly
    algo.train()

    # Create environment and get observation
    env = env_creator(env_config)
    obs, info = env.reset()

    # Compute actions for all agents
    for agent_id in env.possible_agents:
        action = algo.compute_single_action(
            obs[agent_id],
            policy_id='shared_policy'
        )

        # Check action is valid
        assert action is not None
        assert env.action_space(agent_id).contains(action)

    # Cleanup
    algo.stop()


def test_multiagent_microgrids_v2_compatibility(ray_init):
    """Test that MultiAgentMicrogridsV2 works with RLlib."""
    # This test verifies backward compatibility with the new V2 implementation
    register_env("test_mg_v2", env_creator)

    env_config = {'train': True, 'penalty': 10, 'share_reward': True}

    config = (
        PPOConfig()
        .environment(env="test_mg_v2", env_config=env_config)
        .framework("torch")
        .training(
            train_batch_size=256,
            model={"fcnet_hiddens": [32, 32]},
        )
        .multi_agent(
            policies={'shared_policy': (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args: 'shared_policy',
        )
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Should train without errors
    result = algo.train()
    assert 'episode_reward_mean' in result

    # Verify 3 agents (MG1, MG2, MG3)
    env = env_creator(env_config)
    assert len(env.possible_agents) == 3
    assert 'MG1' in env.possible_agents
    assert 'MG2' in env.possible_agents
    assert 'MG3' in env.possible_agents

    algo.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
