"""Unit tests for MultiAgentPowerGridEnv (PettingZoo environment)."""

import pytest
import numpy as np
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.devices import ESS, DG
import os
import pickle
from os.path import dirname, abspath

# Load dataset
dir_path = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(dir_path, 'data', 'data2023-2024.pkl')
with open(data_dir, 'rb') as file:
    dataset = pickle.load(file)


def read_data(d, load_area, renew_area):
    """Extract dataset for specific region."""
    return {
        'load': d['load'][load_area],
        'solar': d['solar'][renew_area],
        'solar bus 23': d['solar bus 23'][renew_area],
        'wind': d['wind'][renew_area],
        'price': d['price']['LMP']
    }


@pytest.fixture
def simple_2mg_config():
    """Create a simple 2-microgrid configuration."""
    return {
        'network': None,
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.5,
                       cost_curve_coefs=[100, 70.0, 0.5]),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AVA', 'NP15')
            },
            {
                'name': 'MG2',
                'network': IEEE13Bus('MG2'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.4,
                       cost_curve_coefs=[100, 60.0, 0.45]),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'BANCMID', 'NP15')
            }
        ],
        'horizontal_protocol': 'none',
        'episode_length': 24,
        'base_power': 1.0,
        'train': True,
        'penalty': 10,
        'share_reward': False
    }


@pytest.fixture
def p2p_trading_config():
    """Create a 3-microgrid P2P trading configuration."""
    return {
        'network': None,
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.5,
                       cost_curve_coefs=[100, 70.0, 0.5]),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AVA', 'NP15')
            },
            {
                'name': 'MG2',
                'network': IEEE13Bus('MG2'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.4,
                       cost_curve_coefs=[100, 60.0, 0.45]),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'BANCMID', 'NP15')
            },
            {
                'name': 'MG3',
                'network': IEEE13Bus('MG3'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.35,
                       cost_curve_coefs=[100, 55.0, 0.4]),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AZPS', 'NP15')
            }
        ],
        'horizontal_protocol': 'p2p_trading',
        'episode_length': 24,
        'base_power': 1.0,
        'train': True,
        'penalty': 10,
        'share_reward': False
    }


def test_environment_creation(simple_2mg_config):
    """Test that environment can be created successfully."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    assert env is not None
    assert len(env.possible_agents) == 2
    assert 'MG1' in env.possible_agents
    assert 'MG2' in env.possible_agents
    assert env.episode_length == 24


def test_pettingzoo_api_compliance(simple_2mg_config):
    """Test PettingZoo API compliance."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    # Check required attributes
    assert hasattr(env, 'possible_agents')
    assert hasattr(env, 'action_spaces')
    assert hasattr(env, 'observation_spaces')
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')

    # Check metadata
    assert hasattr(env, 'metadata')
    assert 'name' in env.metadata

    # Check agent lists
    assert isinstance(env.possible_agents, list)
    assert len(env.possible_agents) > 0

    # Check spaces are dicts
    assert isinstance(env.action_spaces, dict)
    assert isinstance(env.observation_spaces, dict)

    # Check spaces match agents
    for agent_id in env.possible_agents:
        assert agent_id in env.action_spaces
        assert agent_id in env.observation_spaces


def test_reset_functionality(simple_2mg_config):
    """Test reset returns correct format."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    obs, info = env.reset(seed=42)

    # Check return types
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    # Check observations for all agents
    for agent_id in env.possible_agents:
        assert agent_id in obs
        assert isinstance(obs[agent_id], np.ndarray)
        assert obs[agent_id].shape[0] > 0

    # Check infos for all agents
    for agent_id in env.possible_agents:
        assert agent_id in info


def test_step_functionality(simple_2mg_config):
    """Test step returns correct format."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    obs, info = env.reset(seed=42)

    # Sample random actions
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.possible_agents
    }

    # Step environment
    obs, rewards, dones, truncated, infos = env.step(actions)

    # Check return types
    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(dones, dict)
    assert isinstance(truncated, dict)
    assert isinstance(infos, dict)

    # Check all agents present
    for agent_id in env.possible_agents:
        assert agent_id in obs
        assert agent_id in rewards
        assert agent_id in dones
        assert agent_id in truncated
        assert agent_id in infos

    # Check __all__ keys
    assert '__all__' in dones
    assert '__all__' in truncated

    # Check types
    for agent_id in env.possible_agents:
        assert isinstance(obs[agent_id], np.ndarray)
        assert isinstance(rewards[agent_id], (int, float))
        assert isinstance(dones[agent_id], bool)
        assert isinstance(truncated[agent_id], bool)
        assert isinstance(infos[agent_id], dict)


def test_multi_microgrid_control(simple_2mg_config):
    """Test controlling multiple microgrids over full episode."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    obs, info = env.reset(seed=42)

    total_rewards = {aid: 0 for aid in env.possible_agents}

    for t in range(env.episode_length):
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }

        obs, rewards, dones, truncated, infos = env.step(actions)

        for aid in env.possible_agents:
            total_rewards[aid] += rewards[aid]

    # Check episode completed
    assert dones['__all__'] == True

    # Check we got rewards for all agents
    for aid in env.possible_agents:
        assert isinstance(total_rewards[aid], (int, float))


def test_reward_computation(simple_2mg_config):
    """Test reward computation includes cost and safety."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    obs, info = env.reset(seed=42)

    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.possible_agents
    }

    obs, rewards, dones, truncated, infos = env.step(actions)

    # Check info contains cost and safety
    for aid in env.possible_agents:
        assert 'cost' in infos[aid]
        assert 'safety' in infos[aid]
        assert 'converged' in infos[aid]

        # Reward should be finite
        assert np.isfinite(rewards[aid])


def test_convergence_penalty(simple_2mg_config):
    """Test that convergence failures result in penalties."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    env.penalty = 100  # Large penalty

    obs, info = env.reset(seed=42)

    # Run a few steps
    rewards_list = []
    converged_list = []

    for _ in range(5):
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }
        obs, rewards, dones, truncated, infos = env.step(actions)

        converged_list.append(infos['MG1']['converged'])
        rewards_list.append(rewards['MG1'])

    # At least some steps should converge
    assert any(converged_list), "Power flow should converge at least once"


def test_shared_reward(simple_2mg_config):
    """Test shared reward mode."""
    simple_2mg_config['share_reward'] = True
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    obs, info = env.reset(seed=42)
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.possible_agents
    }

    obs, rewards, dones, truncated, infos = env.step(actions)

    # All agents should have same reward
    reward_values = list(rewards.values())
    assert np.allclose(reward_values[0], reward_values[1]), \
        "Shared reward mode should give all agents the same reward"


def test_individual_reward(simple_2mg_config):
    """Test individual reward mode."""
    simple_2mg_config['share_reward'] = False
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    obs, info = env.reset(seed=42)

    # Run multiple steps to see if rewards differ
    all_same = True
    for _ in range(10):
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Check if rewards are different (at least sometimes)
        if not np.isclose(rewards['MG1'], rewards['MG2']):
            all_same = False
            break

    # Individual rewards should differ at least once
    assert not all_same, "Individual rewards should differ across agents"


def test_p2p_trading_protocol(p2p_trading_config):
    """Test P2P trading protocol integration."""
    env = MultiAgentPowerGridEnv(p2p_trading_config)

    # Check protocol is set correctly
    assert env.horizontal_protocol.__class__.__name__ == 'PeerToPeerTradingProtocol'

    obs, info = env.reset(seed=42)

    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.possible_agents
    }

    obs, rewards, dones, truncated, infos = env.step(actions)

    # Check environment runs without errors
    assert len(rewards) == 3


def test_price_signal_protocol():
    """Test price signal vertical protocol integration."""
    config = {
        'network': None,
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.5,
                       cost_curve_coefs=[100, 70.0, 0.5]),
                ],
                'vertical_protocol': 'price_signal',  # Use price signal
                'dataset': read_data(dataset, 'AVA', 'NP15')
            }
        ],
        'horizontal_protocol': 'none',
        'episode_length': 24,
        'train': True,
        'penalty': 10,
    }

    env = MultiAgentPowerGridEnv(config)

    # Check protocol is set
    agent = env.agents['MG1']
    assert agent.vertical_protocol.__class__.__name__ == 'PriceSignalProtocol'

    obs, info = env.reset(seed=42)
    actions = {'MG1': env.action_spaces['MG1'].sample()}
    obs, rewards, dones, truncated, infos = env.step(actions)

    # Should run without errors
    assert 'MG1' in rewards


def test_action_space_dimensions(simple_2mg_config):
    """Test action space has correct dimensions."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)

    for agent_id, agent in env.agents.items():
        # Calculate expected action dimension
        expected_dim = sum(
            sub_agent.action_space.shape[0]
            for sub_agent in agent.subordinates.values()
        )

        actual_dim = env.action_spaces[agent_id].shape[0]
        assert actual_dim == expected_dim, \
            f"Agent {agent_id} action space dimension mismatch"


def test_observation_space_dimensions(simple_2mg_config):
    """Test observation space has correct dimensions."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    obs, info = env.reset(seed=42)

    for agent_id in env.possible_agents:
        obs_shape = obs[agent_id].shape
        space_shape = env.observation_spaces[agent_id].shape

        assert obs_shape[0] == space_shape[0], \
            f"Agent {agent_id} observation dimension mismatch"


def test_deterministic_reset(simple_2mg_config):
    """Test that reset with same seed is deterministic."""
    env1 = MultiAgentPowerGridEnv(simple_2mg_config)
    env2 = MultiAgentPowerGridEnv(simple_2mg_config)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    # Observations should be identical
    for agent_id in env1.possible_agents:
        assert np.allclose(obs1[agent_id], obs2[agent_id]), \
            "Reset with same seed should be deterministic"


def test_episode_length(simple_2mg_config):
    """Test episode terminates at correct length."""
    env = MultiAgentPowerGridEnv(simple_2mg_config)
    obs, info = env.reset(seed=42)

    for t in range(env.episode_length - 1):
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Should not terminate before episode_length
        assert dones['__all__'] == False

    # One more step should terminate
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.possible_agents
    }
    obs, rewards, dones, truncated, infos = env.step(actions)
    assert dones['__all__'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
