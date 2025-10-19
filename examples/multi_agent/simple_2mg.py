"""
Simple 2-Microgrid Example

This example demonstrates:
- Two independent microgrids (MG1, MG2)
- Each with basic devices (ESS, DG, PV)
- No coordination protocols (independent operation)
- Random policy for demonstration

Usage:
    python examples/multi_agent/simple_2mg.py
"""

import numpy as np
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.devices import ESS, DG
import os
import pickle
from os.path import dirname, abspath

# Load dataset
dir_path = dirname(dirname(dirname(abspath(__file__))))
data_dir = os.path.join(dir_path, 'data', 'data2024.pkl')
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


def create_simple_2mg_env():
    """Create a simple 2-microgrid environment."""

    config = {
        'network': None,  # No base network, standalone microgrids
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.5, max_e_mwh=1.5, min_e_mwh=0.15, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.5,
                       cost_curve_coefs=[100, 70.0, 0.5]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                ],
                'vertical_protocol': 'none',  # No coordination
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
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                ],
                'vertical_protocol': 'none',  # No coordination
                'dataset': read_data(dataset, 'BANCMID', 'NP15')
            }
        ],
        'horizontal_protocol': 'none',  # No peer trading
        'episode_length': 24,
        'base_power': 1.0,
        'train': True,
        'penalty': 10,
        'share_reward': False
    }

    return MultiAgentPowerGridEnv(config)


def main():
    """Run simple 2-microgrid example."""
    print("=" * 60)
    print("Simple 2-Microgrid Example")
    print("=" * 60)

    # Create environment
    env = create_simple_2mg_env()
    print(f"\nEnvironment created with agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shapes:")
    for agent_id, ob in obs.items():
        print(f"  {agent_id}: {ob.shape}")

    # Run one episode with random actions
    print(f"\nRunning episode for {env.episode_length} timesteps...")
    print("-" * 60)

    total_rewards = {aid: 0 for aid in env.possible_agents}

    for t in range(env.episode_length):
        # Sample random actions
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }

        # Step environment
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Accumulate rewards
        for aid in env.possible_agents:
            total_rewards[aid] += rewards[aid]

        # Print progress
        if t % 6 == 0:  # Print every 6 hours
            print(f"Timestep {t:2d}:")
            for aid in env.possible_agents:
                print(f"  {aid}: reward={rewards[aid]:8.2f}, "
                      f"cost={infos[aid]['cost']:6.2f}, "
                      f"safety={infos[aid]['safety']:6.2f}, "
                      f"converged={infos[aid]['converged']}")

    print("-" * 60)
    print(f"\nEpisode Summary:")
    print(f"  Episode completed: {dones['__all__']}")
    for aid in env.possible_agents:
        print(f"  {aid} total reward: {total_rewards[aid]:.2f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
