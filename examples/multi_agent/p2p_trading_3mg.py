"""
P2P Trading Example with 3 Microgrids

This example demonstrates:
- Three microgrids with peer-to-peer energy trading
- Horizontal protocol: PeerToPeerTradingProtocol
- Market-based coordination between microgrids
- Trade logging and visualization

Usage:
    python examples/multi_agent/p2p_trading_3mg.py
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
data_dir = os.path.join(dir_path, 'data', 'data2023-2024.pkl')
with open(data_dir, 'rb') as file:
    dataset = pickle.load(file)


def read_data(d, load_area, renew_area, train=True):
    """Extract dataset for specific region with train/test split."""
    split = 'train' if train else 'test'
    data = d[split]
    return {
        'load': data['load'][load_area],
        'solar': data['solar'][renew_area],
        'wind': data['wind'][renew_area],
        'price': data['price']['0096WD_7_N001']
    }


def create_p2p_trading_env():
    """Create a 3-microgrid environment with P2P trading."""

    config = {
        'network': None,  # No base network
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.66,
                       cost_curve_coefs=[100, 72.4, 0.5011]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AVA', 'NP15')
            },
            {
                'name': 'MG2',
                'network': IEEE13Bus('MG2'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.60,
                       cost_curve_coefs=[100, 51.6, 0.4615]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'BANCMID', 'NP15')
            },
            {
                'name': 'MG3',
                'network': IEEE13Bus('MG3'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5,
                        capacity=2.0, max_e_mwh=2.0, min_e_mwh=0.2, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.50,
                       cost_curve_coefs=[100, 51.6, 0.4615]),
                    DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar'),
                    DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind'),
                ],
                'vertical_protocol': 'none',
                'dataset': read_data(dataset, 'AZPS', 'NP15')
            }
        ],
        'horizontal_protocol': 'p2p_trading',  # Enable P2P trading
        'topology': None,  # Fully connected (all can trade with all)
        'episode_length': 24,
        'base_power': 3.0,
        'train': True,
        'penalty': 10,
        'share_reward': False
    }

    return MultiAgentPowerGridEnv(config)


def main():
    """Run P2P trading example."""
    print("=" * 70)
    print("P2P Energy Trading Example - 3 Microgrids")
    print("=" * 70)

    # Create environment
    env = create_p2p_trading_env()
    print(f"\nEnvironment created with agents: {env.possible_agents}")
    print(f"Horizontal protocol: {env.horizontal_protocol.__class__.__name__}")
    print(f"Trading fee: {env.horizontal_protocol.trading_fee * 100:.1f}%")

    # Reset environment
    obs, info = env.reset(seed=42)

    # Run episode
    print(f"\nRunning episode for {env.episode_length} timesteps...")
    print("-" * 70)
    print(f"{'Time':>4} | {'Agent':>4} | {'Reward':>8} | {'Cost':>6} | "
          f"{'Trades':>6} | {'Conv':>5}")
    print("-" * 70)

    total_rewards = {aid: 0 for aid in env.possible_agents}
    total_trades = {aid: 0 for aid in env.possible_agents}

    for t in range(env.episode_length):
        # Sample random actions
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }

        # Step environment
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Count trades from agent mailboxes
        trade_counts = {}
        for aid, agent in env.agents.items():
            trades = [msg for msg in agent.mailbox if 'trades' in msg.content]
            trade_counts[aid] = len(trades[0].content.get('trades', [])) if trades else 0
            total_trades[aid] += trade_counts[aid]

        # Accumulate rewards
        for aid in env.possible_agents:
            total_rewards[aid] += rewards[aid]

        # Print every 6 hours
        if t % 6 == 0:
            for aid in env.possible_agents:
                print(f"{t:4d} | {aid:>4} | {rewards[aid]:8.2f} | "
                      f"{infos[aid]['cost']:6.2f} | {trade_counts[aid]:6d} | "
                      f"{'✓' if infos[aid]['converged'] else '✗':>5}")

    print("-" * 70)
    print(f"\nEpisode Summary:")
    print(f"  Episode completed: {dones['__all__']}")
    print(f"\n  Agent Performance:")
    for aid in env.possible_agents:
        print(f"    {aid}:")
        print(f"      Total reward: {total_rewards[aid]:8.2f}")
        print(f"      Total trades: {total_trades[aid]:8d}")
        print(f"      Avg trades/step: {total_trades[aid]/env.episode_length:6.2f}")

    print("\n  Trading Statistics:")
    total_all_trades = sum(total_trades.values()) // 2  # Each trade counted twice
    print(f"    Total market trades: {total_all_trades}")
    print(f"    Avg trades per timestep: {total_all_trades / env.episode_length:.2f}")

    print("\n" + "=" * 70)
    print("P2P Trading Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
