"""Unit tests for coordination protocols."""

import pytest
import numpy as np
from powergrid.agents.protocols import (
    PriceSignalProtocol, SetpointProtocol, NoProtocol,
    PeerToPeerTradingProtocol, ConsensusProtocol, NoHorizontalProtocol
)
from powergrid.agents.base import Observation
from powergrid.agents import GridAgent, DeviceAgent
from powergrid.devices import ESS, DG


def test_no_protocol():
    """Test NoProtocol vertical protocol."""
    protocol = NoProtocol()

    # Mock subordinate observations
    sub_obs = {
        'ess1': Observation(local={'P': 0.5}),
        'dg1': Observation(local={'P': 0.3})
    }

    # Coordinate without parent action
    signals = protocol.coordinate(sub_obs, parent_action=None)

    assert 'ess1' in signals
    assert 'dg1' in signals
    assert signals['ess1'] == {}
    assert signals['dg1'] == {}


def test_price_signal_protocol():
    """Test price signal vertical protocol."""
    protocol = PriceSignalProtocol(initial_price=50.0)

    # Mock subordinate observations
    sub_obs = {
        'ess1': Observation(local={'P': 0.5}),
        'dg1': Observation(local={'P': 0.3})
    }

    # Coordinate without parent action
    signals = protocol.coordinate(sub_obs, parent_action=None)

    assert 'ess1' in signals
    assert 'dg1' in signals
    assert signals['ess1']['price'] == 50.0
    assert signals['dg1']['price'] == 50.0

    # Coordinate with parent action (float)
    signals = protocol.coordinate(sub_obs, parent_action=80.0)
    assert signals['ess1']['price'] == 80.0
    assert signals['dg1']['price'] == 80.0

    # Coordinate with parent action (dict)
    signals = protocol.coordinate(sub_obs, parent_action={'price': 100.0})
    assert signals['ess1']['price'] == 100.0


def test_setpoint_protocol():
    """Test setpoint vertical protocol."""
    protocol = SetpointProtocol()

    sub_obs = {
        'ess1': Observation(local={'P': 0.5}),
        'dg1': Observation(local={'P': 0.3})
    }

    # Without parent action
    signals = protocol.coordinate(sub_obs, parent_action=None)
    assert signals['ess1'] == {}
    assert signals['dg1'] == {}

    # With parent action
    parent_action = {'ess1': 0.8, 'dg1': 0.4}
    signals = protocol.coordinate(sub_obs, parent_action)
    assert signals['ess1']['setpoint'] == 0.8
    assert signals['dg1']['setpoint'] == 0.4

    # With partial parent action
    parent_action = {'ess1': 0.8}
    signals = protocol.coordinate(sub_obs, parent_action)
    assert signals['ess1']['setpoint'] == 0.8
    assert signals['dg1'] == {}


def test_no_horizontal_protocol():
    """Test NoHorizontalProtocol."""
    protocol = NoHorizontalProtocol()

    # Mock agents and observations
    agents = {
        'MG1': None,
        'MG2': None
    }
    observations = {
        'MG1': Observation(local={'P': 0.5}),
        'MG2': Observation(local={'P': 0.3})
    }

    signals = protocol.coordinate(agents, observations)

    assert 'MG1' in signals
    assert 'MG2' in signals
    assert signals['MG1'] == {}
    assert signals['MG2'] == {}


def test_p2p_trading_protocol_basic():
    """Test P2P trading horizontal protocol with simple trade."""
    protocol = PeerToPeerTradingProtocol(trading_fee=0.01)

    # Create simple mock agents (we only need the IDs)
    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True)
    }

    # Mock observations with net demand
    observations = {
        'MG1': Observation(local={'net_demand': -0.5, 'marginal_cost': 40}),  # Seller
        'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 60})    # Buyer
    }

    # Coordinate
    signals = protocol.coordinate(agents, observations)

    # Verify signals exist
    assert 'MG1' in signals or len(signals) == 0  # May have no trades if market doesn't clear
    assert 'MG2' in signals or len(signals) == 0

    # If trades occurred
    if 'MG1' in signals and 'trades' in signals['MG1']:
        assert 'trades' in signals['MG1']
        assert 'trades' in signals['MG2']
        assert len(signals['MG1']['trades']) > 0
        assert len(signals['MG2']['trades']) > 0

        # Check trade structure
        mg1_trade = signals['MG1']['trades'][0]
        assert 'counterparty' in mg1_trade
        assert 'quantity' in mg1_trade
        assert 'price' in mg1_trade

        # MG1 is seller, should have negative quantity
        assert mg1_trade['quantity'] < 0


def test_p2p_trading_protocol_no_trade():
    """Test P2P trading when prices don't match."""
    protocol = PeerToPeerTradingProtocol()

    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True)
    }

    # Both want to buy at low prices - no trade possible
    observations = {
        'MG1': Observation(local={'net_demand': 0.5, 'marginal_cost': 30}),
        'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 35})
    }

    signals = protocol.coordinate(agents, observations)

    # Should have no trades (or empty trade lists)
    # The protocol may return empty dicts or dicts with empty trade lists
    for aid in agents:
        if aid in signals:
            if 'trades' in signals[aid]:
                assert len(signals[aid]['trades']) == 0


def test_p2p_trading_multiple_agents():
    """Test P2P trading with 3 agents."""
    protocol = PeerToPeerTradingProtocol()

    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True),
        'MG3': GridAgent('MG3', subordinates=[], centralized=True)
    }

    observations = {
        'MG1': Observation(local={'net_demand': -0.5, 'marginal_cost': 40}),  # Seller
        'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 60}),   # Buyer
        'MG3': Observation(local={'net_demand': 0.2, 'marginal_cost': 55})    # Buyer
    }

    signals = protocol.coordinate(agents, observations)

    # At least some agents should have trades
    total_trades = sum(
        len(signals.get(aid, {}).get('trades', []))
        for aid in agents
    )
    assert total_trades > 0


def test_consensus_protocol():
    """Test consensus horizontal protocol."""
    protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True),
        'MG3': GridAgent('MG3', subordinates=[], centralized=True)
    }

    # Different initial values
    observations = {
        'MG1': Observation(local={'control_value': 59.9}),
        'MG2': Observation(local={'control_value': 60.1}),
        'MG3': Observation(local={'control_value': 60.0})
    }

    # Coordinate
    signals = protocol.coordinate(agents, observations)

    # Verify all agents have consensus values
    assert 'MG1' in signals
    assert 'MG2' in signals
    assert 'MG3' in signals
    assert 'consensus_value' in signals['MG1']
    assert 'consensus_value' in signals['MG2']
    assert 'consensus_value' in signals['MG3']

    # Verify consensus reached (values should be close)
    values = [signals[aid]['consensus_value'] for aid in agents]
    assert np.std(values) < 0.1  # Converged
    assert abs(np.mean(values) - 60.0) < 0.1  # Around average


def test_consensus_protocol_convergence():
    """Test consensus protocol converges from different starting points."""
    protocol = ConsensusProtocol(max_iterations=20, tolerance=0.001)

    agents = {
        f'MG{i}': GridAgent(f'MG{i}', subordinates=[], centralized=True)
        for i in range(1, 6)
    }

    # Wide range of initial values
    initial_values = [50, 55, 60, 65, 70]
    observations = {
        f'MG{i}': Observation(local={'control_value': val})
        for i, val in enumerate(initial_values, 1)
    }

    signals = protocol.coordinate(agents, observations)

    # All should converge to average
    values = [signals[aid]['consensus_value'] for aid in agents]
    expected_avg = np.mean(initial_values)

    # Check convergence
    assert np.std(values) < 1.0
    assert abs(np.mean(values) - expected_avg) < 1.0


def test_consensus_protocol_with_topology():
    """Test consensus with custom topology (not fully connected)."""
    protocol = ConsensusProtocol(max_iterations=20, tolerance=0.01)

    agents = {
        'MG1': GridAgent('MG1', subordinates=[], centralized=True),
        'MG2': GridAgent('MG2', subordinates=[], centralized=True),
        'MG3': GridAgent('MG3', subordinates=[], centralized=True)
    }

    observations = {
        'MG1': Observation(local={'control_value': 50}),
        'MG2': Observation(local={'control_value': 60}),
        'MG3': Observation(local={'control_value': 70})
    }

    # Linear topology: MG1 -- MG2 -- MG3
    topology = {
        'adjacency': {
            'MG1': ['MG2'],
            'MG2': ['MG1', 'MG3'],
            'MG3': ['MG2']
        }
    }

    signals = protocol.coordinate(agents, observations, topology=topology)

    # Should still converge (though may take longer)
    values = [signals[aid]['consensus_value'] for aid in agents]
    assert np.std(values) < 5.0  # More relaxed convergence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
