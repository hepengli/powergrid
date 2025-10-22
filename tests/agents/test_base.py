"""Tests for agents.base module."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box

from powergrid.agents.base import Agent, Observation, Message


class TestObservation:
    """Test Observation dataclass and methods."""

    def test_observation_initialization(self):
        """Test observation initialization with default values."""
        obs = Observation()
        assert isinstance(obs.local, dict)
        assert isinstance(obs.global_info, dict)
        assert isinstance(obs.messages, list)
        assert obs.timestamp == 0.0

    def test_observation_custom_values(self):
        """Test observation with custom values."""
        local = {"P": 1.0, "Q": 0.5}
        global_info = {"voltage": 1.05}
        messages = [Message("agent1", {"price": 50.0})]
        timestamp = 10.0

        obs = Observation(
            local=local,
            global_info=global_info,
            messages=messages,
            timestamp=timestamp
        )

        assert obs.local == local
        assert obs.global_info == global_info
        assert len(obs.messages) == 1
        assert obs.timestamp == 10.0

    def test_as_vector_empty(self):
        """Test as_vector with empty observation."""
        obs = Observation()
        vec = obs.as_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) == 0

    def test_as_vector_local_scalars(self):
        """Test as_vector with scalar values."""
        obs = Observation(local={"P": 1.0, "Q": 0.5, "soc": 0.8})
        vec = obs.as_vector()

        assert len(vec) == 3
        assert vec.dtype == np.float32
        # Sorted by key: P, Q, soc
        np.testing.assert_array_almost_equal(vec, [1.0, 0.5, 0.8])

    def test_as_vector_with_arrays(self):
        """Test as_vector with numpy arrays."""
        obs = Observation(local={
            "P": 1.0,
            "voltages": np.array([1.0, 1.05, 0.95])
        })
        vec = obs.as_vector()

        assert len(vec) == 4  # 1 scalar + 3 array elements
        assert vec.dtype == np.float32

    def test_as_vector_nested_dict(self):
        """Test as_vector with nested dictionaries."""
        obs = Observation(local={
            "device1": {"P": 1.0, "Q": 0.5},
            "device2": {"P": 2.0, "Q": 1.0}
        })
        vec = obs.as_vector()

        assert len(vec) == 4
        assert vec.dtype == np.float32

    def test_as_vector_with_global_info(self):
        """Test as_vector includes global_info."""
        obs = Observation(
            local={"P": 1.0},
            global_info={"voltage": 1.05}
        )
        vec = obs.as_vector()

        assert len(vec) == 2
        assert vec.dtype == np.float32


class TestMessage:
    """Test Message dataclass."""

    def test_message_initialization(self):
        """Test message initialization."""
        msg = Message(
            sender="agent1",
            content={"price": 50.0}
        )

        assert msg.sender == "agent1"
        assert msg.content == {"price": 50.0}
        assert msg.recipient is None
        assert msg.timestamp == 0.0

    def test_message_with_recipient(self):
        """Test message with specific recipient."""
        msg = Message(
            sender="agent1",
            content={"setpoint": 100.0},
            recipient="agent2",
            timestamp=5.0
        )

        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.timestamp == 5.0

    def test_message_with_multiple_recipients(self):
        """Test message with multiple recipients."""
        msg = Message(
            sender="grid",
            content={"price": 50.0},
            recipient=["agent1", "agent2", "agent3"]
        )

        assert msg.recipient == ["agent1", "agent2", "agent3"]


class ConcreteAgent(Agent):
    """Concrete implementation of Agent for testing."""

    def observe(self, global_state=None, *args, **kwargs):
        """Return simple observation."""
        return Observation(
            local={"value": 1.0},
            timestamp=self._timestep
        )

    def act(self, observation, *args, **kwargs):
        """Return simple action."""
        return np.array([1.0])


class TestAgent:
    """Test Agent abstract base class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent(
            agent_id="test_agent",
            level=1
        )

        assert agent.agent_id == "test_agent"
        assert agent.level == 1
        assert len(agent.mailbox) == 0
        assert agent._timestep == 0.0

    def test_agent_with_spaces(self):
        """Test agent initialization with gym spaces."""
        obs_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)

        agent = ConcreteAgent(
            agent_id="test_agent",
            level=2,
            observation_space=obs_space,
            action_space=action_space
        )

        assert agent.observation_space == obs_space
        assert agent.action_space == action_space

    def test_agent_reset(self):
        """Test agent reset clears state."""
        agent = ConcreteAgent(agent_id="test")

        # Add some messages
        agent.mailbox.append(Message("other", {"data": 1}))
        agent._timestep = 10.0

        # Reset
        agent.reset()

        assert len(agent.mailbox) == 0
        assert agent._timestep == 0.0

    def test_agent_observe(self):
        """Test agent observe method."""
        agent = ConcreteAgent(agent_id="test")
        agent._timestep = 5.0

        obs = agent.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0
        assert "value" in obs.local

    def test_agent_act(self):
        """Test agent act method."""
        agent = ConcreteAgent(agent_id="test")
        obs = Observation(local={"value": 1.0})

        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        np.testing.assert_array_equal(action, [1.0])

    def test_send_message_broadcast(self):
        """Test sending broadcast message."""
        agent = ConcreteAgent(agent_id="sender")
        agent._timestep = 5.0

        msg = agent.send_message({"price": 50.0})

        assert msg.sender == "sender"
        assert msg.content == {"price": 50.0}
        assert msg.recipient is None
        assert msg.timestamp == 5.0

    def test_send_message_to_recipient(self):
        """Test sending message to specific recipient."""
        agent = ConcreteAgent(agent_id="sender")

        msg = agent.send_message(
            {"setpoint": 100.0},
            recipients="receiver"
        )

        assert msg.recipient == "receiver"

    def test_send_message_to_multiple_recipients(self):
        """Test sending message to multiple recipients."""
        agent = ConcreteAgent(agent_id="sender")

        msg = agent.send_message(
            {"price": 50.0},
            recipients=["agent1", "agent2"]
        )

        assert msg.recipient == ["agent1", "agent2"]

    def test_receive_message(self):
        """Test receiving message."""
        agent = ConcreteAgent(agent_id="receiver")
        msg = Message("sender", {"data": 1})

        agent.receive_message(msg)

        assert len(agent.mailbox) == 1
        assert agent.mailbox[0] == msg

    def test_clear_mailbox(self):
        """Test clearing mailbox."""
        agent = ConcreteAgent(agent_id="test")

        # Add messages
        msg1 = Message("sender1", {"data": 1})
        msg2 = Message("sender2", {"data": 2})
        agent.mailbox.extend([msg1, msg2])

        # Clear mailbox
        messages = agent.clear_mailbox()

        assert len(messages) == 2
        assert len(agent.mailbox) == 0
        assert messages[0] == msg1
        assert messages[1] == msg2

    def test_update_timestep(self):
        """Test updating timestep."""
        agent = ConcreteAgent(agent_id="test")

        agent.update_timestep(10.5)

        assert agent._timestep == 10.5

    def test_agent_repr(self):
        """Test agent string representation."""
        agent = ConcreteAgent(agent_id="test_agent", level=2)

        repr_str = repr(agent)

        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str
        assert "level=2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
