"""Unit tests for base Agent class and related abstractions."""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from powergrid.agents.base import (
    Agent,
    Observation,
    Message,
)
from powergrid.core.policies import Policy, RandomPolicy


class DummyAgent(Agent):
    """Minimal concrete Agent implementation for testing."""

    def observe(self, global_state):
        obs = Observation(timestamp=self._timestep)
        obs.local["value"] = global_state.get("test_value", 0.0)
        return obs

    def act(self, observation):
        return self.action_space.sample()

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)


class TestObservation:
    """Tests for Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an observation with local and global info."""
        obs = Observation(
            local={"P": 1.0, "Q": 0.5},
            global_info={"bus_voltage": 1.05},
            timestamp=10.0,
        )

        assert obs.local["P"] == 1.0
        assert obs.global_info["bus_voltage"] == 1.05
        assert obs.timestamp == 10.0
        assert obs.messages == []

    def test_observation_as_vector(self):
        """Test converting observation to flat vector."""
        obs = Observation(
            local={"b": 2.0, "a": 1.0},  # Test alphabetical sorting
            global_info={"voltage": 1.05},
        )

        vec = obs.as_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        # Should contain: a=1.0, b=2.0, voltage=1.05 (alphabetically sorted)
        expected = np.array([1.0, 2.0, 1.05], dtype=np.float32)
        np.testing.assert_array_almost_equal(vec, expected)

    def test_observation_with_arrays(self):
        """Test observation with numpy arrays."""
        obs = Observation(
            local={"state": np.array([1.0, 2.0, 3.0])},
        )

        vec = obs.as_vector()
        np.testing.assert_array_almost_equal(vec, [1.0, 2.0, 3.0])


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            sender="agent_1",
            content={"price": 50.0},
            timestamp=5.0,
            priority=1,
        )

        assert msg.sender == "agent_1"
        assert msg.content["price"] == 50.0
        assert msg.timestamp == 5.0
        assert msg.priority == 1


class TestAgent:
    """Tests for Agent base class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        obs_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        agent = DummyAgent(
            agent_id="test_agent",
            level=1,
            action_space=action_space,
            observation_space=obs_space,
        )

        assert agent.agent_id == "test_agent"
        assert agent.level == 1
        assert agent.action_space == action_space
        assert agent.observation_space == obs_space
        assert agent.mailbox == []
        assert agent._timestep == 0.0

    def test_agent_observe(self):
        """Test agent observation extraction."""
        agent = DummyAgent(
            agent_id="test",
            level=1,
            action_space=Box(low=-1, high=1, shape=(1,)),
        )

        global_state = {"test_value": 42.0}
        obs = agent.observe(global_state)

        assert obs.local["value"] == 42.0
        assert obs.timestamp == 0.0

    def test_agent_message_passing(self):
        """Test sending and receiving messages."""
        agent = DummyAgent(
            agent_id="receiver",
            level=1,
            action_space=Box(low=-1, high=1, shape=(1,)),
        )

        # Send message
        msg = agent.send_message(
            content={"data": 100},
            recipients=["other_agent"],
            priority=2,
        )

        assert msg.sender == "receiver"
        assert msg.content["data"] == 100
        assert msg.priority == 2

        # Receive message
        incoming = Message(sender="other_agent", content={"signal": 5})
        agent.receive_message(incoming)

        assert len(agent.mailbox) == 1
        assert agent.mailbox[0].sender == "other_agent"
        assert agent.mailbox[0].content["signal"] == 5

    def test_agent_clear_mailbox(self):
        """Test clearing mailbox."""
        agent = DummyAgent(
            agent_id="test",
            level=1,
            action_space=Box(low=-1, high=1, shape=(1,)),
        )

        # Add messages
        agent.receive_message(Message("sender1", {"a": 1}))
        agent.receive_message(Message("sender2", {"b": 2}))

        assert len(agent.mailbox) == 2

        # Clear
        messages = agent.clear_mailbox()
        assert len(messages) == 2
        assert len(agent.mailbox) == 0

    def test_agent_reset(self):
        """Test agent reset."""
        agent = DummyAgent(
            agent_id="test",
            level=1,
            action_space=Box(low=-1, high=1, shape=(1,)),
        )

        # Add some state
        agent.receive_message(Message("sender", {}))
        agent.update_timestep(10.0)

        # Reset
        agent.reset(seed=42)

        assert len(agent.mailbox) == 0
        assert agent._timestep == 0.0

    def test_agent_update_timestep(self):
        """Test timestep updates."""
        agent = DummyAgent(
            agent_id="test",
            level=1,
            action_space=Box(low=-1, high=1, shape=(1,)),
        )

        agent.update_timestep(5.0)
        assert agent._timestep == 5.0

        agent.update_timestep(10.0)
        assert agent._timestep == 10.0


class TestRandomPolicy:
    """Tests for RandomPolicy."""

    def test_random_policy_sampling(self):
        """Test random policy action sampling."""
        action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        policy = RandomPolicy(action_space, seed=42)

        obs = Observation()
        action1 = policy.forward(obs)
        action2 = policy.forward(obs)

        assert action1.shape == (3,)
        assert action2.shape == (3,)
        # Actions should be different (with high probability)
        assert not np.allclose(action1, action2)

    def test_random_policy_reset(self):
        """Test random policy reset."""
        action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        policy = RandomPolicy(action_space)

        policy.reset()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
