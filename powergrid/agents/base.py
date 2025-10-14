"""Base agent abstraction for hierarchical multi-agent control.

This module provides the core abstractions for agents in the PowerGrid platform,
supporting hierarchical control, agent-to-agent communication, and flexible
observation/action interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
import gymnasium as gym

# Type aliases
AgentID = str
Array = np.ndarray


@dataclass
class Observation:
    """Structured observation for an agent.

    Attributes:
        local: Local agent state (e.g., device P, Q, SOC)
        global_info: Global information visible to this agent (e.g., bus voltages)
        messages: Communication messages from other agents
        timestamp: Current simulation time
    """
    local: Dict[str, Any] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)
    messages: List['Message'] = field(default_factory=list)
    timestamp: float = 0.0

    def as_vector(self) -> Array:
        """Convert observation to flat numpy array for RL algorithms.

        Returns:
            Flattened observation vector
        """
        vec = np.array([], dtype=np.float32)

        # Flatten local state
        for key in sorted(self.local.keys()):
            val = self.local[key]
            if isinstance(val, (int, float)):
                vec = np.append(vec, np.float32(val))
            elif isinstance(val, np.ndarray):
                vec = np.concatenate([vec, val.ravel().astype(np.float32)])

        # Flatten global info
        for key in sorted(self.global_info.keys()):
            val = self.global_info[key]
            if isinstance(val, (int, float)):
                vec = np.append(vec, np.float32(val))
            elif isinstance(val, np.ndarray):
                vec = np.concatenate([vec, val.ravel().astype(np.float32)])

        return vec.astype(np.float32)


@dataclass
class Message:
    """Inter-agent communication message.

    Attributes:
        sender: ID of sending agent
        content: Message payload (e.g., price signals, setpoints, constraints)
        timestamp: Time when message was sent
        priority: Message priority for scheduling (higher = more urgent)
    """
    sender: AgentID
    content: Dict[str, Any]
    timestamp: float = 0.0
    priority: int = 0


class Agent(ABC):
    """Abstract base class for all agents in the hierarchy.

    Agents can operate at different levels of the control hierarchy:
    - Device level: Individual DERs (DG, ESS, RES, etc.)
    - Grid level: Microgrid controllers, substations
    - System level: ISO, market operator

    Key responsibilities:
    - Observe: Extract relevant information from global state
    - Act: Compute actions based on observations
    - Communicate: Send/receive messages to/from other agents
    - Reset: Initialize agent state for new episodes

    Attributes:
        agent_id: Unique identifier for this agent
        level: Hierarchy level (1=device, 2=grid, 3=system)
        observation_space: Gymnasium space for observations
        action_space: Gymnasium space for actions
    """

    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
    ):
        """Initialize agent.

        Args:
            agent_id: Unique identifier
            level: Hierarchy level (1=device, 2=grid, 3=system)
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
        """
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space
        self.mailbox: List[Message] = []
        self._timestep = 0.0

    @abstractmethod
    def observe(self, global_state: Dict[str, Any]) -> Observation:
        """Extract relevant observations from global state.

        Args:
            global_state: Complete environment state including:
                - bus voltages/angles
                - device states
                - power flow results
                - dataset values (load, price, etc.)

        Returns:
            Structured observation for this agent
        """
        pass

    @abstractmethod
    def act(self, observation: Observation) -> Any:
        """Compute action from observation.

        Args:
            observation: Structured observation from observe()

        Returns:
            Action in the format defined by action_space
        """
        pass

    def receive_message(self, message: Message) -> None:
        """Handle incoming communication from another agent.

        Default behavior: append to mailbox. Override for custom handling.

        Args:
            message: Message from another agent
        """
        self.mailbox.append(message)

    def send_message(
        self,
        content: Dict[str, Any],
        recipients: Optional[List[AgentID]] = None,
        priority: int = 0,
    ) -> Message:
        """Create a message to send to other agents.

        Args:
            content: Message payload
            recipients: List of recipient agent IDs (None = broadcast)
            priority: Message priority (higher = more urgent)

        Returns:
            Message object (to be delivered by environment)
        """
        return Message(
            sender=self.agent_id,
            content=content,
            timestamp=self._timestep,
            priority=priority,
        )

    def clear_mailbox(self) -> List[Message]:
        """Clear and return all messages from mailbox.

        Returns:
            List of messages received since last clear
        """
        messages = self.mailbox.copy()
        self.mailbox.clear()
        return messages

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent to initial state.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional reset parameters
        """
        self.mailbox.clear()
        self._timestep = 0.0

    def update_timestep(self, timestep: float) -> None:
        """Update internal timestep counter.

        Args:
            timestep: Current simulation time
        """
        self._timestep = timestep

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"


class Policy(ABC):
    """Abstract policy interface for agent decision-making.

    Policies can be:
    - Learned (RL algorithms)
    - Rule-based (heuristics, classical control)
    - Optimization-based (MPC, optimal control)
    """

    @abstractmethod
    def forward(self, observation: Observation) -> Any:
        """Compute action from observation.

        Args:
            observation: Agent observation

        Returns:
            Action
        """
        pass

    def reset(self) -> None:
        """Reset policy state (e.g., hidden states for RNNs)."""
        pass


class RandomPolicy(Policy):
    """Random policy that samples from action space."""

    def __init__(self, action_space: gym.Space, seed: Optional[int] = None):
        """Initialize random policy.

        Args:
            action_space: Gymnasium action space
            seed: Random seed
        """
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def forward(self, observation: Observation) -> Any:
        """Sample random action.

        Args:
            observation: Ignored for random policy

        Returns:
            Random action from action_space
        """
        return self.action_space.sample()
