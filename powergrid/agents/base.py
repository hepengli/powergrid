"""Base agent abstraction for hierarchical multi-agent control.

This module provides the core abstractions for agents in the PowerGrid platform,
supporting hierarchical control, agent-to-agent communication, and flexible
observation/action interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np


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
        vec = self._flatten_dict(self.local, vec)

        # Flatten global info
        vec = self._flatten_dict(self.global_info, vec)

        return vec.astype(np.float32)

    def _flatten_dict(self, d: Dict, vec: np.ndarray) -> np.ndarray:
        """Recursively flatten a dictionary into a numpy array.

        Args:
            d: Dictionary to flatten
            vec: Current vector to append to

        Returns:
            Updated vector
        """
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, (int, float)):
                vec = np.append(vec, np.float32(val))
            elif isinstance(val, np.ndarray):
                vec = np.concatenate([vec, val.ravel().astype(np.float32)])
            elif isinstance(val, dict):
                # Recursively flatten nested dicts
                vec = self._flatten_dict(val, vec)
        return vec


@dataclass
class Message:
    """Inter-agent communication message.

    Attributes:
        sender: ID of sending agent
        content: Message payload (e.g., price signals, setpoints, constraints)
        timestamp: Time when message was sent
    """
    sender: AgentID
    content: Dict[str, Any]
    recipient: Optional[Union[AgentID, List[AgentID]]] = None  # None = broadcast
    timestamp: float = 0.0

    # TODO: add more attributes like expiration, priority, etc.

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

    # Core agent methods (lifecycle)
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent to initial state.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional reset parameters
        """
        self.mailbox.clear()
        self._timestep = 0.0

    @abstractmethod
    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
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
    def act(self, observation: Observation, *args, **kwargs) -> Any:
        """Compute action from observation.

        Args:
            observation: Structured observation from observe()

        Returns:
            Action in the format defined by action_space
        """
        pass

    # Communication methods
    def send_message(
        self,
        content: Dict[str, Any],
        recipients: Optional[Union[AgentID, List[AgentID]]] = None,
    ) -> Message:
        """Create a message to send to other agents.

        Args:
            content: Message payload
            recipients: List of recipient agent IDs (None = broadcast)

        Returns:
            Message object (to be delivered by environment)
        """
        return Message(
            sender=self.agent_id,
            content=content,
            recipient=recipients,
            timestamp=self._timestep,
        )

    def receive_message(self, message: Message) -> None:
        """Handle incoming communication from another agent.

        Default behavior: append to mailbox. Override for custom handling.

        Args:
            message: Message from another agent
        """
        self.mailbox.append(message)

    def clear_mailbox(self) -> List[Message]:
        """Clear and return all messages from mailbox.

        Returns:
            List of messages received since last clear
        """
        messages = self.mailbox.copy()
        self.mailbox.clear()
        return messages

    # Utility methods
    def update_timestep(self, timestep: float) -> None:
        """Update internal timestep counter.

        Args:
            timestep: Current simulation time
        """
        self._timestep = timestep

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level})"
