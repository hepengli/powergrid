"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in power grid systems.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import gymnasium as gym

from ..agents.base import Observation


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
