"""Policy interfaces for agent decision-making.

This module provides abstract policy interfaces and common implementations
for agent control in power grid systems.
"""

from abc import ABC, abstractmethod
from typing import Any

from powergrid.agents.base import Observation


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
