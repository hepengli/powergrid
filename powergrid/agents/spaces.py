"""Utilities for building and manipulating action/observation spaces.

This module provides helpers for composing heterogeneous agent spaces,
partial observability, and action masking.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict


def compose_action_spaces(
    spaces: Dict[str, gym.Space],
    mode: str = "dict",
) -> gym.Space:
    """Compose multiple action spaces into a single space.

    Args:
        spaces: Dict mapping agent_id to action space
        mode: Composition mode:
            - "dict": Return Dict space with agent_id keys
            - "flatten": Flatten all spaces into single Box

    Returns:
        Composed action space
    """
    if not spaces:
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    if mode == "dict":
        return SpaceDict(spaces)
    elif mode == "flatten":
        # Flatten all continuous spaces
        total_dim = 0
        lows, highs = [], []
        for space in spaces.values():
            if isinstance(space, Box):
                total_dim += space.shape[0]
                lows.extend(space.low.ravel().tolist())
                highs.extend(space.high.ravel().tolist())
            else:
                raise ValueError(f"Cannot flatten non-Box space: {type(space)}")

        return Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compose_observation_spaces(
    spaces: Dict[str, gym.Space],
    mode: str = "dict",
) -> gym.Space:
    """Compose multiple observation spaces.

    Args:
        spaces: Dict mapping agent_id to observation space
        mode: Composition mode ("dict" or "flatten")

    Returns:
        Composed observation space
    """
    return compose_action_spaces(spaces, mode)


def split_flat_action(
    flat_action: np.ndarray,
    action_spaces: Dict[str, gym.Space],
) -> Dict[str, np.ndarray]:
    """Split flattened action into per-agent actions.

    Args:
        flat_action: Flattened action array
        action_spaces: Dict mapping agent_id to action space

    Returns:
        Dict mapping agent_id to action
    """
    actions = {}
    offset = 0
    for agent_id, space in action_spaces.items():
        if isinstance(space, Box):
            dim = space.shape[0]
            actions[agent_id] = flat_action[offset : offset + dim]
            offset += dim
        else:
            raise ValueError(f"Cannot split non-Box space: {type(space)}")
    return actions


def create_partial_obs_space(
    full_space: gym.Space,
    visible_keys: Optional[List[str]] = None,
) -> gym.Space:
    """Create partial observation space from full space.

    Args:
        full_space: Full observation space
        visible_keys: Keys to include (for Dict spaces)

    Returns:
        Partial observation space
    """
    if visible_keys is None or not isinstance(full_space, SpaceDict):
        return full_space

    partial_spaces = {
        key: space for key, space in full_space.spaces.items() if key in visible_keys
    }
    return SpaceDict(partial_spaces)


class ActionMask:
    """Action masking for asynchronous multi-agent execution.

    Some agents may not act at every timestep (e.g., different control frequencies).
    ActionMask tracks which agents should act at each step.
    """

    def __init__(self, agent_ids: List[str], frequencies: Optional[Dict[str, int]] = None):
        """Initialize action mask.

        Args:
            agent_ids: List of all agent IDs
            frequencies: Dict mapping agent_id to control frequency (steps)
                        If None, all agents act every step
        """
        self.agent_ids = agent_ids
        self.frequencies = frequencies or {agent_id: 1 for agent_id in agent_ids}
        self._timestep = 0

    def get_active_agents(self, timestep: Optional[int] = None) -> List[str]:
        """Get list of agents that should act at current timestep.

        Args:
            timestep: Current timestep (uses internal counter if None)

        Returns:
            List of active agent IDs
        """
        if timestep is None:
            timestep = self._timestep

        active = []
        for agent_id in self.agent_ids:
            freq = self.frequencies[agent_id]
            if timestep % freq == 0:
                active.append(agent_id)

        return active

    def step(self) -> None:
        """Increment internal timestep counter."""
        self._timestep += 1

    def reset(self) -> None:
        """Reset timestep counter."""
        self._timestep = 0

    def get_mask(self, timestep: Optional[int] = None) -> Dict[str, bool]:
        """Get binary mask indicating which agents are active.

        Args:
            timestep: Current timestep

        Returns:
            Dict mapping agent_id to bool (True = active)
        """
        active = self.get_active_agents(timestep)
        return {agent_id: agent_id in active for agent_id in self.agent_ids}


def box_space_from_dims(
    dims: int,
    low: Union[float, np.ndarray] = -np.inf,
    high: Union[float, np.ndarray] = np.inf,
) -> Box:
    """Create Box space from dimension count.

    Args:
        dims: Number of dimensions
        low: Lower bound(s)
        high: Upper bound(s)

    Returns:
        Box space
    """
    if np.isscalar(low):
        low = np.full(dims, low, dtype=np.float32)
    if np.isscalar(high):
        high = np.full(dims, high, dtype=np.float32)

    return Box(low=low, high=high, dtype=np.float32)


def validate_action(action: Any, action_space: gym.Space) -> bool:
    """Validate that action conforms to action space.

    Args:
        action: Action to validate
        action_space: Expected action space

    Returns:
        True if valid
    """
    try:
        return action in action_space
    except:
        return False
