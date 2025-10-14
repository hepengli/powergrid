"""Agent abstraction layer for multi-agent power grid control."""

from .base import Agent, Observation, Message, AgentID
from .policies import Policy, RandomPolicy
from .device_agent import DeviceAgent
from .grid_agent import (
    GridCoordinatorAgent,
    Protocol,
    NoProtocol,
    PriceSignalProtocol,
    SetpointProtocol,
)
from .spaces import (
    compose_action_spaces,
    compose_observation_spaces,
    split_flat_action,
    ActionMask,
)

__all__ = [
    # Base abstractions
    "Agent",
    "Observation",
    "Message",
    "AgentID",
    # Policies
    "Policy",
    "RandomPolicy",
    # Agent types
    "DeviceAgent",
    "GridCoordinatorAgent",
    # Protocols
    "Protocol",
    "NoProtocol",
    "PriceSignalProtocol",
    "SetpointProtocol",
    # Space utilities
    "compose_action_spaces",
    "compose_observation_spaces",
    "split_flat_action",
    "ActionMask",
]
