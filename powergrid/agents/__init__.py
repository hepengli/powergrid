"""Agent abstraction layer for multi-agent power grid control."""

from .base import Agent, Observation, Message, AgentID
from .policies import Policy, RandomPolicy
from .device_agent import DeviceAgent
from .grid_agent import (
    GridAgent,
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
    "GridAgent",
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
