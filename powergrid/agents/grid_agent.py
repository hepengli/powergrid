"""Grid-level coordinator agents for hierarchical control.

GridCoordinatorAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpaceDict

from .base import Agent, Observation, Policy, Message, AgentID
from .device_agent import DeviceAgent


class Protocol:
    """Abstract coordination protocol interface.

    Protocols define how grid agents coordinate subordinate device agents.
    Examples: price signals, setpoints, ADMM, droop control.
    """

    def coordinate(
        self,
        observations: Dict[AgentID, Observation],
        coordinator_action: Optional[Any] = None,
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Compute coordination signals for subordinate agents.

        Args:
            observations: Observations from all subordinate agents
            coordinator_action: Optional action from coordinator's policy

        Returns:
            Dict mapping agent_id to coordination signal (e.g., setpoint, price)
        """
        raise NotImplementedError


class NoProtocol(Protocol):
    """No coordination - subordinate agents act independently."""

    def coordinate(
        self,
        observations: Dict[AgentID, Observation],
        coordinator_action: Optional[Any] = None,
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Return empty coordination signals.

        Args:
            observations: Ignored
            coordinator_action: Ignored

        Returns:
            Empty dict for each agent
        """
        return {agent_id: {} for agent_id in observations}


class PriceSignalProtocol(Protocol):
    """Price-based coordination via marginal price signals.

    Coordinator broadcasts a price, subordinate agents optimize locally.
    """

    def __init__(self, initial_price: float = 50.0):
        """Initialize price signal protocol.

        Args:
            initial_price: Initial electricity price ($/MWh)
        """
        self.price = initial_price

    def coordinate(
        self,
        observations: Dict[AgentID, Observation],
        coordinator_action: Optional[Any] = None,
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Broadcast price signal to all agents.

        Args:
            observations: Observations from subordinates
            coordinator_action: New price (if provided)

        Returns:
            Price signal for each agent
        """
        # Update price from coordinator action if provided
        if coordinator_action is not None:
            if isinstance(coordinator_action, dict):
                self.price = coordinator_action.get("price", self.price)
            else:
                self.price = float(coordinator_action)

        # Broadcast to all agents
        return {
            agent_id: {"price": self.price}
            for agent_id in observations
        }


class SetpointProtocol(Protocol):
    """Setpoint-based coordination.

    Coordinator computes power setpoints, subordinate agents track them.
    """

    def coordinate(
        self,
        observations: Dict[AgentID, Observation],
        coordinator_action: Optional[Any] = None,
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Distribute setpoints to subordinate agents.

        Args:
            observations: Observations from subordinates
            coordinator_action: Dict of {agent_id: setpoint}

        Returns:
            Setpoint for each agent
        """
        if coordinator_action is None:
            # No setpoint, agents act independently
            return {agent_id: {} for agent_id in observations}

        # Distribute setpoints
        signals = {}
        for agent_id in observations:
            if agent_id in coordinator_action:
                signals[agent_id] = {"setpoint": coordinator_action[agent_id]}
            else:
                signals[agent_id] = {}

        return signals


class GridCoordinatorAgent(Agent):
    """Grid-level agent that coordinates multiple device agents.

    GridCoordinator implements hierarchical control by:
    1. Collecting observations from subordinate agents
    2. Running coordination protocol (e.g., compute prices, setpoints)
    3. Broadcasting coordination signals via messages
    4. Optionally aggregating subordinate actions into single action

    Attributes:
        subordinates: List of device agents managed by this coordinator
        protocol: Coordination protocol
        policy: High-level policy (optional, for learned coordination)
        centralized: If True, output single action for all subordinates
    """

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: List[DeviceAgent],
        protocol: Optional[Protocol] = None,
        policy: Optional[Policy] = None,
        centralized: bool = False,
    ):
        """Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: List of device agents to coordinate
            protocol: Coordination protocol (defaults to NoProtocol)
            policy: High-level policy (optional)
            centralized: If True, outputs single action for all subordinates
        """
        action_space = self._build_action_space(subordinates, centralized)
        observation_space = self._build_observation_space(subordinates)

        super().__init__(
            agent_id=agent_id,
            level=2,  # Grid level
            observation_space=observation_space,
            action_space=action_space,
        )

        self.subordinates = {agent.agent_id: agent for agent in subordinates}
        self.protocol = protocol or NoProtocol()
        self.policy = policy
        self.centralized = centralized

    def _build_action_space(
        self,
        subordinates: List[DeviceAgent],
        centralized: bool,
    ) -> gym.Space:
        """Build action space for coordinator.

        Args:
            subordinates: List of subordinate agents
            centralized: Whether to use centralized action space

        Returns:
            Action space
        """
        if centralized:
            # Concatenate all subordinate action spaces
            total_dim = sum(
                agent.action_space.shape[0]
                if isinstance(agent.action_space, Box)
                else 1
                for agent in subordinates
            )
            return Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        else:
            # Coordinator outputs coordination signals (e.g., price)
            # For now, single continuous value (e.g., price or reserve requirement)
            return Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _build_observation_space(
        self,
        subordinates: List[DeviceAgent],
    ) -> gym.Space:
        """Build observation space (aggregate of subordinate obs).

        Args:
            subordinates: List of subordinate agents

        Returns:
            Observation space
        """
        # Aggregate observation space from subordinates
        # For simplicity, concatenate all subordinate obs spaces
        total_dim = sum(
            agent.observation_space.shape[0]
            if isinstance(agent.observation_space, Box)
            else 10  # Default size
            for agent in subordinates
        )
        return Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def observe(self, global_state: Dict[str, Any]) -> Observation:
        """Collect observations from subordinate agents.

        Args:
            global_state: Environment state

        Returns:
            Aggregated observation from all subordinates
        """
        obs = Observation(timestamp=self._timestep)

        # Collect subordinate observations
        sub_obs = {}
        for agent_id, agent in self.subordinates.items():
            sub_obs[agent_id] = agent.observe(global_state)

        # Aggregate local state
        obs.local["subordinate_states"] = {
            agent_id: sub_ob.local for agent_id, sub_ob in sub_obs.items()
        }

        # Aggregate global info (take from first subordinate)
        if sub_obs:
            first_obs = next(iter(sub_obs.values()))
            obs.global_info = first_obs.global_info.copy()

        # Include messages
        obs.messages = self.mailbox.copy()

        return obs

    def act(self, observation: Observation) -> Any:
        """Compute coordination action and distribute to subordinates.

        Args:
            observation: Aggregated observation

        Returns:
            Coordinator action (or None if decentralized)
        """
        # Get coordinator action from policy if available
        coordinator_action = None
        if self.policy is not None:
            coordinator_action = self.policy.forward(observation)

        # Run coordination protocol
        subordinate_obs = {
            agent_id: Observation(local=observation.local["subordinate_states"][agent_id])
            for agent_id in self.subordinates
        }
        signals = self.protocol.coordinate(subordinate_obs, coordinator_action)

        # Send coordination signals as messages
        for agent_id, signal in signals.items():
            if signal:  # Only send non-empty signals
                msg = self.send_message(
                    content=signal,
                    recipients=[agent_id],
                    priority=1,
                )
                # Deliver message directly to subordinate
                self.subordinates[agent_id].receive_message(msg)

        return coordinator_action if self.centralized else None

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset coordinator and all subordinates.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset subordinates
        for agent in self.subordinates.values():
            agent.reset(seed=seed, **kwargs)

        # Reset policy
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_subordinate_actions(
        self,
        observations: Dict[AgentID, Observation],
    ) -> Dict[AgentID, Any]:
        """Get actions from all subordinate agents.

        Args:
            observations: Observations for each subordinate

        Returns:
            Dict mapping agent_id to action
        """
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.subordinates[agent_id].act(obs)
        return actions

    def __repr__(self) -> str:
        num_subs = len(self.subordinates)
        protocol_name = self.protocol.__class__.__name__
        return f"GridCoordinatorAgent(id={self.agent_id}, subordinates={num_subs}, protocol={protocol_name})"
