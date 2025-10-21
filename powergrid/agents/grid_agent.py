"""Grid-level coordinator agents for hierarchical control.

GridAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpaceDict

from .base import Agent, Observation, Message, AgentID
from ..core.policies import Policy
from .device_agent import DeviceAgent
from ..core.protocols import VerticalProtocol, NoProtocol, Protocol


class GridAgent(Agent):
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
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        centralized: bool = False,
    ):
        """Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            subordinates: List of device agents to coordinate
            protocol: Protocol for coordinating subordinate devices (agent-owned)
            policy: High-level policy (optional)
            centralized: If True, outputs single action for all subordinates
        """
        # Temporarily set subordinates for space building
        self.subordinates = {agent.agent_id: agent for agent in subordinates}

        action_space = self._get_action_space(subordinates, centralized)
        observation_space = self._get_observation_space(subordinates)

        super().__init__(
            agent_id=agent_id,
            level=2,  # Grid level
            observation_space=observation_space,
            action_space=action_space,
        )
        self.protocol = protocol
        self.policy = policy
        self.centralized = centralized

    def _get_action_space(
        self,
        subordinates: List[DeviceAgent],
        centralized: bool,
    ) -> gym.Space:
        """Build action space for grid.

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

    def _get_observation_space(
        self,
        subordinates: List[DeviceAgent],
    ) -> gym.Space:
        """Build observation space (aggregate of subordinate obs).

        Args:
            subordinates: List of subordinate agents

        Returns:
            Observation space
        """
        # Calculate total observation size
        # Each subordinate's observation space + global_info from first subordinate
        total_dim = sum(
            agent.observation_space.shape[0]
            if isinstance(agent.observation_space, Box)
            else 10  # Default size
            for agent in subordinates
        )

        # Note: GridAgent.observe() returns nested subordinate_states + global_info
        # When flattened with as_vector(), both local and global_info are included
        # The actual size will match what subordinates return
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

    def coordinate_subordinates(self, global_state: Dict) -> None:
        """
        Coordinate subordinate devices using vertical protocol.

        This method is called by the environment during the coordination phase.
        The GridAgent runs its vertical protocol and sends messages to subordinates.

        Args:
            global_state: Global environment state for subordinates to observe
        """
        if not self.subordinates:
            return

        # Collect subordinate observations
        sub_obs = {
            sub_id: sub_agent.observe(global_state)
            for sub_id, sub_agent in self.subordinates.items()
        }

        # Run vertical protocol (coordinator_action could come from self.policy)
        coordinator_action = None  # Or self.policy.forward(...) if using learned coordination
        signals = self.protocol.coordinate(sub_obs, coordinator_action)

        # Send coordination signals to subordinates
        for sub_id, signal in signals.items():
            if signal:  # Only send non-empty signals
                msg = self.send_message(content=signal, recipients=[sub_id])
                self.subordinates[sub_id].receive_message(msg)

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
        return f"GridAgent(id={self.agent_id}, subordinates={num_subs}, protocol={protocol_name})"
