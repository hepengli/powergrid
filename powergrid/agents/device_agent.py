"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.
"""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from powergrid.agents.base import Agent, Observation
from powergrid.core.action import Action
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.state import DeviceState


class DeviceAgent(Agent):
    """Wraps a Device as an autonomous agent.

    DeviceAgent maintains compatibility with the existing Device interface while
    adding agent capabilities like observation extraction, communication, and
    pluggable policies.

    DeviceAgent only observes its local device state. Global information should
    be provided by parent GridAgent through coordination protocols/messages.

    Attributes:
        device: Underlying Device object (DG, ESS, RES, etc.)
        policy: Decision-making policy (learned or rule-based)
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ):
        """Initialize device agent.

        Args:
            device: Device to wrap (must have set_action_space called)
            policy: Decision policy (defaults to random)
            agent_id: Agent ID (defaults to device.name)
        """
        self.state = DeviceState()
        self.action = Action()
        self.action_callback = device_config.get('action_callback', False)  # True if external logic sets state
        self.cost = 0.0
        self.safety = 0.0
        self.adversarial = False
        self.config = device_config
        self.policy = policy
        self.protocol = protocol

        self.set_action_space()
        self.set_device_state()

        assert agent_id or "name" in device_config, "DeviceAgent requires agent_id or device_config['name']"
        agent_id = agent_id or device_config["name"]

        super().__init__(
            agent_id=agent_id,
            level=1,  # Device level
            action_space=self._get_action_space(),
            observation_space=self._get_observation_space(),
        )

    # Initialization methods
    def set_action_space(self) -> None:
        """Define action space based on underlying device action.

        This method should be overridden by subclasses to define device-specific action spaces.
        """
        pass

    def set_device_state(self) -> None:
        """Initialize device-specific state attributes.

        This method can be overridden by subclasses to initialize device-specific state.
        """
        pass

    # Space construction methods
    def _get_action_space(self) -> gym.Space:
        """Construct Gymnasium action space from device action configuration.

        Returns:
            Gymnasium space for device actions

        Raises:
            ValueError: If action configuration is invalid
        """
        # Devices with action_callback don't have actions (controlled externally)
        if self.action_callback:
            return Discrete(1)  # Dummy action space
        
        action = self.action

        # Continuous actions
        if action.dim_c > 0:
            if action.range is None:
                raise ValueError("Device action.range must be set for continuous actions.")
            low, high = action.range
            if self.config.get('discrete_action'):
                cats = self.config.get('discrete_action_cats')
                if low.size == 1:
                    return Discrete(cats)
                else:
                    return MultiDiscrete([cats] * low.size)
            return Box(
                low=low,
                high=high,
                dtype=np.float32,
            )

        # Discrete actions
        if action.dim_d > 0:
            if not action.ncats:
                raise ValueError("Device action.ncats must be set to a positive integer for discrete actions.")

            return Discrete(action.ncats)

        raise ValueError("Device must have either continuous or discrete actions defined.")

    def _get_observation_space(self) -> gym.Space:
        """Construct Gymnasium observation space from device state.

        Returns:
            Gymnasium space for device observations
        """
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self.state.as_vector().shape,
            dtype=np.float32
        )

    # Core agent lifecycle methods
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent and underlying device.

        Args:
            seed: Random seed
            **kwargs: Additional reset params (e.g., init_soc for ESS)
        """
        super().reset()
        self.reset_device(**kwargs)

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract device observation from global state.

        Args:
            global_state: Complete environment state (optional)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Structured observation for this device
        """
        obs = Observation(
            timestamp=self._timestep,
            messages=self.mailbox.copy()
        )

        # Local device state only
        obs.local['state'] = self.state.as_vector().astype(np.float32)

        # TODO: aggregate global info if needed
        obs.global_info = global_state

        return obs

    def act(self, observation: Observation, given_action: Any = None) -> Any:
        """Compute action using policy.

        Args:
            observation: Structured observation
            given_action: Action provided by the parent grid agent

        Returns:
            Action in format defined by action_space
        """
        if given_action:
            action = given_action
        else:
            assert self.policy is not None, "DeviceAgent requires a policy to compute actions."
            action = self.policy.forward(observation)

        self._set_device_action(action)
        # TODO: Add communication logic (send/receive message) if needed

        return action

    def _set_device_action(self, action: Any) -> None:
        """Set action on underlying device.

        Args:
            action: Action from policy (numpy array)
        """
        # TODO: verify action format matches policy forward output
        assert action.size == self.action.dim_c + self.action.dim_d
        self.action.c[:] = action[:self.action.c.size]
        if self.config.get('discrete_action'):
            cats = self.config.get('discrete_action_cats')
            low, high = self.action.range
            acts = np.linspace(low, high, cats).transpose()
            self.action.c[:] = [a[action[i]] for i, a in enumerate(acts)]
        self.action.d[:] = action[self.action.c.size:]

    # Device-specific methods (to be implemented by subclasses)
    def reset_device(self, *args, **kwargs) -> None:
        """Reset device to initial state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def update_state(self, *args, **kwargs) -> None:
        """Update device state (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def update_cost_safety(self, *args, **kwargs) -> None:
        """Update device cost and safety metrics (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """Get reward signal from device cost/safety.

        Returns:
            Reward value (negative cost minus safety penalty)
        """
        return {"cost": self.cost, "safety": self.safety, "adversarial": self.adversarial}

    def feasible_action(self) -> None:
        """Clamp/adjust current action to ensure feasibility.

        This is an optional hook that can be overridden by subclasses to
        enforce device-specific constraints on actions.
        """
        return None

    # Utility methods
    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String representation
        """
        raise NotImplementedError
