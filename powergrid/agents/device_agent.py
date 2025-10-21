"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.
"""
from builtins import float
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from .base import Agent, Observation
from ..core.policies import Policy
from ..core.actions import Action
from ..core.state import DeviceState
from ..core.protocols import VerticalProtocol, NoProtocol, Protocol


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
        self.action_callback = False  # True if external logic sets state
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

    def set_action_space(self) -> None:
        """Define action space based on underlying device action."""
        raise NotImplementedError


    def _get_action_space(self) -> gym.Space:
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
                    return MultiDiscrete([cats]*low.size)
            return Box(
                low=low,
                high=high,
                dtype=np.float32,
            )

        # Discrete actions
        if action.dim_d > 0:
            if not action.ncats:
                raise ValueError("Device action.ncats must be set a positive integer for discrete actions.")
            
            return Discrete(action.ncats)
        
        raise ValueError("Device must have either continuous or discrete actions defined.") 

    def _get_observation_space(self) -> gym.Space:
        return Box(
            low=-np.inf, 
            high=np.inf, 
            shape=self.state.as_vector().shape, 
            dtype=np.float32
        )

    def observe(self, global_state: Dict[str, Any]) -> Observation:
        obs = Observation(timestamp=self._timestep)

        # Local device state only
        obs.local = self.state.as_vector().astype(np.float32)

        # Messages from coordinator/other agents (e.g., price signals)
        obs.messages = self.mailbox.copy()

        return obs

    def act(self, observation: Observation, given_action: Any) -> Any:
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
        return action

    def _set_device_action(self, action: Any) -> None:
        """Set action on underlying device.

        Args:
            action: Action from policy
        """
        # TODO: verify action format matches device.action
        assert action.size == self.action.dim_c + self.action.dim_d
        self.action.c[:] = action[:self.action.c.size]
        if self.config.get('discrete_action'):
            cats = self.config.get('discrete_action_cats')
            low, high = self.action.range
            acts = np.linspace(low, high, cats).transpose()
            self.action.c[:] = [a[action[i]] for i, a in enumerate(acts)]
        self.action.d[:] = action[self.action.c.size:]

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent and underlying device.

        Args:
            seed: Random seed
            **kwargs: Additional reset params (e.g., init_soc for ESS)
        """
        super().reset()

    def get_reward(self) -> float:
        """Get reward signal from device cost/safety.

        Returns:
            Negative cost minus safety penalty
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


    def update_state(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def update_cost_safety(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def reset_device(self, *args, **kwargs) -> None:
        raise NotImplementedError

    # Optional hook
    def feasible_action(self) -> None:
        """Clamp/adjust current action so it is feasible for this step."""
        return None