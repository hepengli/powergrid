"""Device-level agents that wrap Device objects as autonomous agents.

DeviceAgent provides a bridge between the existing Device abstraction and the
new Agent abstraction, enabling devices to participate in multi-agent control.
"""

from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpaceDict

from .base import Agent, Observation, Policy, RandomPolicy
from ..devices.base import Device


class DeviceAgent(Agent):
    """Wraps a Device as an autonomous agent.

    DeviceAgent maintains compatibility with the existing Device interface while
    adding agent capabilities like observation extraction, communication, and
    pluggable policies.

    Attributes:
        device: Underlying Device object (DG, ESS, RES, etc.)
        policy: Decision-making policy (learned or rule-based)
        partial_obs: Whether to use partial observability (only local info)
    """

    def __init__(
        self,
        device: Device,
        policy: Optional[Policy] = None,
        partial_obs: bool = False,
        agent_id: Optional[str] = None,
    ):
        """Initialize device agent.

        Args:
            device: Device to wrap (must have set_action_space called)
            policy: Decision policy (defaults to random)
            partial_obs: If True, only observe local device state
            agent_id: Agent ID (defaults to device.name)
        """
        agent_id = agent_id or getattr(device, "name", f"device_{id(device)}")

        # Build action space from device
        action_space = self._build_action_space(device)

        # Build observation space
        observation_space = self._build_observation_space(device, partial_obs)

        super().__init__(
            agent_id=agent_id,
            level=1,  # Device level
            observation_space=observation_space,
            action_space=action_space,
        )

        self.device = device
        self.policy = policy or RandomPolicy(action_space)
        self.partial_obs = partial_obs
        self._bus_id = getattr(device, "bus", None)

    def _build_action_space(self, device: Device) -> gym.Space:
        """Build Gymnasium action space from device action spec.

        Args:
            device: Device with action attribute

        Returns:
            Gymnasium action space
        """
        action = device.action
        spaces = {}

        # Continuous actions
        if action.dim_c > 0 and action.range is not None:
            low, high = action.range
            spaces["continuous"] = Box(
                low=low.ravel(),
                high=high.ravel(),
                dtype=np.float32,
            )

        # Discrete actions
        if action.dim_d > 0 and action.ncats > 0:
            from gymnasium.spaces import Discrete, MultiDiscrete
            if action.dim_d == 1:
                spaces["discrete"] = Discrete(action.ncats)
            else:
                spaces["discrete"] = MultiDiscrete([action.ncats] * action.dim_d)

        # Return single space if only one type, else Dict space
        if len(spaces) == 0:
            # No actions (e.g., passive device)
            return Box(low=0, high=0, shape=(0,), dtype=np.float32)
        elif len(spaces) == 1:
            return list(spaces.values())[0]
        else:
            return SpaceDict(spaces)

    def _build_observation_space(
        self,
        device: Device,
        partial_obs: bool,
    ) -> gym.Space:
        """Build observation space.

        Args:
            device: Device object
            partial_obs: Whether to use partial observability

        Returns:
            Gymnasium observation space
        """
        # For now, return unbounded Box (will be refined based on actual obs)
        # In practice, this should be computed from device state dimensions
        return Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def observe(self, global_state: Dict[str, Any]) -> Observation:
        """Extract observations from global state.

        Args:
            global_state: Environment state containing:
                - bus_vm: Bus voltages (magnitude)
                - bus_va: Bus angles
                - converged: Whether power flow converged
                - dataset: Current load/price/solar/wind values
                - devices: Dict of all devices (for neighbor info)

        Returns:
            Structured observation
        """
        obs = Observation(timestamp=self._timestep)

        # Local device state
        obs.local = {
            "P": self.device.state.P,
            "Q": getattr(self.device.state, "Q", 0.0),
            "on": self.device.state.on,
        }

        # Add device-specific state
        if hasattr(self.device.state, "soc"):
            obs.local["soc"] = self.device.state.soc

        if hasattr(self.device.state, "Pmax"):
            obs.local["Pmax"] = self.device.state.Pmax
            obs.local["Pmin"] = self.device.state.Pmin

        # Global information (if not partial obs)
        if not self.partial_obs:
            # Bus voltage at device location
            if self._bus_id is not None and "bus_vm" in global_state:
                bus_vm = global_state.get("bus_vm", {})
                bus_va = global_state.get("bus_va", {})
                if isinstance(bus_vm, dict):
                    obs.global_info["bus_voltage"] = bus_vm.get(self._bus_id, 1.0)
                    obs.global_info["bus_angle"] = bus_va.get(self._bus_id, 0.0)

            # Dataset values (price, load forecast, etc.)
            if "dataset" in global_state:
                dataset = global_state["dataset"]
                obs.global_info["price"] = dataset.get("price", 0.0)
                obs.global_info["load"] = dataset.get("load", 1.0)

            # Power flow convergence
            obs.global_info["converged"] = global_state.get("converged", True)

        # Messages from other agents
        obs.messages = self.mailbox.copy()

        return obs

    def act(self, observation: Observation) -> Any:
        """Compute action using policy.

        Args:
            observation: Structured observation

        Returns:
            Action in format defined by action_space
        """
        action = self.policy.forward(observation)

        # Update device action (for compatibility with existing Device interface)
        self._set_device_action(action)

        return action

    def _set_device_action(self, action: Any) -> None:
        """Set action on underlying device.

        Args:
            action: Action from policy
        """
        if isinstance(self.action_space, SpaceDict):
            # Mixed continuous/discrete
            if "continuous" in action:
                self.device.action.c = np.array(action["continuous"], dtype=np.float32)
            if "discrete" in action:
                d = action["discrete"]
                self.device.action.d = np.array([d] if np.isscalar(d) else d, dtype=np.int32)
        elif isinstance(self.action_space, Box):
            # Continuous only
            self.device.action.c = np.array(action, dtype=np.float32).ravel()
        else:
            # Discrete only
            d = action
            self.device.action.d = np.array([d] if np.isscalar(d) else d, dtype=np.int32)

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent and underlying device.

        Args:
            seed: Random seed
            **kwargs: Additional reset params (e.g., init_soc for ESS)
        """
        super().reset(seed=seed)

        # Reset device
        rng = np.random.default_rng(seed) if seed is not None else None
        if hasattr(self.device, "reset"):
            self.device.reset(rnd=rng, **kwargs)

        # Reset policy
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_reward(self) -> float:
        """Get reward signal from device cost/safety.

        Returns:
            Negative cost minus safety penalty
        """
        return -self.device.cost - self.device.safety

    def __repr__(self) -> str:
        device_type = self.device.__class__.__name__
        return f"DeviceAgent(id={self.agent_id}, type={device_type})"
