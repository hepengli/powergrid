from builtins import float
import numpy as np
from typing import Any, Dict, Optional

from powergrid.core.policies import Policy

from ..agents.device_agent import DeviceAgent
from ..core.protocols import NoProtocol, Protocol

class Shunt(DeviceAgent):
    """Switched shunt (capacitor/reactor bank) — **controllable**, not passive.

    Discrete action selects number of steps (0..max_step). Optional switching
    cost applies when the step changes.
    """
    def __init__(
        self, 
        name: str, 
        bus: int,
        q_mvar: float, 
        *, 
        max_step: int = 1, 
        switching_cost: float = 0.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        self.type = "SCB"
        self.name = name
        self.bus = bus
        self.q_mvar = float(q_mvar)
        self.max_step = int(max_step)
        self.switching_cost = float(switching_cost)
        self._last_step = 0
        
        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        # discrete steps: 0..max_step
        self.action.ncats = self.max_step + 1
        self.action.dim_d = 1
        
        self.action.sample()

    def set_device_state(self):
         # state one-hot
        self.state.max_step = self.max_step
        self.state.step = np.zeros(self.max_step + 1, dtype=np.float32)

    def update_state(self) -> None:
        step = int(self.action.d[0]) if self.action.d.size else 0
        self.state.step = np.zeros(self.max_step + 1, dtype=np.float32)
        self.state.step[step] = 1.0
        self._current_step = step  # for cost calculation

    def update_cost_safety(self) -> None:
        changed = int(getattr(self, "_current_step", 0) != getattr(self, "_last_step", 0))
        self.cost = float(self.switching_cost * changed)
        self.safety = 0.0
        self._last_step = getattr(self, "_current_step", self._last_step)

    def reset_device(self, rnd=None) -> None:
        self.state.step = np.zeros(self.max_step + 1, dtype=np.float32)
        self._last_step = 0
        self.cost = 0.0
        self.safety = 0.0

    def __repr__(self) -> str:
        return f"Shunt(name={self.name}, bus={self.bus}, q_mvar={self.q_mvar}, max_step={self.max_step})"