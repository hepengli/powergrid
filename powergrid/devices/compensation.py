import numpy as np
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from powergrid.core.policies import Policy
from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec
from powergrid.utils.registry import provider
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol


# Create provider for step-based discrete state
@provider()
@dataclass(slots=True)
class StepState(FeatureProvider):
    """Provider for discrete step state (e.g., shunt capacitor banks)."""
    max_step: int = 0
    step: Optional[Array] = None  # One-hot encoded current step

    def vector(self) -> Array:
        if self.step is not None:
            return self.step.astype(np.float32, copy=False)
        return np.zeros(self.max_step + 1, dtype=np.float32)

    def names(self) -> List[str]:
        return [f"step_{i}" for i in range(self.max_step + 1)]

    def clamp_(self) -> None:
        if self.step is not None:
            # Ensure one-hot encoding
            self.step = self.step.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "max_step": self.max_step,
            "step": None if self.step is None else self.step.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepState":
        step_data = d.get("step")
        return cls(
            max_step=d.get("max_step", 0),
            step=None if step_data is None else np.array(step_data, dtype=np.float32),
        )

    def to_phase_model(self, model: PhaseModel, spec: PhaseSpec, policy=None) -> "StepState":
        return self

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
        # Create step state provider
        step_state = StepState(
            max_step=self.max_step,
            step=np.zeros(self.max_step + 1, dtype=np.float32),
        )
        self.state.providers = [step_state]

    def update_state(self) -> None:
        step_state = self._get_step_state()
        step = int(self.action.d[0]) if self.action.d.size else 0
        step_state.step = np.zeros(self.max_step + 1, dtype=np.float32)
        step_state.step[step] = 1.0
        self._current_step = step  # for cost calculation

    def update_cost_safety(self) -> None:
        changed = int(getattr(self, "_current_step", 0) != getattr(self, "_last_step", 0))
        self.cost = float(self.switching_cost * changed)
        self.safety = 0.0
        self._last_step = getattr(self, "_current_step", self._last_step)

    def reset_device(self, rnd=None) -> None:
        step_state = self._get_step_state()
        step_state.step = np.zeros(self.max_step + 1, dtype=np.float32)
        self._last_step = 0
        self.cost = 0.0
        self.safety = 0.0

    def _get_step_state(self) -> StepState:
        """Get the StepState provider from state."""
        for provider in self.state.providers:
            if isinstance(provider, StepState):
                return provider
        raise ValueError("StepState provider not found in state")

    def __repr__(self) -> str:
        return f"Shunt(name={self.name}, bus={self.bus}, q_mvar={self.q_mvar}, max_step={self.max_step})"