from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from powergrid.core.action import Action
from powergrid.core.state import DeviceState


class Device(ABC):
    """Base device with state, action and bookkeeping."""

    def __init__(self) -> None:
        self.state = DeviceState()
        self.action = Action()
        self.action_callback: bool = False  # True if external logic sets state
        self.cost: float = 0.0
        self.safety: float = 0.0
        self.adversarial: bool = False

    @abstractmethod
    def set_action_space(self) -> None:  # define action.dim_c/d, ranges, etc.
        ...

    @abstractmethod
    def update_state(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def update_cost_safety(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        ...

    # Optional hook
    def feasible_action(self) -> None:
        """Clamp/adjust current action so it is feasible for this step."""
        return None