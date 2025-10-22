"""State representations for power system devices.

This module defines state containers that support various device types
including generators, storage systems, transformers, and shunts.
"""

from dataclasses import dataclass
from typing import Optional
from builtins import float

import numpy as np

Array = np.ndarray


@dataclass
class DeviceState:
    """State of a device with conversion to numeric vector representation.

    Only attributes that are not None contribute to the vector, which allows
    different device types to expose different state elements.

    Attributes:
        P: Active power (MW)
        Q: Reactive power (MVAr)
        on: On/off status (0 or 1)
        Pmax: Maximum active power limit
        Pmin: Minimum active power limit
        Qmax: Maximum reactive power limit
        Qmin: Minimum reactive power limit
        shutting: Shutting down counter for generators
        starting: Starting up counter for generators
        soc: State of charge (0-100%) for storage
        max_step: Maximum number of steps for shunt devices
        step: One-hot encoded step position for shunt devices
        tap_max: Maximum tap position for transformers
        tap_min: Minimum tap position for transformers
        tap_position: Current tap position for transformers
        loading_percentage: Transformer loading (0-100%)
        price: Grid electricity price
    """

    P: float = 0.0
    Q: float = 0.0
    on: int = 1  # 0/1

    # Optional attributes used by various devices
    Pmax: Optional[float] = None
    Pmin: Optional[float] = None
    Qmax: Optional[float] = None
    Qmin: Optional[float] = None

    shutting: Optional[int] = None
    starting: Optional[int] = None

    soc: Optional[float] = None

    # Shunt
    max_step: Optional[int] = None
    step: Optional[np.ndarray] = None  # one-hot length = max_step+1

    # Transformer
    tap_max: Optional[int] = None
    tap_min: Optional[int] = None
    tap_position: Optional[int] = None

    # Transformer loading
    loading_percentage: Optional[float] = None

    # Grid price
    price: Optional[float] = None

    def as_vector(self) -> np.ndarray:
        """Convert device state to flat numeric vector.

        Only non-None attributes are included in the vector. The order is fixed
        to ensure consistency across calls.

        Returns:
            Float32 numpy array containing the state representation
        """
        state = np.array([], dtype=np.float32)
        if self.Pmax is not None:
            state = np.append(state, np.float32(self.P))
        if self.Qmax is not None:
            state = np.append(state, np.float32(self.Q))
        if self.price is not None:
            state = np.append(state, np.float32(float(self.price) / 100.))

        if self.shutting is not None:
            on_state = np.zeros(2, dtype=np.float32)
            on_state[1 if self.on else 0] = 1
            state = np.concatenate([state, on_state])
            state = np.append(state, np.float32(self.shutting))

        if self.starting is not None:
            state = np.append(state, np.float32(self.starting))

        if self.soc is not None:
            state = np.append(state, np.float32(self.soc))

        if self.max_step is not None:
            step_vec = (
                self.step
                if isinstance(self.step, np.ndarray)
                else np.zeros(self.max_step + 1, dtype=np.float32)
            )
            state = np.append(state, step_vec)

        if self.tap_max is not None and self.tap_min is not None:
            count = self.tap_max - self.tap_min + 1
            one_hot = np.zeros(count, dtype=np.float32)
            pos = (self.tap_position if self.tap_position is not None else self.tap_min) - self.tap_min
            pos = int(np.clip(pos, 0, count - 1))
            one_hot[pos] = 1
            state = np.append(state, one_hot)

        if self.loading_percentage is not None:
            state = np.append(state, np.float32(float(self.loading_percentage) / 100.0))

        return state