from dataclasses import dataclass, field
from typing import Optional
import numpy as np

Array = np.ndarray

@dataclass
class DeviceState:
    """State of a device and helper to build a numeric state vector.

    Only attributes that exist contribute to the vector, which allows different
    devices to expose different state elements.
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
        state = np.array([], dtype=np.float32)
        if self.Pmax is not None:
            state = np.append(state, self.P)
        if self.Qmax is not None:
            state = np.append(state, self.Q)
        if self.price is not None:
            state = np.append(state, float(self.price) / 100.)

        if self.shutting is not None:
            on_state = np.zeros(2, dtype=np.float32)
            on_state[1 if self.on else 0] = 1
            state = np.concatenate([state, on_state])
            state = np.append(state, float(self.shutting))

        if self.starting is not None:
            state = np.append(state, float(self.starting))

        if self.soc is not None:
            state = np.append(state, float(self.soc))

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
            state = np.append(state, float(self.loading_percentage) / 100.0)

        return state