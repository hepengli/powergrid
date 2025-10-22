"""Action representations for device control.

This module defines action containers for devices supporting both continuous
and discrete control actions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Array = np.ndarray


@dataclass
class Action:
    """Container for device actions.

    Supports both continuous and discrete actions, which can be used
    independently or combined depending on the device type.

    Attributes:
        c: Continuous action vector (float32, possibly empty)
        d: Discrete action vector (int32, length 1 or empty if unused)
        dim_c: Dimension of continuous action space
        dim_d: Dimension of discrete action space
        ncats: Number of categories for discrete actions
        range: Bounds for continuous actions, shape (2, dim_c) as (lower, upper)
    """

    c: Array = field(default_factory=lambda: np.array([], dtype=np.float32))
    d: Array = field(default_factory=lambda: np.array([], dtype=np.int32))
    dim_c: int = 0
    dim_d: int = 0
    ncats: int = 0
    range: Optional[Array] = None  # shape (2, dim_c)

    def sample(self) -> None:
        """Sample random action from the defined action space.

        For continuous actions, samples uniformly from the specified range.
        For discrete actions, samples uniformly from available categories.
        """
        if self.dim_c > 0 and self.range is not None:
            lb, ub = self.range
            self.c = np.random.uniform(lb, ub).astype(np.float32)
        if self.dim_d > 0 and self.ncats > 0:
            self.d = np.array([np.random.randint(self.ncats)], dtype=np.int32)