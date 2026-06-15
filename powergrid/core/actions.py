from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

Array = np.ndarray

@dataclass
class Action:
    """Container for device actions.

    - Continuous action `c`: float32 vector (possibly empty)
    - Discrete action `d`: **length-1** int32 vector (or empty if unused)
    - `dim_c`/`dim_d`: sizes of c/d components
    - `range`: (lb, ub) arrays for continuous sampling
    """

    c: Array = field(default_factory=lambda: np.array([], dtype=np.float32))
    d: Array = field(default_factory=lambda: np.array([], dtype=np.int32))
    dim_c: int = 0
    dim_d: int = 0
    ncats: int = 0
    range: Optional[Array] = None  # shape (2, dim_c)

    def sample(self) -> None:
        if self.dim_c > 0 and self.range is not None:
            lb, ub = self.range
            self.c = np.random.uniform(lb, ub).astype(np.float32)
        if self.dim_d > 0 and self.ncats > 0:
            self.d = np.array([np.random.randint(self.ncats)], dtype=np.int32)