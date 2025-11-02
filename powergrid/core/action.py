from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence, Union, List
import numpy as np

from powergrid.utils.typing import Array, FloatArray, IntArray
from powergrid.utils.array_utils import _cat_f32


@dataclass(slots=True)
class Action:
    """
    - Continuous: `c` in physical units, shape (dim_c,)
    - Multi-discrete: `d`, shape (dim_d,), each d[i] in {0..ncats_i-1}
    - `ncats`: either an int (same categories for all discrete heads)
               or a sequence[int] of length dim_d for per-head categories.
    - `masks`: optional list of boolean arrays, one per head, masks[i].shape==(ncats_i,)
               True=allowed, False=disallowed.

    Use `scale()` / `unscale()` for continuous normalization [-1, 1].
    """

    c: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float32))
    d: IntArray   = field(default_factory=lambda: np.array([], dtype=np.int32))

    dim_c: int = 0
    dim_d: int = 0
    ncats: Union[int, Sequence[int]] = 0

    # (lb, ub) for continuous (each shape (dim_c,))
    range: Optional[Tuple[FloatArray, FloatArray]] = None

    # Optional per-head masks; list length == dim_d, each shape (ncats_i,)
    masks: Optional[List[np.ndarray]] = None

    def _norm_ncats(self) -> np.ndarray:
        """Normalize `ncats` into an int array of length dim_d."""
        if self.dim_d == 0:
            if isinstance(self.ncats, (int, np.integer)) and int(self.ncats) in (0,):
                return np.zeros(0, dtype=np.int32)
            if isinstance(self.ncats, Sequence) and len(self.ncats) == 0:
                return np.zeros(0, dtype=np.int32)
            raise ValueError("When dim_d==0, ncats must be 0 or [].")

        if isinstance(self.ncats, (int, np.integer)):
            K = int(self.ncats)
            if K <= 0:
                raise ValueError("ncats must be >=1 when dim_d>0.")
            return np.full(self.dim_d, K, dtype=np.int32)

        # sequence case
        Ks = np.asarray(self.ncats, dtype=np.int32)
        if Ks.shape != (self.dim_d,):
            raise ValueError("len(ncats) must equal dim_d when given as a sequence.")
        if np.any(Ks <= 0):
            raise ValueError("All ncats[i] must be >=1.")
        return Ks

    def _validate_and_prepare(self) -> None:
        # shape init
        if self.dim_c and self.c.size == 0:
            self.c = np.zeros(self.dim_c, dtype=np.float32)
        if self.dim_d and self.d.size == 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        if self.dim_d == 0:
            self.d = np.array([], dtype=np.int32)

        # range validation
        if self.range is not None:
            lb, ub = self.range
            lb = np.asarray(lb, dtype=np.float32); ub = np.asarray(ub, dtype=np.float32)
            if lb.shape != ub.shape:
                raise ValueError("range must be a tuple of (lb, ub) with identical shapes.")
            if lb.ndim != 1 or (self.dim_c and lb.shape[0] != self.dim_c):
                raise ValueError("range arrays must be 1D with length == dim_c.")
            if not np.all(lb <= ub):
                raise ValueError("range lower bounds must be <= upper bounds.")
            self.range = (lb, ub)

        # ncats normalized + masks validation
        Ks = self._norm_ncats()  # raises if inconsistent
        if self.masks is not None:
            if len(self.masks) != self.dim_d:
                raise ValueError("len(masks) must equal dim_d.")
            for i, (mask, K) in enumerate(zip(self.masks, Ks)):
                m = np.asarray(mask, dtype=bool)
                if m.shape != (K,):
                    raise ValueError(f"masks[{i}] must have shape ({K},).")
                if not np.any(m):
                    raise ValueError(f"masks[{i}] excludes all categories.")

    def __post_init__(self) -> None:
        self._validate_and_prepare()

    def set_specs(
        self,
        dim_c: int = 0,
        dim_d: int = 0,
        ncats: Union[int, Sequence[int]] = 0,
        range: Optional[Tuple[Array, Array]] = None,
        masks: Optional[Sequence[Array]] = None,
    ) -> "Action":
        self.dim_c, self.dim_d = int(dim_c), int(dim_d)
        self.ncats = ncats
        self.c = np.zeros(self.dim_c, dtype=np.float32)
        self.d = np.array([], dtype=np.int32)
        if self.dim_d > 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        self.range = None if range is None else (
            np.asarray(range[0], dtype=np.float32),
            np.asarray(range[1], dtype=np.float32),
        )
        self.masks = None if masks is None else [np.asarray(m, dtype=bool) for m in masks]
        self._validate_and_prepare()
        return self

    def sample(self, rng: object | None = None) -> None:
        """Sample random action (continuous in physical units, discrete honoring masks)."""
        # rng normalize
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        elif not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator, a seed int, or None")

        # continuous
        if self.dim_c:
            if self.range is not None:
                lb, ub = self.range
                self.c = rng.uniform(lb, ub).astype(np.float32)
            else:
                self.c = rng.standard_normal(self.dim_c).astype(np.float32)

        # multi-discrete
        if self.dim_d:
            Ks = self._norm_ncats()
            out = np.empty(self.dim_d, dtype=np.int32)
            if self.masks is None:
                for i, K in enumerate(Ks):
                    out[i] = int(rng.integers(K))
            else:
                for i, (K, mask) in enumerate(zip(Ks, self.masks)):
                    valid = np.flatnonzero(mask)
                    out[i] = int(rng.choice(valid)) if valid.size else 0
            self.d = out

    def clip_(self) -> "Action":
        """Clip `c` to `range` in-place; no-op if no range."""
        if self.range is not None and self.c.size:
            lb, ub = self.range
            np.clip(self.c, lb, ub, out=self.c)
        return self

    def scale(self) -> FloatArray:
        """Return normalized [-1, 1] copy of `c`. Zero-span axes → 0."""
        if self.range is None or self.c.size == 0:
            return self.c.astype(np.float32, copy=True)
        lb, ub = self.range
        span = ub - lb
        x = np.zeros_like(self.c, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            x[mask] = 2.0 * (self.c[mask] - lb[mask]) / span[mask] - 1.0
        return x

    def unscale(self, x: Sequence[float]) -> FloatArray:
        """Set `c` from normalized [-1, 1] vector `x` (physical units). 
           Zero-span axes → lb.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != self.dim_c:
            raise ValueError("normalized vector length must equal dim_c")
        if self.range is None:
            self.c = x.copy()
            return self.c
        lb, ub = self.range
        span = ub - lb
        self.c = np.empty_like(lb, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            self.c[mask] = lb[mask] + 0.5 * (x[mask] + 1.0) * span[mask]
        if np.any(~mask):
            self.c[~mask] = lb[~mask]
        return self.c

    def as_vector(self) -> FloatArray:
        """Flatten to `[c..., d...]` (float32) for logging/export."""
        if self.dim_d:
            parts = [self.c.astype(np.float32), self.d.astype(np.float32)]
            return _cat_f32(parts)
        return self.c.astype(np.float32, copy=True)

    @classmethod
    def from_vector(
        cls,
        vec: Sequence[float],
        dim_c: int,
        dim_d: int,
        ncats: Union[int, Sequence[int]] = 0,
        range: Optional[Tuple[Array, Array]] = None,
        masks: Optional[Sequence[Array]] = None,
    ) -> "Action":
        """Create an Action from a flat vector `[c..., d...]` (d length = dim_d)."""
        vec = np.asarray(vec, dtype=np.float32)
        expected = dim_c + dim_d
        if vec.size != expected:
            raise ValueError(f"vector length {vec.size} != expected {expected}")
        c = vec[:dim_c].astype(np.float32)
        d = vec[dim_c:].astype(np.int32) if dim_d else np.array([], dtype=np.int32)
        return cls(
            c=c, d=d, dim_c=dim_c, dim_d=dim_d, ncats=ncats, range=range,
            masks=None if masks is None else [np.asarray(m, bool) for m in masks]
        )

    def reset(self) -> None:
        if self.dim_c:
            self.c[...] = 0.0
        if self.dim_d:
            self.d[...] = 0
