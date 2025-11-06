from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from powergrid.devices.features.base import FeatureProvider
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array


@provider()
@dataclass(slots=True)
class StatusBlock(FeatureProvider):
    """
    Generic device status with rich categorical mode and timing.

    Fields
    ------
    # Back-compat booleans (optional)
    online:  device considered available / running
    blocked: device unable to act due to constraint (permits "online & blocked")

    # Categorical state (string token)
    state:              e.g. 'off','startup','online','shutdown','fault',
                        'idle','charging','waiting','driving', ...
    states_vocab:       ordered vocabulary enforcing/encoding `state`
                        (if provided, state must be in vocab)

    # Optional timing/progress
    t_in_state_s:       seconds spent in current state (>=0)
    t_to_next_s:        ETA seconds to next state / completion (>=0)
    progress_frac:      [0..1], percent completion of current transition
                        (e.g., startup ramp or charging progress)

    # Vector export controls
    emit_state_one_hot: export one-hot over states_vocab (if state present)
    emit_state_index:   export scalar index of state in vocab (if state present)
                        (off by default; one-hot is usually safer)
    """
    # booleans (back-compat)
    online: Optional[bool] = None
    blocked: Optional[bool] = None

    # categorical state token
    state: Optional[str] = None
    states_vocab: Optional[List[str]] = None

    # timing / progress
    t_in_state_s: Optional[float] = None
    t_to_next_s: Optional[float] = None
    progress_frac: Optional[float] = None  # [0..1]

    # export controls
    emit_state_one_hot: bool = True
    emit_state_index: bool = False

    def __post_init__(self):
        self._validate_()
        self.clamp_()

    def _validate_(self) -> None:
        # states_vocab sanity
        if self.states_vocab is not None:
            if not isinstance(self.states_vocab, list) or not self.states_vocab:
                raise ValueError("states_vocab must be a non-empty list of strings.")
            if len(set(self.states_vocab)) != len(self.states_vocab):
                raise ValueError("states_vocab contains duplicates.")
            if not all(isinstance(s, str) and s for s in self.states_vocab):
                raise ValueError("states_vocab must contain non-empty strings.")
        # state ∈ vocab if both given
        if self.state is not None and self.states_vocab is not None:
            if self.state not in self.states_vocab:
                raise ValueError(
                    f"state '{self.state}' not in states_vocab {self.states_vocab}."
                )
        # times must be non-negative if provided
        for v, nm in ((self.t_in_state_s, "t_in_state_s"),
                      (self.t_to_next_s, "t_to_next_s")):
            if v is not None and float(v) < 0.0:
                raise ValueError(f"{nm} must be >= 0.")
        # progress range
        if self.progress_frac is not None:
            p = float(self.progress_frac)
            if not (0.0 <= p <= 1.0):
                raise ValueError("progress_frac must be in [0,1].")
        # export: at least one of one-hot or index if state present
        if self.state is not None and self.states_vocab is not None:
            if not (self.emit_state_one_hot or self.emit_state_index):
                # Allow, but warn by raising? Prefer explicit.
                # We'll allow silently and emit nothing for state.
                pass

    def _state_index(self) -> Optional[int]:
        if self.state is None or self.states_vocab is None:
            return None
        return int(self.states_vocab.index(self.state))

    def _one_hot(self, idx: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        if 0 <= idx < n:
            out[idx] = 1.0
        return out

    def vector(self) -> Array:
        parts: List[np.ndarray] = []

        # booleans
        for b in (self.online, self.blocked):
            if b is not None:
                parts.append(np.array([1.0 if b else 0.0], np.float32))

        # categorical state
        idx = self._state_index()
        if idx is not None:
            n = len(self.states_vocab)  # type: ignore
            if self.emit_state_one_hot:
                parts.append(self._one_hot(idx, n))
            if self.emit_state_index:
                parts.append(np.array([float(idx)], np.float32))

        # timing / progress
        if self.t_in_state_s is not None:
            parts.append(np.array([float(self.t_in_state_s)], np.float32))
        if self.t_to_next_s is not None:
            parts.append(np.array([float(self.t_to_next_s)], np.float32))
        if self.progress_frac is not None:
            parts.append(np.array([float(self.progress_frac)], np.float32))

        return (np.concatenate(parts, dtype=np.float32)
                if parts else np.zeros(0, np.float32))

    def names(self) -> List[str]:
        n: List[str] = []
        if self.online is not None:
            n.append("online")
        if self.blocked is not None:
            n.append("blocked")

        idx = self._state_index()
        if idx is not None:
            if self.emit_state_one_hot:
                # Use provided vocab order for stable names
                n += [f"state_{tok}" for tok in self.states_vocab]  # type: ignore
            if self.emit_state_index:
                n.append("state_idx")

        if self.t_in_state_s is not None:
            n.append("t_in_state_s")
        if self.t_to_next_s is not None:
            n.append("t_to_next_s")
        if self.progress_frac is not None:
            n.append("progress_frac")

        return n

    def clamp_(self) -> None:
        # clamp times to non-negative
        if self.t_in_state_s is not None:
            self.t_in_state_s = float(max(0.0, self.t_in_state_s))
        if self.t_to_next_s is not None:
            self.t_to_next_s = float(max(0.0, self.t_to_next_s))
        # clamp progress to [0,1]
        if self.progress_frac is not None:
            self.progress_frac = float(np.clip(self.progress_frac, 0.0, 1.0))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "StatusBlock":
        fields = (
            "online", "blocked", "state", "states_vocab",
            "t_in_state_s", "t_to_next_s", "progress_frac",
            "emit_state_one_hot", "emit_state_index",
        )
        return cls(**{k: d.get(k) for k in fields if k in d})
