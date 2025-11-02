import numpy as np

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Tuple, Iterable

class CollapsePolicy(Enum):
    """How to collapse 3φ → 1φ for voltage."""
    SUM_PQ_MEAN_V = "sum_pq_mean_v"       # P,Q sum; |V| mean; θ circular mean
    SUM_PQ_POSSEQ_V = "sum_pq_posseq_v"   # P,Q sum; positive-sequence |V|,∠ (needs 3 phases)


class PhaseModel(Enum):
    BALANCED_1PH = "balanced_1ph"
    THREE_PHASE = "three_phase"


@dataclass(slots=True)
class PhaseSpec:
    phases: str = "ABC"  # e.g. "A", "AB", "ABC" (order matters in names/arrays)
    has_neutral: bool = True
    earth_bond: bool = True

    def __post_init__(self):
        # sanitize phases: uppercase, keep subset of ABC, canonical ABC order
        s = "".join([p for p in self.phases.upper() if p in "ABC"])
        ordered = "".join([p for p in "ABC" if p in s])
        self.phases = ordered or "A"
        # if no neutral, cannot have earth bond
        if not self.has_neutral and self.earth_bond:
            self.earth_bond = False

    def nph(self) -> int:
        return len(self.phases)

    def index(self, ph: str) -> int:
        return self.phases.index(ph)

    # Normalization helper used by DeviceState
    def normalized_for_model(self, model: "PhaseModel") -> "PhaseSpec":
        if model == PhaseModel.BALANCED_1PH:
            # Balanced -> 1φ (pick first available phase)
            return PhaseSpec(
                self.phases[0], 
                has_neutral=self.has_neutral, 
                earth_bond=self.earth_bond
            )

        # THREE_PHASE keeps as-is (canonicalized in __post_init__)
        return PhaseSpec(
            self.phases, 
            has_neutral=self.has_neutral, 
            earth_bond=self.earth_bond
        )

    def to_dict(self) -> Dict:
        return {
            "phases": self.phases,
            "has_neutral": self.has_neutral, 
            "earth_bond": self.earth_bond,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PhaseSpec":
        return cls(
            d.get("phases", "ABC"), 
            d.get("has_neutral", True), 
            d.get("earth_bond", True)
        )

    def index_map_to(self, other: "PhaseSpec") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (src_idx, dst_idx) so that arr_new[dst_idx] = arr_old[src_idx] 
        reorders a per-phase array from self -> other. Phases not present in 
        source or dest are skipped.
        """
        src_pos = {ph: i for i, ph in enumerate(self.phases)}
        pairs = [(src_pos[ph], j) for j, ph in enumerate(other.phases) if ph in src_pos]
        if not pairs:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)

        src_idx, dst_idx = zip(*pairs)
        return np.asarray(src_idx, np.int32), np.asarray(dst_idx, np.int32)

    def align_array(
            self, arr: Iterable[float], 
            other: "PhaseSpec", 
            fill: float = 0.0, 
            dtype=np.float32
    ) -> np.ndarray:
        """
        Align a per-phase array defined on `self` into the order/length of `other`.
        Missing phases are filled with `fill`; extra phases are dropped.
        """
        a = np.asarray(arr, dtype=dtype).ravel()
        if a.size != self.nph():
            raise ValueError(
                f"align_array: expected shape ({self.nph()},), got {a.shape}"
            )
    
        out = np.full(other.nph(), fill, dtype)
        si, di = self.index_map_to(other)
        if si.size:
            out[di] = a[si]
    
        return out

def ensure_phase_context(
    model: PhaseModel,
    spec: PhaseSpec,
    *,
    strict: bool = True
) -> PhaseSpec:
    """
    Return a PhaseSpec consistent with `model`. If `strict` is True,
    raise ValueError instead of normalizing.
    """
    # Balanced must be 1φ
    if model == PhaseModel.BALANCED_1PH:
        if spec.nph() == 1:
            # also sanitize neutral/bond for 1φ
            if not spec.has_neutral and spec.earth_bond:
                return replace(spec, earth_bond=False)
            return spec
        if strict:
            raise ValueError(
                f"BALANCED_1PH requires 1 phase, got '{spec.phases}'"
            )
        # normalize: keep the first listed phase, sanitize earth_bond if no neutral
        first = spec.phases[0]
        eb = spec.earth_bond if spec.has_neutral else False
        return PhaseSpec(first, has_neutral=spec.has_neutral, earth_bond=eb)

    # THREE_PHASE: allow A, AB, ABC (1–3 conductors out of ABC)
    # Just sanitize ordering; leave neutral/bond as provided.
    s = "".join([p for p in "ABC" if p in spec.phases.upper()])
    if not s:
        if strict:
            raise ValueError("THREE_PHASE requires at least one of A/B/C")
        s = "A"
    return PhaseSpec(s, has_neutral=spec.has_neutral, earth_bond=spec.earth_bond)