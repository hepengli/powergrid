from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from powergrid.core.state import PhaseModel, PhaseSpec
from powergrid.devices.features.base import FeatureProvider
from powergrid.utils.array_utils import _as_f32
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array


@provider()
@dataclass(slots=True)
class ThermalLoading(FeatureProvider):
    """
    Thermal loading as percent of rating.

    Rules:
      • BALANCED_1PH:
          - use scalar 'loading_percentage' (percent)
          - per-phase array is forbidden
      • THREE_PHASE (nph ∈ {1,2,3} from PhaseSpec):
          - use per-phase 'loading_percentage_ph' with shape (nph,)
          - scalar aggregate is forbidden

    vector() returns FRACTIONS (percent/100), not percent.
    names() returns 'loading_frac' (balanced) or per-phase names.
    """
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    # Aggregate (percent) — BALANCED_1PH only
    loading_percentage: Optional[float] = None

    # Per-phase (percent) — THREE_PHASE only; shape (nph,)
    loading_percentage_ph: Optional[Array] = None

    def __post_init__(self):
        if self.phase_model == PhaseModel.BALANCED_1PH:
            self.phase_spec = None
        elif self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            n = self.phase_spec.nph()
            if n not in (1, 2, 3):
                raise ValueError(
                    "THREE_PHASE requires PhaseSpec with 1, 2, or 3 phases "
                    "(e.g., 'A', 'BC', or 'ABC')."
                )
        else:
            raise ValueError(f"Unsupported phase model: {self.phase_model}")

        self._validate_inputs_()
        self._ensure_shapes_()

    def _validate_inputs_(self) -> None:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.loading_percentage_ph is not None:
                raise ValueError(
                    "BALANCED_1PH forbids per-phase field: "
                    "loading_percentage_ph"
                )
            if self.loading_percentage is None:
                raise ValueError(
                    "BALANCED_1PH requires 'loading_percentage' (percent)."
                )
        else:  # THREE_PHASE
            if self.loading_percentage is not None:
                raise ValueError(
                    "THREE_PHASE forbids scalar 'loading_percentage'."
                )
            if self.loading_percentage_ph is None:
                raise ValueError(
                    "THREE_PHASE requires 'loading_percentage_ph' "
                    "with shape (nph,)."
                )

    def _ensure_shapes_(self) -> None:
        if self.phase_model == PhaseModel.THREE_PHASE:
            n = self.phase_spec.nph()
            arr = _as_f32(self.loading_percentage_ph).ravel()
            if arr.shape != (n,):
                raise ValueError(
                    f"loading_percentage_ph must have shape ({n},), "
                    f"got {arr.shape}"
                )
            self.loading_percentage_ph = arr

    def vector(self) -> Array:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            val = float(self.loading_percentage) / 100.0
            return np.array([val], np.float32)

        # THREE_PHASE → per-phase fractions
        arr = _as_f32(self.loading_percentage_ph).ravel()
        return (arr / 100.0).astype(np.float32, copy=False)

    def names(self) -> List[str]:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return ["loading_frac"]
        phases = self.phase_spec.phases  # nph may be 1/2/3
        return [f"loading_frac_{ph}" for ph in phases]

    def clamp_(self) -> None:
        if self.loading_percentage is not None:
            self.loading_percentage = float(
                np.clip(self.loading_percentage, 0.0, 200.0)
            )
        if self.loading_percentage_ph is not None:
            arr = _as_f32(self.loading_percentage_ph).ravel()
            arr = np.clip(arr, 0.0, 200.0).astype(np.float32)
            self.loading_percentage_ph = arr

    def to_dict(self) -> Dict:
        d = asdict(self)

        # numpy → list for JSON
        v = d.get("loading_percentage_ph")
        if isinstance(v, np.ndarray):
            d["loading_percentage_ph"] = v.astype(np.float32).tolist()

        # normalize phase spec
        ps = d.pop("phase_spec", None)
        if ps is None:
            d["phase_spec"] = None
        elif isinstance(ps, dict):
            d["phase_spec"] = {
                "phases": ps.get("phases", "ABC"),
                "has_neutral": ps.get("has_neutral", False),
                "earth_bond": ps.get("earth_bond", True),
            }
        else:
            d["phase_spec"] = {
                "phases": ps.phases,
                "has_neutral": ps.has_neutral,
                "earth_bond": ps.earth_bond,
            }

        pm = self.phase_model
        d["phase_model"] = pm.value if isinstance(pm, PhaseModel) else str(pm)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ThermalLoading":
        pm = d.get("phase_model", PhaseModel.BALANCED_1PH)
        pm = pm if isinstance(pm, PhaseModel) else PhaseModel(pm)

        psd = d.get("phase_spec", None)
        if psd is None:
            ps = None
        elif isinstance(psd, PhaseSpec):
            ps = psd
        else:
            ps = PhaseSpec(
                psd.get("phases", "ABC"),
                psd.get("has_neutral", False),
                psd.get("earth_bond", True),
            )

        def arr(key: str):
            v = d.get(key)
            return None if v is None else _as_f32(v)

        return cls(
            phase_model=pm,
            phase_spec=ps,
            loading_percentage=d.get("loading_percentage"),
            loading_percentage_ph=arr("loading_percentage_ph"),
        )
