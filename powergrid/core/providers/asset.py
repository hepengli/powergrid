from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec, CollapsePolicy
from powergrid.core.utils import _as_f32
from powergrid.core.registry import provider


@provider()
@dataclass(slots=True)
class ThermalLoading(FeatureProvider):
    """
    Thermal loading as percent of rating.

    Supports either:
      - aggregate: loading_percentage (scalar, %)
      - per-phase: loading_percentage_ph (shape (nph,), %), aligned to PhaseSpec.phases

    Vector is always FRACTION units (0.0..2.0 typical), i.e., percent / 100.

    Collapse 3φ -> 1φ (when per-phase present): MAX of phases (conservative).
    Expand 1φ -> 3φ: broadcast the single value to all phases.
    """
    # Phase context (only used when per-phase values are provided)
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Aggregate representation (percent)
    loading_percentage: Optional[float] = None

    # Per-phase representation (percent), shape (nph,)
    loading_percentage_ph: Optional[Array] = None

    def _ensure_shapes(self) -> None:
        if self.loading_percentage_ph is None:
            return
        arr = _as_f32(self.loading_percentage_ph).ravel()
        if self.phase_model == PhaseModel.THREE_PHASE:
            n = self.phase_spec.nph()
            if arr.shape != (n,):
                raise ValueError(f"loading_percentage_ph must have shape ({n},), got {arr.shape}")
        self.loading_percentage_ph = arr

    def vector(self) -> Array:
        # Prefer per-phase if present; else aggregate
        if self.loading_percentage_ph is not None:
            self._ensure_shapes()
            return (self.loading_percentage_ph / 100.0).astype(np.float32, copy=False)
        if self.loading_percentage is not None:
            return np.array([float(self.loading_percentage) / 100.0], np.float32)
        return np.zeros(0, np.float32)

    def names(self) -> List[str]:
        if self.loading_percentage_ph is not None:
            if self.phase_model == PhaseModel.BALANCED_1PH:
                return ["loading_frac"]
            return [f"loading_frac_{ph}" for ph in self.phase_spec.phases]
        return ["loading_frac"] if self.loading_percentage is not None else []

    def clamp_(self) -> None:
        # clamp aggregate
        if self.loading_percentage is not None:
            self.loading_percentage = float(np.clip(self.loading_percentage, 0.0, 200.0))
        # clamp per-phase
        if self.loading_percentage_ph is not None:
            self._ensure_shapes()
            clipped_pctg = np.clip(self.loading_percentage_ph, 0.0, 200.0)
            self.loading_percentage_ph = clipped_pctg.astype(np.float32)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Serialize PhaseSpec/PhaseModel explicitly
        ps = d.pop("phase_spec")
        d["phase_spec"] = {
            "phases": ps.phases,
            "has_neutral": ps.has_neutral,
            "earth_bond": ps.earth_bond,
        }
        d["phase_model"] = self.phase_model.value
        # ndarray -> list for JSON
        if isinstance(d.get("loading_percentage_ph"), np.ndarray):
            d["loading_percentage_ph"] = d["loading_percentage_ph"].astype(np.float32).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ThermalLoading":
        pm = PhaseModel(d.get("phase_model", "balanced_1ph"))
        psd = d.get("phase_spec", {"phases": "ABC", "has_neutral": False, "earth_bond": True})
        ps = PhaseSpec(psd["phases"], psd["has_neutral"], psd.get("earth_bond", True))
        l_agg = d.get("loading_percentage")
        l_ph = d.get("loading_percentage_ph")
        arr = None if l_ph is None else _as_f32(l_ph)
        return cls(
            phase_model=pm,
            phase_spec=ps,
            loading_percentage=l_agg,
            loading_percentage_ph=arr,
        )

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "ThermalLoading":
        """
        - If only aggregate is set:
            BALANCED_1PH -> BALANCED_1PH: keep aggregate (phase-agnostic)
            BALANCED_1PH -> THREE_PHASE: broadcast aggregate to per-phase
            THREE_PHASE  -> THREE_PHASE : keep aggregate (still phase-agnostic)
            THREE_PHASE  -> BALANCED_1PH: keep aggregate
        (Aggregate means loading_percentage is used; per-phase is None.)
        - If per-phase is set:
            THREE_PHASE -> BALANCED_1PH: MAX-of-phases (percent)
            BALANCED_1PH -> THREE_PHASE: broadcast the single value
            THREE_PHASE -> THREE_PHASE: pad/truncate to new phases
        """
        # --- aggregate-only path ---
        if self.loading_percentage_ph is None:
            # No per-phase values exist.
            if model == self.phase_model and spec.phases == self.phase_spec.phases:
                return self
            if model == PhaseModel.THREE_PHASE:
                # Broadcast aggregate (if present) to all phases; else remain None.
                if self.loading_percentage is None:
                    return ThermalLoading(
                        phase_model=PhaseModel.THREE_PHASE,
                        phase_spec=spec,
                        loading_percentage=None,
                        loading_percentage_ph=None,
                    )
                n = spec.nph()
                return ThermalLoading(
                    phase_model=PhaseModel.THREE_PHASE,
                    phase_spec=spec,
                    loading_percentage=None,
                    loading_percentage_ph=np.full(n, float(self.loading_percentage), np.float32),
                )
            # Else target is BALANCED_1PH: keep aggregate representation.
            return ThermalLoading(
                phase_model=model,
                phase_spec=spec,
                loading_percentage=self.loading_percentage,
                loading_percentage_ph=None,
            )

        # --- per-phase path ---
        self._ensure_shapes()
        if model == self.phase_model and spec.phases == self.phase_spec.phases:
            return self

        if model == PhaseModel.BALANCED_1PH:
            if self.loading_percentage_ph.size == 0:
                return ThermalLoading(
                    phase_model=PhaseModel.BALANCED_1PH,
                    phase_spec=spec,
                    loading_percentage=None,
                    loading_percentage_ph=None,
                )
            val_pct = float(np.max(self.loading_percentage_ph))  # conservative collapse
            return ThermalLoading(
                phase_model=PhaseModel.BALANCED_1PH,
                phase_spec=spec,
                loading_percentage=val_pct,
                loading_percentage_ph=None,
            )

        # Expand / remap within THREE_PHASE
        if self.phase_model == PhaseModel.BALANCED_1PH:
            # (This branch will rarely trigger because aggregate-only was handled above,
            # but keep for completeness if someone sets both fields inconsistently.)
            if self.loading_percentage is None:
                return ThermalLoading(
                    phase_model=PhaseModel.THREE_PHASE,
                    phase_spec=spec,
                    loading_percentage=None,
                    loading_percentage_ph=None,
                )
            n = spec.nph()
            return ThermalLoading(
                phase_model=PhaseModel.THREE_PHASE,
                phase_spec=spec,
                loading_percentage=None,
                loading_percentage_ph=np.full(n, float(self.loading_percentage), np.float32),
            )

        # Already per-phase but target phases differ: pad/truncate
        n = spec.nph()
        src = self.loading_percentage_ph
        dst = np.zeros(n, np.float32)
        k = min(n, src.size)
        dst[:k] = src[:k]
        return ThermalLoading(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=spec,
            loading_percentage=None,
            loading_percentage_ph=dst,
        )
    