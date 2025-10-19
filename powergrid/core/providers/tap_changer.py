from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec, CollapsePolicy
from powergrid.core.utils import _as_f32, _one_hot, _pos_seq_voltage_mag_angle, _circ_mean
from powergrid.core.registry import provider


@provider()
@dataclass(slots=True)
class TapChangerPh(FeatureProvider):
    """OLTC that can be balanced or have per-phase positions."""
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # balanced representation
    tap_position: Optional[int] = None
    tap_min: Optional[int] = None
    tap_max: Optional[int] = None
    one_hot: bool = True

    # per-phase optional
    tap_pos_ph: Optional[Array] = None  # shape (nph,), int

    def _count(self) -> int:
        if self.tap_min is None or self.tap_max is None:
            return 0
        return int(self.tap_max - self.tap_min + 1)

    def vector(self) -> Array:
        nsteps = self._count()
        if nsteps <= 0:
            return np.zeros(0, np.float32)

        if (
            self.phase_model == PhaseModel.BALANCED_1PH 
            or self.tap_pos_ph is None
        ):
            pos = (
                0 if self.tap_position is None 
                else self.tap_position - (self.tap_min or 0)
            )
            pos = int(np.clip(pos, 0, max(nsteps - 1, 0)))

            if self.one_hot:
                return _one_hot(pos, nsteps)
            else:
                return np.array([pos / max(nsteps - 1, 1)], np.float32)

        outs: List[Array] = []
        for p in _as_f32(self.tap_pos_ph).astype(np.int32):
            pos = int(np.clip(p - (self.tap_min or 0), 0, max(nsteps - 1, 0)))
            if self.one_hot:
                outs.append(_one_hot(pos, nsteps))
            else:
                outs.append(np.array([pos / max(nsteps - 1, 1)], np.float32))
        return np.concatenate(outs, dtype=np.float32)

    def names(self) -> List[str]:
        nsteps = self._count()
        if nsteps <= 0:
            return []
        if self.phase_model == PhaseModel.BALANCED_1PH or self.tap_pos_ph is None:
            if self.one_hot:
                return [f"tap_{k}" for k in range(self.tap_min, self.tap_max + 1)]
            else:
                return ["tap_pos_norm"]
        labels: List[str] = []
        for ph in self.phase_spec.phases:
            if self.one_hot:
                labels += [f"tap_{ph}_{k}" for k in range(self.tap_min, self.tap_max + 1)]
            else:
                labels += [f"tap_{ph}_pos_norm"]
        return labels

    def clamp_(self) -> None:
        if (
            self.tap_pos_ph is not None 
            and self.tap_min is not None 
            and self.tap_max is not None
        ):
            self.tap_pos_ph = np.clip(
                np.asarray(self.tap_pos_ph, np.int32), 
                self.tap_min, 
                self.tap_max
            )

    def to_dict(self) -> Dict:
        return {
            "phase_model": self.phase_model.value,
            "phase_spec": {
                "phases": self.phase_spec.phases,
                "has_neutral": self.phase_spec.has_neutral,
                "earth_bond": self.phase_spec.earth_bond,
            },
            "tap_position": self.tap_position,
            "tap_min": self.tap_min,
            "tap_max": self.tap_max,
            "one_hot": self.one_hot,
            "tap_pos_ph": (
                None 
                if self.tap_pos_ph is None 
                else np.asarray(self.tap_pos_ph, np.int32).tolist()
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TapChangerPh":
        pm = PhaseModel(d.get("phase_model", "balanced_1ph"))
        psd = d.get(
            "phase_spec", 
            {"phases": "ABC", "has_neutral": False, "earth_bond": True}
        )
        ps = PhaseSpec(
            psd.get("phases", "ABC"), 
            psd.get("has_neutral", False), 
            psd.get("earth_bond", True)
        )
        tpph = d.get("tap_pos_ph")
        tpph = None if tpph is None else np.asarray(tpph, np.int32)
        return cls(
            pm, 
            ps, 
            d.get("tap_position"), 
            d.get("tap_min"), 
            d.get("tap_max"), 
            d.get("one_hot", True), 
            tpph,
        )

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "TapChangerPh":
        if model == self.phase_model and spec.phases == self.phase_spec.phases:
            return self
        if model == PhaseModel.BALANCED_1PH:
            if self.tap_pos_ph is not None and self.tap_pos_ph.size:
                val = int(np.median(self.tap_pos_ph))
            else:
                val = self.tap_position
            return TapChangerPh(
                PhaseModel.BALANCED_1PH, 
                spec, 
                tap_position=val, 
                tap_min=self.tap_min, 
                tap_max=self.tap_max, 
                one_hot=self.one_hot,
            )
        n = spec.nph()
        tap_pos_ph = int(self.tap_position if self.tap_position is not None else 0)
        arr = np.full(n, tap_pos_ph, np.int32)
        return TapChangerPh(
            PhaseModel.THREE_PHASE, 
            spec,
            tap_position=None, 
            tap_min=self.tap_min, 
            tap_max=self.tap_max,
            one_hot=self.one_hot, 
            tap_pos_ph=arr,
        )
