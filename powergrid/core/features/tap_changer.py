from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.utils.typing import Array
from powergrid.core.features.base import FeatureProvider

from powergrid.utils.registry import provider
from powergrid.utils.array_utils import _as_f32, _one_hot
from powergrid.utils.phase import PhaseModel, PhaseSpec


@provider()
@dataclass(slots=True)
class TapChangerPh(FeatureProvider):
    """OLTC that can be balanced or per-phase."""
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    # encoding
    one_hot: bool = True

    # tap step range (required)
    tap_min: Optional[int] = None
    tap_max: Optional[int] = None

    # balanced scalar position
    tap_position: Optional[int] = None

    # per-phase positions (shape (nph,), int)
    tap_pos_ph: Optional[Array] = None

    def __post_init__(self):
        # model/spec checks (subset phases allowed)
        if self.phase_model == PhaseModel.BALANCED_1PH:
            self.phase_spec = None
        elif self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            n = self.phase_spec.nph()
            if n not in (1, 2, 3):
                raise ValueError(
                    "THREE_PHASE requires PhaseSpec with 1, 2, or 3 phases."
                )
        else:
            raise ValueError(f"Unsupported phase model: {self.phase_model}")

        self._validate_inputs_()
        self.clamp_()  # keep invariants

    def _nsteps(self) -> int:
        if self.tap_min is None or self.tap_max is None:
            return 0
        return int(self.tap_max - self.tap_min + 1)

    def _validate_inputs_(self) -> None:
        # require a valid tap range
        if self.tap_min is None or self.tap_max is None:
            raise ValueError("Provide 'tap_min' and 'tap_max'.")
        if int(self.tap_max) < int(self.tap_min):
            raise ValueError("'tap_max' must be ≥ 'tap_min'.")
        if self._nsteps() <= 0:
            raise ValueError("Tap range yields zero steps.")

        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.tap_pos_ph is not None:
                raise ValueError("BALANCED_1PH forbids 'tap_pos_ph'.")
            if self.tap_position is None:
                raise ValueError(
                    "BALANCED_1PH requires 'tap_position' (scalar)."
                )
        else:  # THREE_PHASE
            if self.tap_position is not None:
                raise ValueError("THREE_PHASE forbids 'tap_position'.")
            if self.tap_pos_ph is None:
                raise ValueError(
                    "THREE_PHASE requires 'tap_pos_ph' with shape (nph,)."
                )
            n = self.phase_spec.nph()  # type: ignore
            a = _as_f32(self.tap_pos_ph).ravel()
            if a.shape != (n,):
                raise ValueError(
                    f"'tap_pos_ph' must have shape ({n},), got {a.shape}."
                )

    def vector(self) -> np.ndarray:
        nsteps = self._nsteps()
        if nsteps <= 0:
            return np.zeros(0, np.float32)

        if self.phase_model == PhaseModel.BALANCED_1PH:
            pos = int(self.tap_position) - int(self.tap_min)  # type: ignore
            pos = int(np.clip(pos, 0, nsteps - 1))
            if self.one_hot:
                return _one_hot(pos, nsteps)
            frac = pos / max(nsteps - 1, 1)
            return np.array([frac], np.float32)

        # THREE_PHASE
        n = self.phase_spec.nph()  # type: ignore
        outs: List[np.ndarray] = []
        base = int(self.tap_min)  # type: ignore
        for p in np.asarray(self.tap_pos_ph, dtype=np.int32).ravel():
            pos = int(np.clip(int(p) - base, 0, nsteps - 1))
            if self.one_hot:
                outs.append(_one_hot(pos, nsteps))
            else:
                frac = pos / max(nsteps - 1, 1)
                outs.append(np.array([frac], np.float32))
        return np.concatenate(outs, dtype=np.float32) if outs else \
            np.zeros(0, np.float32)

    def names(self) -> List[str]:
        nsteps = self._nsteps()
        if nsteps <= 0:
            return []

        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.one_hot:
                return [f"tap_{k}"
                        for k in range(int(self.tap_min),
                                       int(self.tap_max) + 1)]  # type: ignore
            return ["tap_pos_norm"]

        # THREE_PHASE
        phases = self.phase_spec.phases  # type: ignore
        labels: List[str] = []
        if self.one_hot:
            for ph in phases:
                labels += [f"tap_{ph}_{k}"
                           for k in range(int(self.tap_min),
                                          int(self.tap_max) + 1)]  # type: ignore
        else:
            labels += [f"tap_{ph}_pos_norm" for ph in phases]
        return labels

    def clamp_(self) -> None:
        if self.tap_min is None or self.tap_max is None:
            return
        lo = int(self.tap_min)
        hi = int(self.tap_max)
        if hi < lo:
            lo, hi = hi, lo
            self.tap_min, self.tap_max = lo, hi

        if self.tap_position is not None:
            self.tap_position = int(np.clip(int(self.tap_position), lo, hi))

        if self.tap_pos_ph is not None:
            a = np.asarray(self.tap_pos_ph, np.int32).ravel()
            a = np.clip(a, lo, hi).astype(np.int32)
            self.tap_pos_ph = a

    def to_dict(self) -> Dict:
        d = asdict(self)
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
        # model as value
        pm = self.phase_model
        d["phase_model"] = pm.value if isinstance(pm, PhaseModel) else str(pm)
        # arrays → lists
        v = d.get("tap_pos_ph")
        if isinstance(v, np.ndarray):
            d["tap_pos_ph"] = v.astype(int).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TapChangerPh":
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

        tpph = d.get("tap_pos_ph")
        arr = None if tpph is None else np.asarray(tpph, np.int32)

        return cls(
            phase_model=pm,
            phase_spec=ps,
            one_hot=d.get("one_hot", True),
            tap_min=d.get("tap_min"),
            tap_max=d.get("tap_max"),
            tap_position=d.get("tap_position"),
            tap_pos_ph=arr,
        )
