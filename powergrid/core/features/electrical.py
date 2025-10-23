from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import numpy as np

from powergrid.core.utils.typing import Array, FeatureProvider
from powergrid.core.utils.registry import provider
from powergrid.core.utils.utils import _as_f32, _cat_f32
from powergrid.core.state import PhaseModel, PhaseSpec


@provider()
@dataclass(slots=True)
class ElectricalBasePh(FeatureProvider):
    """
    Phase-aware electrical fundamentals at a connection point.

    BALANCED_1PH:
      - Scalars only: P_MW, Q_MVAr, V_pu, theta_rad.

    THREE_PHASE:
      - Per-phase arrays only: *_ph with shape (3,) in spec order.
      - Neutral telemetry allowed only if spec.has_neutral is True.
    """
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    # Balanced scalars
    P_MW: Optional[float] = None
    Q_MVAr: Optional[float] = None
    V_pu: Optional[float] = None
    theta_rad: Optional[float] = None

    # Three-phase arrays
    P_MW_ph: Optional[Array] = None
    Q_MVAr_ph: Optional[Array] = None
    V_pu_ph: Optional[Array] = None
    theta_rad_ph: Optional[Array] = None

    # Neutral telemetry (needs has_neutral=True)
    I_neutral_A: Optional[float] = None
    Vn_earth_V: Optional[float] = None

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
            # Balanced: ignore spec (already set to None in __post_init__)

            # 1) Forbid any per-phase arrays
            bad = []
            for name in ("P_MW_ph", "Q_MVAr_ph", "V_pu_ph", "theta_rad_ph"):
                if getattr(self, name) is not None:
                    bad.append(name)
            if bad:
                raise ValueError(
                    "Balanced model forbids per-phase fields: " + ", ".join(bad)
                )

            # 2) Neutral telemetry not meaningful in balanced
            if self.I_neutral_A is not None or self.Vn_earth_V is not None:
                raise ValueError("Neutral telemetry not allowed in BALANCED_1PH.")

            # 3) Require at least one scalar present
            if all(
                getattr(self, k) is None
                for k in ("P_MW", "Q_MVAr", "V_pu", "theta_rad")
            ):
                raise ValueError(
                    "BALANCED_1PH requires at least one of "
                    "P_MW, Q_MVAr, V_pu, or theta_rad."
                )
            return

        # THREE_PHASE
        bad = [
            k for k, v in {
                "P_MW": self.P_MW,
                "Q_MVAr": self.Q_MVAr,
                "V_pu": self.V_pu,
                "theta_rad": self.theta_rad,
            }.items() if v is not None
        ]
        if bad:
            raise ValueError("THREE_PHASE forbids scalar fields: " + ", ".join(bad))

        if all(x is None for x in
            (self.P_MW_ph, self.Q_MVAr_ph, self.V_pu_ph, self.theta_rad_ph)):
            raise ValueError(
                "THREE_PHASE requires at least one per-phase array: "
                "P_MW_ph, Q_MVAr_ph, V_pu_ph, or theta_rad_ph."
            )

        # Neutral only if spec.has_neutral
        if self.phase_spec is not None and not self.phase_spec.has_neutral:
            if self.I_neutral_A is not None or self.Vn_earth_V is not None:
                raise ValueError(
                    "Neutral telemetry requires has_neutral=True in PhaseSpec."
                )

    def _ensure_shapes_(self) -> None:
        if self.phase_model == PhaseModel.THREE_PHASE:
            n = self.phase_spec.nph()
            def chk(name: str):
                arr = getattr(self, name)
                if arr is None:
                    return
                a = _as_f32(arr).ravel()
                if a.shape != (n,):
                    raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
                setattr(self, name, a)
            for nm in ("P_MW_ph", "Q_MVAr_ph", "V_pu_ph", "theta_rad_ph"):
                chk(nm)

    def vector(self) -> Array:
        parts: List[Array] = []
        if self.phase_model == PhaseModel.BALANCED_1PH:
            for v in (self.P_MW, self.Q_MVAr, self.V_pu, self.theta_rad):
                if v is not None:
                    parts.append(np.array([float(v)], np.float32))
            return _cat_f32(parts)

        # THREE_PHASE
        def push(a: Optional[Array]) -> None:
            if a is not None:
                parts.append(_as_f32(a).ravel())
        push(self.P_MW_ph)
        push(self.Q_MVAr_ph)
        push(self.V_pu_ph)
        push(self.theta_rad_ph)
        return _cat_f32(parts)

    def names(self) -> List[str]:
        out: List[str] = []
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.P_MW is not None:
                out.append("P_MW")
            if self.Q_MVAr is not None:
                out.append("Q_MVAr")
            if self.V_pu is not None:
                out.append("V_pu")
            if self.theta_rad is not None:
                out.append("theta_rad")
            return out

        # THREE_PHASE
        assert self.phase_spec is not None
        phases = self.phase_spec.phases
        def per(prefix: str) -> List[str]:
            return [f"{prefix}_{ph}" for ph in phases]
        if self.P_MW_ph is not None:
            out += per("P_MW")
        if self.Q_MVAr_ph is not None:
            out += per("Q_MVAr")
        if self.V_pu_ph is not None:
            out += per("V_pu")
        if self.theta_rad_ph is not None:
            out += per("theta_rad")
        return out

    def clamp_(self) -> None:
        if self.I_neutral_A is not None:
            self.I_neutral_A = float(max(0.0, self.I_neutral_A))
        if self.Vn_earth_V is not None:
            self.Vn_earth_V = float(max(0.0, self.Vn_earth_V))

    def to_dict(self) -> Dict:
        d = asdict(self)

        # numpy → list for JSON
        for k in ("P_MW_ph", "Q_MVAr_ph", "V_pu_ph", "theta_rad_ph"):
            v = d.get(k)
            if isinstance(v, np.ndarray):
                d[k] = v.astype(np.float32).tolist()

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
    def from_dict(cls, d: Dict) -> "ElectricalBasePh":
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
            P_MW=d.get("P_MW"),
            Q_MVAr=d.get("Q_MVAr"),
            V_pu=d.get("V_pu"),
            theta_rad=d.get("theta_rad"),
            P_MW_ph=arr("P_MW_ph"),
            Q_MVAr_ph=arr("Q_MVAr_ph"),
            V_pu_ph=arr("V_pu_ph"),
            theta_rad_ph=arr("theta_rad_ph"),
            I_neutral_A=d.get("I_neutral_A"),
            Vn_earth_V=d.get("Vn_earth_V"),
        )