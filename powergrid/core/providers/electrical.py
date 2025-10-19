from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
import numpy as np

from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec, CollapsePolicy
from powergrid.core.utils import _as_f32, _one_hot, _pos_seq_voltage_mag_angle, _circ_mean
from powergrid.core.registry import provider


@provider()
@dataclass(slots=True)
class ElectricalBasePh(FeatureProvider):
    """
    Phase-aware electrical fundamentals at a connection point.

    BALANCED_1PH:
      - P_MW, Q_MVAr, V_pu, theta_rad are scalars.

    THREE_PHASE:
      - *_ph arrays shape (nph,) in PhaseSpec order (e.g., A,B,C).
      - If spec.has_neutral: optional neutral telemetry (I_neutral_A, Vn_earth_V).
    """
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Balanced fields
    P_MW: Optional[float] = None
    Q_MVAr: Optional[float] = None
    V_pu: Optional[float] = None
    theta_rad: Optional[float] = None

    # Three-phase fields
    P_MW_ph: Optional[Array] = None
    Q_MVAr_ph: Optional[Array] = None
    V_pu_ph: Optional[Array] = None       # per-phase magnitudes
    theta_rad_ph: Optional[Array] = None  # per-phase angles

    # Neutral telemetry (only meaningful if has_neutral=True)
    I_neutral_A: Optional[float] = None
    Vn_earth_V: Optional[float] = None

    def _ensure_shapes(self) -> None:
        if self.phase_model == PhaseModel.THREE_PHASE:
            n = self.phase_spec.nph()
            for name in ("P_MW_ph", "Q_MVAr_ph", "V_pu_ph", "theta_rad_ph"):
                arr = getattr(self, name)
                if arr is None:
                    setattr(self, name, np.zeros(n, np.float32))
                else:
                    a = _as_f32(arr)
                    if a.shape != (n,):
                        raise ValueError(f"{name} must have shape ({n},)")
                    setattr(self, name, a)

    def vector(self) -> Array:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            parts: List[Array] = []
            for v in (self.P_MW, self.Q_MVAr, self.V_pu, self.theta_rad):
                if v is not None:
                    parts.append(np.array([v], np.float32))
            return np.concatenate(parts, dtype=np.float32) if parts else ZEROS
        else:
            self._ensure_shapes()
            parts: List[Array] = []
            for name in ("P_MW_ph", "Q_MVAr_ph", "V_pu_ph", "theta_rad_ph"):
                arr = getattr(self, name)
                if arr is not None:
                    parts.append(arr.astype(np.float32, copy=False))
            if self.phase_spec.has_neutral:
                if self.I_neutral_A is not None:
                    parts.append(np.array([float(self.I_neutral_A)], np.float32))
                if self.Vn_earth_V is not None:
                    parts.append(np.array([float(self.Vn_earth_V)], np.float32))
            return np.concatenate(parts, dtype=np.float32) if parts else ZEROS

    def names(self) -> List[str]:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            out: List[str] = []
            if self.P_MW is not None:
                out.append("P_MW")
            if self.Q_MVAr is not None:
                out.append("Q_MVAr")
            if self.V_pu is not None:
                out.append("V_pu")
            if self.theta_rad is not None:
                out.append("theta_rad")
            return out
        else:
            phs = list(self.phase_spec.phases)
            nms: List[str] = []
            for base in ("P_MW", "Q_MVAr", "V_pu", "theta_rad"):
                field_name = f"{base}_ph"
                if getattr(self, field_name) is not None:
                    nms.extend([f"{base}_{ph}" for ph in phs])
            if self.phase_spec.has_neutral:
                if self.I_neutral_A is not None:
                    nms.append("I_neutral_A")
                if self.Vn_earth_V is not None:
                    nms.append("Vn_earth_V")
            return nms

    def clamp_(self) -> None:
        if self.I_neutral_A is not None:
            self.I_neutral_A = float(max(0.0, self.I_neutral_A))
        if self.Vn_earth_V is not None:
            self.Vn_earth_V = float(max(0.0, self.Vn_earth_V))

    def to_dict(self) -> Dict:
        d = {
            "phase_model": self.phase_model.value,
            "phase_spec": {
                "phases": self.phase_spec.phases,
                "has_neutral": self.phase_spec.has_neutral,
                "earth_bond": self.phase_spec.earth_bond,
            },
            "P_MW": self.P_MW,
            "Q_MVAr": self.Q_MVAr,
            "V_pu": self.V_pu,
            "theta_rad": self.theta_rad,
            "P_MW_ph": None if self.P_MW_ph is None else self.P_MW_ph.tolist(),
            "Q_MVAr_ph": None if self.Q_MVAr_ph is None else self.Q_MVAr_ph.tolist(),
            "V_pu_ph": None if self.V_pu_ph is None else self.V_pu_ph.tolist(),
            "theta_rad_ph": None if self.theta_rad_ph is None else self.theta_rad_ph.tolist(),
            "I_neutral_A": self.I_neutral_A,
            "Vn_earth_V": self.Vn_earth_V,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ElectricalBasePh":
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

        def arr(key):
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

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "ElectricalBasePh":
        if (
            model == self.phase_model
            and spec.phases == self.phase_spec.phases
            and spec.has_neutral == self.phase_spec.has_neutral
        ):
            return self

        # Collapse 3φ → 1φ
        if model == PhaseModel.BALANCED_1PH:
            self._ensure_shapes()
            P = (
                float(np.sum(self.P_MW_ph)) 
                if self.P_MW_ph is not None 
                else float(self.P_MW or 0.0)
            )
            Q = (
                float(np.sum(self.Q_MVAr_ph)) 
                if self.Q_MVAr_ph is not None 
                else float(self.Q_MVAr or 0.0)
            )

            if (
                policy == CollapsePolicy.SUM_PQ_POSSEQ_V
                and self.V_pu_ph is not None
                and self.theta_rad_ph is not None
                and self.phase_spec.nph() == 3
            ):
                Vmag, ang = _pos_seq_voltage_mag_angle(self.V_pu_ph, self.theta_rad_ph)
            else:
                Vmag = (
                    float(np.mean(self.V_pu_ph))
                    if self.V_pu_ph is not None
                    else (self.V_pu if self.V_pu is not None else 1.0)
                )
                ang = (
                    _circ_mean(self.theta_rad_ph)
                    if self.theta_rad_ph is not None
                    else (self.theta_rad if self.theta_rad is not None else 0.0)
                )

            return ElectricalBasePh(
                phase_model=PhaseModel.BALANCED_1PH,
                phase_spec=spec,
                P_MW=P,
                Q_MVAr=Q,
                V_pu=Vmag,
                theta_rad=ang,
            )

        # Expand 1φ → 3φ
        n = spec.nph()
        return ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=spec,
            P_MW_ph=(np.full(n, float(self.P_MW or 0.0), np.float32)),
            Q_MVAr_ph=(np.full(n, float(self.Q_MVAr or 0.0), np.float32)),
            V_pu_ph=(np.full(n, float(self.V_pu if self.V_pu is not None else 1.0), np.float32)),
            theta_rad_ph=(np.full(n, float(self.theta_rad or 0.0), np.float32)),
            # Omit neutral telemetry unless explicitly provided/computed
            I_neutral_A=None,
            Vn_earth_V=None,
        )

