from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np

from powergrid.devices.features.base import FeatureProvider
from powergrid.utils.phase import PhaseModel, PhaseSpec
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array

CtrlMode = Literal["q_set", "pf_set", "volt_var", "off"]


@provider()
@dataclass(slots=True)
class InverterBasedSource(FeatureProvider):
    """
    Inverter-based renewable (PV / wind).
    Active power largely exogenous; reactive controlled.
    Vector ALWAYS starts with [p_MW, q_MVAr].
    """

    # Phase context
    phase_model: PhaseModel = PhaseModel.THREE_PHASE
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)
    alloc_frac_ph: Optional[Array] = None

    # Ratings / limits
    s_rated_MVA: Optional[float] = None
    q_min_MVAr: Optional[float] = None
    q_max_MVAr: Optional[float] = None

    # Active power (availability / set / measured)
    p_avail_MW: Optional[float] = None
    p_set_MW: Optional[float] = None
    p_MW: Optional[float] = None  # measured instantaneous P (>=0)

    # Reactive power (measured)
    q_MVAr: Optional[float] = None  # measured instantaneous Q

    # Reactive control
    ctrl_mode: CtrlMode = "off"
    q_set_MVAr: Optional[float] = None
    pf_target: Optional[float] = None
    pf_leading: bool = True  # True→capacitive (+Q), False→inductive (-Q)

    # Volt-VAR curve
    vv_v1_pu: Optional[float] = None
    vv_v2_pu: Optional[float] = None
    vv_q1_MVAr: Optional[float] = None
    vv_q2_MVAr: Optional[float] = None

    # Export options
    expand_phases: bool = False
    include_derived: bool = True
    all_params: bool = False  # include all params at end of vector

    def __post_init__(self):
        if self.phase_model == PhaseModel.BALANCED_1PH:
            self.phase_spec = None
            self.alloc_frac_ph = None
        elif self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            n = self.phase_spec.nph()
            if n not in (1, 2, 3):
                raise ValueError("PhaseSpec must have 1, 2, or 3 phases.")
            self._ensure_alloc_(n)
        else:
            raise ValueError(f"Unsupported phase model: {self.phase_model}")

        self._infer_ctrl_mode_()   # <-- NEW: infer mode if hints were given
        self._validate_()
        self.clamp_()

    def _infer_ctrl_mode_(self) -> None:
        if self.ctrl_mode != "off":
            return
        if self.q_set_MVAr is not None:
            self.ctrl_mode = "q_set"
            return
        if self.pf_target is not None:
            self.ctrl_mode = "pf_set"
            return
        if all(x is not None for x in
            (self.vv_v1_pu, self.vv_v2_pu, self.vv_q1_MVAr, self.vv_q2_MVAr)):
            self.ctrl_mode = "volt_var"
            return

    def _ensure_alloc_(self, n: int) -> None:
        if self.alloc_frac_ph is None:
            return
        a = np.asarray(self.alloc_frac_ph, np.float32).ravel()
        if a.shape != (n,):
            raise ValueError(f"alloc_frac_ph must have shape ({n},).")
        if np.any(a < 0.0):
            raise ValueError("alloc_frac_ph must be nonnegative.")
        s = float(a.sum())
        if s <= 0.0:
            raise ValueError("alloc_frac_ph must sum to > 0.")
        self.alloc_frac_ph = (a / s).astype(np.float32)

    def _validate_(self) -> None:
        if self.ctrl_mode not in ("q_set", "pf_set", "volt_var", "off"):
            raise ValueError(
                "ctrl_mode must be one of ['q_set','pf_set','volt_var','off']."
            )

        # P availability / setpoint consistency
        if (self.p_avail_MW is not None and
                self.p_set_MW is not None and
                self.p_set_MW > self.p_avail_MW + 1e-6):
            raise ValueError("p_set_MW cannot exceed p_avail_MW.")

        if self.ctrl_mode == "pf_set":
            if (self.p_MW is None and self.p_set_MW is None
                    and self.p_avail_MW is None):
                raise ValueError("pf_set requires some P (inst/set/avail).")
            if self.pf_target is None or not (0.0 < float(self.pf_target) <= 1.0):
                raise ValueError("pf_target in (0,1] is required for pf_set.")

        if self.ctrl_mode == "q_set" and self.q_set_MVAr is None:
            raise ValueError("q_set requires q_set_MVAr.")

        if self.ctrl_mode == "volt_var":
            req = [self.vv_v1_pu, self.vv_v2_pu, self.vv_q1_MVAr, self.vv_q2_MVAr]
            if any(x is None for x in req):
                raise ValueError(
                    "volt_var requires vv_v1_pu,vv_v2_pu, vv_q1_MVAr,vv_q2_MVAr."
                )
            if not (float(self.vv_v1_pu) < float(self.vv_v2_pu)):  # type: ignore
                raise ValueError("vv_v1_pu must be < vv_v2_pu.")

        if (self.q_min_MVAr is not None and self.q_max_MVAr is not None
                and self.q_max_MVAr < self.q_min_MVAr):
            raise ValueError("q_max_MVAr must be ≥ q_min_MVAr.")

        if self.s_rated_MVA is not None and self.s_rated_MVA < 0.0:
            raise ValueError("s_rated_MVA must be ≥ 0.")

    def _alloc(self) -> Optional[np.ndarray]:
        if self.phase_model != PhaseModel.THREE_PHASE:
            return None
        n = self.phase_spec.nph()  # type: ignore
        if self.alloc_frac_ph is None:
            return np.ones(n, np.float32) / float(n)
        return np.asarray(self.alloc_frac_ph, np.float32).ravel()

    def _p_active(self) -> Optional[float]:
        if self.p_MW is not None:
            return float(self.p_MW)
        if self.p_set_MW is not None:
            return float(self.p_set_MW)
        return None if self.p_avail_MW is None else float(self.p_avail_MW)

    def _q_from_pf(self, p_MW: float) -> float:
        pf = float(self.pf_target)  # type: ignore
        if pf <= 0.0:
            return 0.0
        q = p_MW * float(np.sqrt(max(1.0 / (pf * pf) - 1.0, 0.0)))
        return float(+q if self.pf_leading else -q)

    def _q_from_vv(self, V_pu: float) -> float:
        v1 = float(self.vv_v1_pu)     # type: ignore
        v2 = float(self.vv_v2_pu)     # type: ignore
        q1 = float(self.vv_q1_MVAr)   # type: ignore
        q2 = float(self.vv_q2_MVAr)   # type: ignore
        if V_pu <= v1:
            return q1
        if V_pu >= v2:
            return q2
        t = (V_pu - v1) / (v2 - v1)
        return (1.0 - t) * q1 + t * q2

    def _clip_q(self, q: float) -> (float, float):
        lo = -np.inf if self.q_min_MVAr is None else float(self.q_min_MVAr)
        hi = +np.inf if self.q_max_MVAr is None else float(self.q_max_MVAr)
        q0 = q
        q = float(np.clip(q, lo, hi))
        return q, float(q != q0)

    def vector(self, V_pu_for_vv: Optional[float] = None) -> np.ndarray:
        parts: List[np.ndarray] = []

        # 1) instantaneous P (with fallback) and Q (measured or derived)
        p_val = self._p_active()
        q_val: Optional[float]
        if self.q_MVAr is not None:
            q_val = float(self.q_MVAr)
            was_clip = 0.0
        else:
            # derive Q from control mode if not measured
            if self.ctrl_mode == "off":
                q_val = 0.0
            elif self.ctrl_mode == "q_set":
                q_val = float(self.q_set_MVAr or 0.0)
            elif self.ctrl_mode == "pf_set":
                q_val = 0.0 if p_val is None else self._q_from_pf(p_val)
            else:  # volt_var
                V = 1.0 if V_pu_for_vv is None else float(V_pu_for_vv)
                q_val = self._q_from_vv(V)
            q_val, was_clip = self._clip_q(q_val)

        # ALWAYS emit p_MW and q_MVAr (with fallbacks)
        parts.append(np.array([0.0 if p_val is None else p_val], np.float32))
        parts.append(np.array([q_val], np.float32))

        # command Q (for observability)
        if self.ctrl_mode in ("q_set", "pf_set", "volt_var"):
            # recompute command from mode (not the measured one)
            if self.ctrl_mode == "q_set":
                q_cmd = float(self.q_set_MVAr or 0.0)
            elif self.ctrl_mode == "pf_set":
                q_cmd = 0.0 if p_val is None else self._q_from_pf(p_val)
            else:
                V = 1.0 if V_pu_for_vv is None else float(V_pu_for_vv)
                q_cmd = self._q_from_vv(V)
            q_cmd, _ = self._clip_q(q_cmd)
            parts.append(np.array([q_cmd], np.float32))

        # derived: curtailed P and clipping flag
        if self.include_derived:
            if self.p_avail_MW is not None:
                p_set = (self.p_set_MW
                         if self.p_set_MW is not None else self.p_avail_MW)
                curtailed = float(max(0.0, float(self.p_avail_MW) - float(p_set)))
                parts.append(np.array([curtailed], np.float32))
            parts.append(np.array([was_clip], np.float32))

        # per-phase expansion (optional)
        if self.expand_phases and self.phase_model == PhaseModel.THREE_PHASE:
            alloc = self._alloc()
            parts.append(np.asarray((0.0 if p_val is None else p_val) * alloc,
                                    np.float32))
            parts.append(np.asarray(q_val * alloc, np.float32))

        # all params (optional)
        if self.all_params:
            for attr in [
                "p_set_MW", "q_set_MVAr", "q_min_MVAr", "q_max_MVAr",
                "pf_target", "pf_leading", "ctrl_mode", "expand_phases",
            ]:
                val = getattr(self, attr, None)
                if isinstance(val, (float, int)):
                    parts.append(np.array([float(val)], np.float32))

        return np.concatenate(parts, dtype=np.float32) \
            if parts else np.zeros(0, np.float32)

    def names(self) -> List[str]:
        n: List[str] = []
        # ALWAYS start with these
        n += ["p_MW", "q_MVAr"]

        # command label (if mode uses one)
        if self.ctrl_mode in ("q_set", "pf_set", "volt_var"):
            n.append("q_cmd_MVAr")

        # derived
        if self.include_derived:
            if self.p_avail_MW is not None:
                n.append("p_curtailed_MW")
            n.append("q_clip_flag")

        # per-phase names
        if self.expand_phases and self.phase_model == PhaseModel.THREE_PHASE:
            phases = self.phase_spec.phases  # type: ignore
            n += [f"p_MW_{ph}" for ph in phases]
            n += [f"q_MVAr_{ph}" for ph in phases]

        # all params (optional)
        if self.all_params:
            n.extend([
                "p_set_MW", "q_set_MVAr", "q_min_MVAr", "q_max_MVAr",
                "pf_target", "pf_leading", "ctrl_mode", "expand_phases",
            ])
        return n

    def clamp_(self) -> None:
        for fld in ("s_rated_MVA", "p_avail_MW", "p_set_MW", "p_MW"):
            v = getattr(self, fld)
            if v is not None:
                setattr(self, fld, float(max(0.0, v)))
        if (self.p_avail_MW is not None and self.p_set_MW is not None
                and self.p_set_MW > self.p_avail_MW):
            self.p_set_MW = float(self.p_avail_MW)
        if (self.q_min_MVAr is not None and self.q_max_MVAr is not None
                and self.q_max_MVAr < self.q_min_MVAr):
            self.q_min_MVAr, self.q_max_MVAr = self.q_max_MVAr, self.q_min_MVAr
        if self.pf_target is not None:
            self.pf_target = float(np.clip(self.pf_target, 1e-6, 1.0))

    def to_dict(self) -> Dict:
        d = asdict(self)
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
        if isinstance(d.get("alloc_frac_ph"), np.ndarray):
            d["alloc_frac_ph"] = d["alloc_frac_ph"].astype(np.float32).tolist()
        pm = self.phase_model
        d["phase_model"] = pm.value if isinstance(pm, PhaseModel) else str(pm)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "InverterBasedSource":
        pm = d.get("phase_model", PhaseModel.THREE_PHASE)
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

        alloc = d.get("alloc_frac_ph")
        alloc_arr = None if alloc is None else np.asarray(alloc, np.float32)

        return cls(
            phase_model=pm,
            phase_spec=ps,
            alloc_frac_ph=alloc_arr,
            s_rated_MVA=d.get("s_rated_MVA"),
            q_min_MVAr=d.get("q_min_MVAr"),
            q_max_MVAr=d.get("q_max_MVAr"),
            p_avail_MW=d.get("p_avail_MW"),
            p_set_MW=d.get("p_set_MW"),
            p_MW=d.get("p_MW"),
            q_MVAr=d.get("q_MVAr"),
            ctrl_mode=d.get("ctrl_mode", "off"),
            q_set_MVAr=d.get("q_set_MVAr"),
            pf_target=d.get("pf_target"),
            pf_leading=d.get("pf_leading", True),
            vv_v1_pu=d.get("vv_v1_pu"),
            vv_v2_pu=d.get("vv_v2_pu"),
            vv_q1_MVAr=d.get("vv_q1_MVAr"),
            vv_q2_MVAr=d.get("vv_q2_MVAr"),
            expand_phases=d.get("expand_phases", False),
            include_derived=d.get("include_derived", True),
        )
