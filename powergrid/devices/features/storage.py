from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from powergrid.core.state import PhaseModel, PhaseSpec
from powergrid.devices.features.base import FeatureProvider
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array


@provider()
@dataclass(slots=True)
class StorageBlock(FeatureProvider):
    """
    Device-level storage state (battery/thermal), internally phase-agnostic.

    Connection semantics (THREE_PHASE only):
      - phase_spec defines which phases the device is wired to:
          'A' | 'B' | 'C' | 'AB' | 'BC' | 'AC' | 'ABC'
      - alloc_frac_ph (optional, shape (n_conn,)):
          nonnegative, sums to ~1; how charge/discharge power
          is allocated across connected phases. If None, use
          equal split (1/n_conn each).

    Scalars (state/capabilities):
      - soc, soc_min, soc_max ∈ [0,1]
      - e_capacity_MWh, p_ch_max_MW, p_dis_max_MW ≥ 0
      - eta_ch, eta_dis, soh_frac, reserve_min_frac,
        reserve_max_frac, degradation_frac ∈ [0,1]
      - cycle_throughput_MWh ≥ 0
      - include_derived: append headroom & TTF/TTE in vector()
    """
    # Context
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    # Optional per-connection allocation (THREE_PHASE only)
    alloc_frac_ph: Optional[Array] = None  # shape (n_conn,), sums ~1
    

    # Core SoC & limits
    soc: Optional[float] = None
    soc_min: Optional[float] = None
    soc_max: Optional[float] = None
    e_capacity_MWh: Optional[float] = None
    p_ch_max_MW: Optional[float] = None
    p_dis_max_MW: Optional[float] = None

    # Efficiencies & health
    eta_ch: Optional[float] = None
    eta_dis: Optional[float] = None
    soh_frac: Optional[float] = None

    # Reserves (policy)
    reserve_min_frac: Optional[float] = None
    reserve_max_frac: Optional[float] = None

    # Degradation & metering
    cycle_throughput_MWh: Optional[float] = None
    degradation_frac: Optional[float] = None

    # Vector options
    include_derived: bool = False
    expand_phases: bool = False  # optional: output per-phase power values

    def __post_init__(self):
        # Phase context
        if self.phase_model == PhaseModel.BALANCED_1PH:
            self.phase_spec = None
            if self.alloc_frac_ph is not None:
                raise ValueError(
                    "BALANCED_1PH forbids alloc_frac_ph."
                )
        elif self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            n = self.phase_spec.nph()
            if n not in (1, 2, 3):
                raise ValueError(
                    "THREE_PHASE requires PhaseSpec with 1, 2, or 3 phases."
                )
            self._ensure_alloc_shape_(n)
        else:
            raise ValueError(f"Unsupported phase model: {self.phase_model}")

        self._validate_inputs_()
        self.clamp_()

    def _cap_eff(self) -> Optional[float]:
        if self.e_capacity_MWh is None:
            return None
        cap = float(self.e_capacity_MWh)
        if self.soh_frac is not None:
            cap *= float(np.clip(self.soh_frac, 0.0, 1.0))
        return cap

    def _soc_bounds(self) -> Tuple[float, float]:
        lo = 0.0 if self.soc_min is None else float(np.clip(self.soc_min, 0.0, 1.0))
        hi = 1.0 if self.soc_max is None else float(np.clip(self.soc_max, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _ensure_alloc_shape_(self, n_conn: int) -> None:
        if self.alloc_frac_ph is None:
            return
        a = np.asarray(self.alloc_frac_ph, dtype=np.float32).ravel()
        if a.shape != (n_conn,):
            raise ValueError(
                f"alloc_frac_ph must have shape ({n_conn},), got {a.shape}."
            )
        if np.any(a < 0.0):
            raise ValueError("alloc_frac_ph must be nonnegative.")
        s = float(np.sum(a))
        if s <= 0.0:
            raise ValueError("alloc_frac_ph must sum to a positive value.")
        # Normalize to sum exactly 1.0 (tolerate rounding)
        self.alloc_frac_ph = (a / s).astype(np.float32)

    # Expose connection info to controllers/schedulers
    def connected_phases(self) -> Optional[str]:
        return None if self.phase_spec is None else self.phase_spec.phases

    def get_phase_allocation(self) -> Optional[np.ndarray]:
        """
        Allocation across connected phases, aligned to phase_spec.phases.
        Returns None for BALANCED_1PH. If not provided, returns equal split.
        """
        if self.phase_model != PhaseModel.THREE_PHASE:
            return None
        n = self.phase_spec.nph()  # type: ignore
        if self.alloc_frac_ph is None:
            return (np.ones(n, np.float32) / float(n))
        return np.asarray(self.alloc_frac_ph, dtype=np.float32).ravel()

    def _validate_inputs_(self) -> None:
        fields = (
            self.soc, self.soc_min, self.soc_max,
            self.e_capacity_MWh, self.p_ch_max_MW, self.p_dis_max_MW,
            self.eta_ch, self.eta_dis, self.soh_frac,
            self.reserve_min_frac, self.reserve_max_frac,
            self.cycle_throughput_MWh, self.degradation_frac,
        )
        if all(v is None for v in fields):
            raise ValueError(
                "StorageBlock requires at least one scalar field "
                "(e.g., soc, capacity, or power limits)."
            )
        if (self.reserve_min_frac is not None and
                self.reserve_max_frac is not None and
                self.reserve_max_frac < self.reserve_min_frac):
            raise ValueError("reserve_max_frac must be ≥ reserve_min_frac.")

    def vector(self) -> np.ndarray:
        parts: List[np.ndarray] = []

        def emit(val: Optional[float]) -> None:
            if val is not None:
                parts.append(np.array([float(val)], np.float32))

        for v in (
            self.soc, self.soc_min, self.soc_max,
            self.p_ch_max_MW, self.p_dis_max_MW, self.e_capacity_MWh,
            self.cycle_throughput_MWh, self.degradation_frac,
            self.eta_ch, self.eta_dis, self.soh_frac,
            self.reserve_min_frac, self.reserve_max_frac,
        ):
            emit(v)

        if self.include_derived:
            cap = self._cap_eff()
            lo, hi = self._soc_bounds()
            if (self.soc is not None) and (cap is not None):
                soc = float(np.clip(self.soc, 0.0, 1.0))
                E = soc * cap
                Emin = lo * cap
                Emax = hi * cap
                head_up = max(0.0, Emax - E)
                head_down = max(0.0, E - Emin)
                parts.append(np.array([head_up], np.float32))
                parts.append(np.array([head_down], np.float32))

                # time-to-full: p_ch_max * eta_ch
                if self.p_ch_max_MW and self.p_ch_max_MW > 0.0 and head_up > 0.0:
                    eff = float(self.eta_ch) if self.eta_ch is not None else 1.0
                    denom = float(self.p_ch_max_MW) * max(eff, 1e-6)
                    parts.append(np.array([head_up / denom], np.float32))
                else:
                    parts.append(np.array([0.0], np.float32))

                # time-to-empty: p_dis_max / eta_dis
                if self.p_dis_max_MW and self.p_dis_max_MW > 0.0 and head_down > 0.0:
                    eff = float(self.eta_dis) if self.eta_dis is not None else 1.0
                    denom = float(self.p_dis_max_MW) / max(eff, 1e-6)
                    parts.append(np.array([head_down / denom], np.float32))
                else:
                    parts.append(np.array([0.0], np.float32))

        if self.expand_phases and self.phase_model == PhaseModel.THREE_PHASE:
            alloc = self.get_phase_allocation()
            if self.p_ch_max_MW is not None:
                pch_ph = np.asarray(self.p_ch_max_MW * alloc, np.float32)
                parts.append(pch_ph)
            if self.p_dis_max_MW is not None:
                pdis_ph = np.asarray(self.p_dis_max_MW * alloc, np.float32)
                parts.append(pdis_ph)

        return (np.concatenate(parts, dtype=np.float32)
                if parts else np.zeros(0, np.float32))

    def names(self) -> List[str]:
        n: List[str] = []
        if self.soc is not None:
            n.append("soc_frac")
        if self.soc_min is not None:
            n.append("soc_min_frac")
        if self.soc_max is not None:
            n.append("soc_max_frac")
        if self.p_ch_max_MW is not None:
            n.append("p_ch_max_MW")
        if self.p_dis_max_MW is not None:
            n.append("p_dis_max_MW")
        if self.e_capacity_MWh is not None:
            n.append("e_capacity_MWh")
        if self.cycle_throughput_MWh is not None:
            n.append("cycle_throughput_MWh")
        if self.degradation_frac is not None:
            n.append("degradation_frac")
        if self.eta_ch is not None:
            n.append("eta_ch")
        if self.eta_dis is not None:
            n.append("eta_dis")
        if self.soh_frac is not None:
            n.append("soh_frac")
        if self.reserve_min_frac is not None:
            n.append("reserve_min_frac")
        if self.reserve_max_frac is not None:
            n.append("reserve_max_frac")
        if (self.include_derived and self.soc is not None and
                self._cap_eff() is not None):
            n += ["headroom_up_MWh", "headroom_down_MWh", "ttf_h", "tte_h"]

        # Inside names(), near the end
        if self.expand_phases and self.phase_model == PhaseModel.THREE_PHASE:
            phases = self.phase_spec.phases
            if self.p_ch_max_MW is not None:
                n += [f"p_ch_max_MW_{ph}" for ph in phases]
            if self.p_dis_max_MW is not None:
                n += [f"p_dis_max_MW_{ph}" for ph in phases]

        return n

    def clamp_(self) -> None:
        # fractions in [0, 1]
        for fld in (
            "soc", "soc_min", "soc_max", "eta_ch", "eta_dis",
            "soh_frac", "degradation_frac",
            "reserve_min_frac", "reserve_max_frac",
        ):
            v = getattr(self, fld)
            if v is not None:
                setattr(self, fld, float(np.clip(v, 0.0, 1.0)))

        # NEW: enforce ordering of SOC bounds
        if (self.soc_min is not None and
                self.soc_max is not None and
                self.soc_max < self.soc_min):
            self.soc_min, self.soc_max = self.soc_max, self.soc_min

        # keep SOC within [soc_min, soc_max]
        if self.soc is not None:
            lo = 0.0 if self.soc_min is None else float(self.soc_min)
            hi = 1.0 if self.soc_max is None else float(self.soc_max)
            self.soc = float(np.clip(self.soc, lo, hi))

        # non-negative energies/powers
        for fld in ("p_ch_max_MW", "p_dis_max_MW",
                    "e_capacity_MWh", "cycle_throughput_MWh"):
            v = getattr(self, fld)
            if v is not None:
                setattr(self, fld, float(max(0.0, v)))

        # reserve ordering (existing)
        if (self.reserve_min_frac is not None and
                self.reserve_max_frac is not None and
                self.reserve_max_frac < self.reserve_min_frac):
            self.reserve_min_frac, self.reserve_max_frac = (
                self.reserve_max_frac, self.reserve_min_frac
            )

    def to_dict(self) -> Dict:
        d = asdict(self)
        # alloc array → list
        v = d.get("alloc_frac_ph")
        if isinstance(v, np.ndarray):
            d["alloc_frac_ph"] = v.astype(np.float32).tolist()

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
    def from_dict(cls, d: Dict) -> "StorageBlock":
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

        alloc = d.get("alloc_frac_ph")
        alloc_arr = None if alloc is None else np.asarray(alloc, np.float32)

        return cls(
            phase_model=pm,
            phase_spec=ps,
            alloc_frac_ph=alloc_arr,
            soc=d.get("soc"),
            soc_min=d.get("soc_min"),
            soc_max=d.get("soc_max"),
            e_capacity_MWh=d.get("e_capacity_MWh"),
            p_ch_max_MW=d.get("p_ch_max_MW"),
            p_dis_max_MW=d.get("p_dis_max_MW"),
            eta_ch=d.get("eta_ch"),
            eta_dis=d.get("eta_dis"),
            soh_frac=d.get("soh_frac"),
            reserve_min_frac=d.get("reserve_min_frac"),
            reserve_max_frac=d.get("reserve_max_frac"),
            cycle_throughput_MWh=d.get("cycle_throughput_MWh"),
            degradation_frac=d.get("degradation_frac"),
            include_derived=d.get("include_derived", False),
        )
