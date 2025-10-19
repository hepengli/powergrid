from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec, CollapsePolicy
from powergrid.core.utils import _as_f32, _one_hot, _pos_seq_voltage_mag_angle, _circ_mean
from powergrid.core.registry import provider


@provider()
@dataclass(slots=True)
class StorageBlock(FeatureProvider):
    """
    Device-level energy storage state (battery or thermal), phase-agnostic.
    Electrical terminal (P/Q/V/theta) lives in ElectricalBasePh.

    Primary (back-compat) fields:
      - soc:               State of charge FRACTION [0..1] (alias soc_frac)
      - soc_min/soc_max:   Allowed SOC bounds [0..1]
      - p_ch_max_MW:       Max charge power (>0) at AC terminal
      - p_dis_max_MW:      Max discharge power (>0) at AC terminal
      - e_capacity_MWh:    Nameplate / usable energy capacity
      - cycle_throughput_MWh: Lifetime cumulative energy throughput (charge+dis)
      - degradation_frac:  0..1 fraction of lost capacity (legacy; prefer soh_frac)

    New (optional) fields:
      - eta_ch, eta_dis:   Charge / discharge efficiencies [0..1]
      - soh_frac:          State of health [0..1], 1.0=new; if set and e_capacity_MWh set,
                           effective capacity = e_capacity_MWh * soh_frac
      - reserve_min_frac, reserve_max_frac: policy reserves (0..1), inclusive
      - include_derived:   if True, vector() appends derived features:
            headroom_up_MWh   = max(0, Emax - E)
            headroom_down_MWh = max(0, E - Emin)
            ttf_h (time to full)   = headroom_up_MWh   / (p_ch_max_MW * eta_ch)    if >0
            tte_h (time to empty)  = headroom_down_MWh / (p_dis_max_MW / eta_dis)  if >0
    """
    # Context (storage is phase-agnostic; keep for consistency)
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Core SoC & limits
    soc: Optional[float] = None           # fraction [0..1]
    soc_min: Optional[float] = None       # fraction [0..1]
    soc_max: Optional[float] = None       # fraction [0..1]
    e_capacity_MWh: Optional[float] = None
    p_ch_max_MW: Optional[float] = None
    p_dis_max_MW: Optional[float] = None

    # Efficiencies & health
    eta_ch: Optional[float] = None        # [0..1]
    eta_dis: Optional[float] = None       # [0..1]
    soh_frac: Optional[float] = None      # [0..1]; effective capacity scaling

    # Reserves (policy)
    reserve_min_frac: Optional[float] = None   # min SOC reserve
    reserve_max_frac: Optional[float] = None   # max SOC reserve

    # Degradation & metering
    cycle_throughput_MWh: Optional[float] = None
    degradation_frac: Optional[float] = None   # legacy, [0..1]

    # Vector options
    include_derived: bool = False

    def _cap_eff(self) -> Optional[float]:
        if self.e_capacity_MWh is None:
            return None
        if self.soh_frac is None:
            return float(self.e_capacity_MWh)
        return float(self.e_capacity_MWh) * float(np.clip(self.soh_frac, 0.0, 1.0))

    def _soc_bounds(self) -> (float, float): # type: ignore
        lo = 0.0 if self.soc_min is None else float(np.clip(self.soc_min, 0.0, 1.0))
        hi = 1.0 if self.soc_max is None else float(np.clip(self.soc_max, 0.0, 1.0))
        if hi < lo:  # auto-correct
            lo, hi = hi, lo
        return lo, hi

    def vector(self) -> Array:
        parts: List[Array] = []

        # Primary scalars (emit if set)
        for v in (self.soc, self.soc_min, self.soc_max,
                  self.p_ch_max_MW, self.p_dis_max_MW, self.e_capacity_MWh,
                  self.cycle_throughput_MWh, self.degradation_frac,
                  self.eta_ch, self.eta_dis, self.soh_frac,
                  self.reserve_min_frac, self.reserve_max_frac):
            if v is not None:
                parts.append(np.array([float(v)], np.float32))

        # Derived metrics (optional)
        if self.include_derived:
            cap = self._cap_eff()
            lo, hi = self._soc_bounds()
            if (self.soc is not None) and (cap is not None):
                soc = float(np.clip(self.soc, 0.0, 1.0))
                E    = soc * cap
                Emin = lo  * cap
                Emax = hi  * cap
                head_up   = max(0.0, Emax - E)
                head_down = max(0.0, E - Emin)
                parts.append(np.array([head_up],   np.float32))   # headroom_up_MWh
                parts.append(np.array([head_down], np.float32))   # headroom_down_MWh

                # time-to-full (charging limited by p_ch_max * eta_ch)
                if self.p_ch_max_MW and self.p_ch_max_MW > 0.0 and head_up > 0.0:
                    eff = float(self.eta_ch) if self.eta_ch is not None else 1.0
                    denom = float(self.p_ch_max_MW) * max(eff, 1e-6)
                    parts.append(np.array([head_up / denom], np.float32))
                else:
                    parts.append(np.array([0.0], np.float32))

                # time-to-empty (discharging limited by p_dis_max / eta_dis)
                if self.p_dis_max_MW and self.p_dis_max_MW > 0.0 and head_down > 0.0:
                    eff = float(self.eta_dis) if self.eta_dis is not None else 1.0
                    denom = float(self.p_dis_max_MW) / max(eff, 1e-6)
                    parts.append(np.array([head_down / denom], np.float32))
                else:
                    parts.append(np.array([0.0], np.float32))

        return np.concatenate(parts, dtype=np.float32) if parts else np.zeros(0, np.float32)

    def names(self) -> List[str]:
        n: List[str] = []
        if self.soc               is not None: n.append("soc_frac")
        if self.soc_min           is not None: n.append("soc_min_frac")
        if self.soc_max           is not None: n.append("soc_max_frac")
        if self.p_ch_max_MW       is not None: n.append("p_ch_max_MW")
        if self.p_dis_max_MW      is not None: n.append("p_dis_max_MW")
        if self.e_capacity_MWh    is not None: n.append("e_capacity_MWh")
        if self.cycle_throughput_MWh is not None: n.append("cycle_throughput_MWh")
        if self.degradation_frac  is not None: n.append("degradation_frac")
        if self.eta_ch            is not None: n.append("eta_ch")
        if self.eta_dis           is not None: n.append("eta_dis")
        if self.soh_frac          is not None: n.append("soh_frac")
        if self.reserve_min_frac  is not None: n.append("reserve_min_frac")
        if self.reserve_max_frac  is not None: n.append("reserve_max_frac")
        if self.include_derived:
            n += ["headroom_up_MWh", "headroom_down_MWh", "ttf_h", "tte_h"]
        return n

    def clamp_(self) -> None:
        # fractions
        for fld in ("soc", "soc_min", "soc_max", "eta_ch", "eta_dis",
                    "soh_frac", "degradation_frac", "reserve_min_frac", "reserve_max_frac"):
            v = getattr(self, fld)
            if v is not None:
                setattr(self, fld, float(np.clip(v, 0.0, 1.0)))
        # ensure SOC within [soc_min, soc_max]
        if self.soc is not None:
            lo, hi = self._soc_bounds()
            self.soc = float(np.clip(self.soc, lo, hi))
        # non-negative powers/energies
        for fld in ("p_ch_max_MW", "p_dis_max_MW", "e_capacity_MWh", "cycle_throughput_MWh"):
            v = getattr(self, fld)
            if v is not None:
                setattr(self, fld, float(max(0.0, v)))

        # fix reserve ordering if both set
        if self.reserve_min_frac is not None and self.reserve_max_frac is not None:
            if self.reserve_max_frac < self.reserve_min_frac:
                self.reserve_min_frac = self.reserve_max_frac
                self.reserve_max_frac = self.reserve_min_frac

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "StorageBlock":
        fields = (
            "phase_model","phase_spec",
            "soc","soc_min","soc_max","e_capacity_MWh",
            "p_ch_max_MW","p_dis_max_MW",
            "eta_ch","eta_dis","soh_frac",
            "reserve_min_frac","reserve_max_frac",
            "cycle_throughput_MWh","degradation_frac",
            "include_derived",
        )
        # PhaseSpec comes as nested dict when serialized from DeviceState; allow both ways
        ps = d.get("phase_spec")
        if isinstance(ps, dict):
            spec = PhaseSpec(
                ps.get("phases","ABC"), 
                ps.get("has_neutral",False), 
                ps.get("earth_bond",True)
            )
            d = dict(d); d["phase_spec"] = spec
        pm = d.get("phase_model")
        if isinstance(pm, str):
            d = dict(d); d["phase_model"] = PhaseModel(pm)
        return cls(**{k: d.get(k) for k in fields if k in d})

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "StorageBlock":
        # Storage internals are not phase-dependent; just update context
        return StorageBlock(
            phase_model=model,
            phase_spec=spec,
            soc=self.soc,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            e_capacity_MWh=self.e_capacity_MWh,
            p_ch_max_MW=self.p_ch_max_MW,
            p_dis_max_MW=self.p_dis_max_MW,
            eta_ch=self.eta_ch,
            eta_dis=self.eta_dis,
            soh_frac=self.soh_frac,
            reserve_min_frac=self.reserve_min_frac,
            reserve_max_frac=self.reserve_max_frac,
            cycle_throughput_MWh=self.cycle_throughput_MWh,
            degradation_frac=self.degradation_frac,
            include_derived=self.include_derived,
        )
