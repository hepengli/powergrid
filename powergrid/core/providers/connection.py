from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.core.typing import Array, FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec, CollapsePolicy
from powergrid.core.utils import _as_f32, _one_hot, _pos_seq_voltage_mag_angle, _circ_mean
from powergrid.core.registry import provider

_CONN_SET = {"A","B","C","AB","BC","CA","ABC"}


@provider()
@dataclass(slots=True)
class PhaseConnection(FeatureProvider):
    """
    Encodes how a device's AC port is wired into a 3ϕ network.
    Exports a per-phase boolean mask (1.0 connected, 0.0 not) aligned to PhaseSpec.phases.
    """
    phase_model: PhaseModel = PhaseModel.THREE_PHASE
    phase_spec: PhaseSpec = field(default_factory=lambda: PhaseSpec("ABC"))

    connection: Optional[str] = None  # e.g., "A", "AB", "ABC"

    def _mask(self) -> Array:
        if self.connection is None:
            return np.zeros(self.phase_spec.nph(), np.float32)
        conn = self.connection.upper()
        if conn not in _CONN_SET:
            raise ValueError(
                f"Unknown connection '{self.connection}'. Expected one of {_CONN_SET}."
            )
        phs = self.phase_spec.phases
        m = np.zeros(len(phs), np.float32)
        for i, p in enumerate(phs):
            if p in conn:  # simple char membership
                m[i] = 1.0
        return m

    def vector(self) -> Array:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            # single-phase model: 1 if connected at all, else 0
            return np.array([1.0 if self.connection else 0.0], np.float32)
        return self._mask()

    def names(self) -> List[str]:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return ["conn_present"] if self.connection is not None else []

        if self.connection is not None:
            return [f"conn_{p}" for p in self.phase_spec.phases]
        else:
            return []

    def clamp_(self) -> None:
        pass

    def to_dict(self) -> Dict:
        d = asdict(self)
        ps = d.pop("phase_spec")
        d["phase_spec"] = {
            "phases": ps.phases,
            "has_neutral": ps.has_neutral,
            "earth_bond": ps.earth_bond,
        }
        d["phase_model"] = self.phase_model.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PhaseConnection":
        pm = PhaseModel(d.get("phase_model","three_phase"))
        psd = d.get("phase_spec", {"phases":"ABC","has_neutral":False,"earth_bond":True})
        ps = PhaseSpec(psd["phases"], psd["has_neutral"], psd.get("earth_bond", True))
        return cls(phase_model=pm, phase_spec=ps, connection=d.get("connection"))

    def to_phase_model(
        self, model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V
    ) -> "PhaseConnection":
        # If collapsing to 1ϕ, just preserve the fact it's connected.
        return PhaseConnection(
            phase_model=model, 
            phase_spec=spec, 
            connection=self.connection
        )


@provider()
@dataclass(slots=True)
class PowerAllocation(FeatureProvider):
    """
    Per-phase weights to distribute an aggregate P/Q into phases.
    Adds an optional mask (connection) and enforces it internally.
    """
    phase_model: PhaseModel = PhaseModel.THREE_PHASE
    phase_spec: PhaseSpec = field(default_factory=lambda: PhaseSpec("ABC"))

    weights_ph: Optional[Array] = None     # shape (nph,), non-negative
    mask_ph: Optional[Array] = None        # shape (nph,), 0/1 per-phase connection mask
    enforce_mask: bool = True              # if True, project weights onto mask in clamp_/vector

    def _ensure_shape(self, arr: Optional[Array], name: str) -> Optional[Array]:
        if arr is None: return None
        a = _as_f32(arr).ravel()
        n = self.phase_spec.nph()
        if a.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
        return a

    def set_mask_from_connection(self, connection: Optional[str]) -> "PowerAllocation":
        """Convenience: set mask from a connection string like 'A','AB','ABC'."""
        if connection is None:
            self.mask_ph = None
            return self
        conn = connection.upper()
        m = np.zeros(self.phase_spec.nph(), np.float32)
        for i, ph in enumerate(self.phase_spec.phases):
            if ph in conn:
                m[i] = 1.0
        self.mask_ph = m
        return self

    # Optional: bind from a PhaseConnection provider (if you have it handy)
    def bind_mask_from_phase_connection(self, pc) -> "PowerAllocation":
        return self.set_mask_from_connection(getattr(pc, "connection", None))

    def vector(self) -> Array:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            # Nothing to emit in balanced mode (optional scalar 1.0 if you want)
            return np.zeros(0, np.float32)

        # Validate shapes
        w = self._ensure_shape(self.weights_ph, "weights_ph")
        m = self._ensure_shape(self.mask_ph, "mask_ph")

        if w is None:
            return np.zeros(0, np.float32)

        w = np.maximum(0.0, w)

        if self.enforce_mask and m is not None:
            w = w * m

        s = float(w.sum())
        if s > 0:
            w = w / s
        return w.astype(np.float32, copy=False)

    def names(self) -> List[str]:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return []
        
        name_list = [f"alloc_{p}" for p in self.phase_spec.phases]
        return name_list if self.weights_ph is not None else []

    def clamp_(self) -> None:
        # Validate / sanitize in place
        if self.weights_ph is not None:
            self.weights_ph = self._ensure_shape(self.weights_ph, "weights_ph")
            self.weights_ph = np.maximum(0.0, self.weights_ph).astype(np.float32)

        if self.mask_ph is not None:
            self.mask_ph = self._ensure_shape(self.mask_ph, "mask_ph")
            # clamp mask to {0,1} range (tolerant of small floats)
            self.mask_ph = np.where(self.mask_ph > 0.5, 1.0, 0.0).astype(np.float32)

        if (
            self.enforce_mask 
            and (self.weights_ph is not None) 
            and (self.mask_ph is not None)
        ):
            w = self.weights_ph * self.mask_ph
            s = float(w.sum())
            if s > 0:
                w = w / s
            else:
                # if mask zeroed all weights, default to uniform over connected phases
                cnt = int(self.mask_ph.sum())
                if cnt > 0:
                    w = self.mask_ph / cnt
            self.weights_ph = w.astype(np.float32)

    def to_dict(self) -> Dict:
        d = asdict(self)
        ps = d.pop("phase_spec")
        d["phase_spec"] = {
            "phases": ps.phases,
            "has_neutral": ps.has_neutral,
            "earth_bond": ps.earth_bond
        }
        d["phase_model"] = self.phase_model.value
        for k in ("weights_ph", "mask_ph"):
            v = d.get(k)
            if isinstance(v, np.ndarray):
                d[k] = v.astype(np.float32).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PowerAllocation":
        pm = PhaseModel(d.get("phase_model", "three_phase"))
        psd = d.get(
            "phase_spec", 
            {"phases":"ABC","has_neutral":False,"earth_bond":True}
        )
        ps = PhaseSpec(
            psd["phases"], 
            psd["has_neutral"], 
            psd.get("earth_bond", True)
        )
        def arr(k):
            v = d.get(k)
            return None if v is None else _as_f32(v)
        return cls(
            phase_model=pm, phase_spec=ps,
            weights_ph=arr("weights_ph"),
            mask_ph=arr("mask_ph"),
            enforce_mask=bool(d.get("enforce_mask", True)),
        )

    def to_phase_model(
            self,
            model: PhaseModel,
            spec: PhaseSpec,
            policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V
    ) -> "PowerAllocation":
        if model == PhaseModel.BALANCED_1PH:
            return PowerAllocation(
                phase_model=model,
                phase_spec=spec,
                weights_ph=None,
                mask_ph=None,
                enforce_mask=self.enforce_mask
            )

        # expand/pad to new spec length with zeros
        n = spec.nph()
        def pad(v):
            if v is None:
                return None

            out = np.zeros(n, np.float32)
            k = min(n, v.size); out[:k] = v[:k]
            return out

        return PowerAllocation(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=spec,
            weights_ph=pad(self.weights_ph),
            mask_ph=pad(self.mask_ph),
            enforce_mask=self.enforce_mask,
        )