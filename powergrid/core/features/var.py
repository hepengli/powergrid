from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.utils.registry import provider
from powergrid.utils.array_utils import _as_f32, _cat_f32
from powergrid.utils.typing import Array
from powergrid.core.features.base import FeatureProvider

from powergrid.utils.phase import PhaseModel, PhaseSpec


@provider()
@dataclass(slots=True)
class ShuntCapacitorBlock(FeatureProvider):
    """
    Shunt capacitor bank (fixed or staged).

    Rules:
      • BALANCED_1PH
          - phase_spec ignored (set to None)
          - forbid per-phase arrays
          - require kvar_total OR stage_kvar_total
      • THREE_PHASE
          - require 3-phase spec
          - forbid aggregate staged arrays
          - require kvar_ph (len=3) or kvar_total
          - if staged: stage_kvar_ph must be (3, n_stages)
    """
    phase_model: PhaseModel = PhaseModel.THREE_PHASE
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)

    strict_checks: bool = True   # kept for compat; no-op

    # Aggregate (1φ only)
    kvar_total: Optional[float] = None
    n_stages: Optional[int] = None
    stage_enabled: Optional[Array] = None
    stage_kvar_total: Optional[Array] = None

    # Per-phase (3φ only)
    kvar_ph: Optional[Array] = None
    stage_enabled_ph: Optional[Array] = None
    stage_kvar_ph: Optional[Array] = None

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
            bad = [
                k for k, v in {
                    "kvar_ph": self.kvar_ph,
                    "stage_enabled_ph": self.stage_enabled_ph,
                    "stage_kvar_ph": self.stage_kvar_ph,
                }.items() if v is not None
            ]
            if bad:
                raise ValueError(
                    "BALANCED_1PH forbids per-phase fields: " + ", ".join(bad)
                )

        elif self.phase_model == PhaseModel.THREE_PHASE:
            bad = [
                k for k, v in {
                    "stage_enabled": self.stage_enabled,
                    "stage_kvar_total": self.stage_kvar_total,
                }.items() if v is not None
            ]
            if bad:
                raise ValueError(
                    "THREE_PHASE forbids aggregate staged fields: "
                    + ", ".join(bad)
                )

        self._require_minimum_fields_()

        # --- cross-field consistency ---
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.kvar_total is not None and self.kvar_ph is not None:
                total_ph = float(np.sum(_as_f32(self.kvar_ph)))
                if not np.isclose(float(self.kvar_total), total_ph, rtol=1e-3):
                    raise ValueError(
                        f"Conflict: kvar_total={self.kvar_total} "
                        f"≠ sum(kvar_ph)={total_ph}"
                    )
            # if both scalar and staged exist, they should match too
            if (self.kvar_total is not None and
                    self.stage_kvar_ph is not None):
                staged_sum = float(np.sum(_as_f32(self.stage_kvar_ph)))
                if not np.isclose(float(self.kvar_total), staged_sum, rtol=1e-3):
                    raise ValueError(
                        f"Conflict: kvar_total={self.kvar_total} "
                        f"≠ sum(stage_kvar_ph)={staged_sum}"
                    )
        else:
            if (self.kvar_total is not None and
                    self.stage_kvar_total is not None):
                total_staged = float(np.sum(_as_f32(self.stage_kvar_total)))
                if not np.isclose(float(self.kvar_total),
                                total_staged, rtol=1e-3):
                    raise ValueError(
                        f"Conflict: kvar_total={self.kvar_total} "
                        f"≠ sum(stage_kvar_total)={total_staged}"
                    )

    def _require_minimum_fields_(self) -> None:
        """
        Enforce that users supplied enough (and correctly sized) values to
        represent a shunt capacitor for the selected phase model.

        BALANCED_1PH:
        - Need 'kvar_total' OR staged 'stage_kvar_total'.
        - If staged: require positive n_stages and matching lengths.

        THREE_PHASE (nph in {1,2,3}):
        - Need one of: 'kvar_ph' (len nph), 'kvar_total',
            or staged 'stage_kvar_ph' (nph, m).
        - If staged: require positive n_stages and shape checks.
        - If staged and 'kvar_ph' missing, infer it by summing rows.
        """
        if self.phase_model == PhaseModel.BALANCED_1PH:
            has_scalar = self.kvar_total is not None
            has_staged = (
                self.stage_kvar_total is not None or (self.n_stages or 0) > 0
            )

            if not (has_scalar or has_staged):
                raise ValueError(
                    "BALANCED_1PH requires 'kvar_total' or staged "
                    "'stage_kvar_total'."
                )

            if has_staged:
                m = int(self.n_stages or 0)
                if m <= 0 and self.stage_kvar_total is not None:
                    m = int(_as_f32(self.stage_kvar_total).ravel().size)
                if m <= 0:
                    raise ValueError(
                        "Staged config needs positive 'n_stages' or a "
                        "non-empty 'stage_kvar_total'."
                    )
                if self.stage_kvar_total is None:
                    raise ValueError(
                        "Provide 'stage_kvar_total' of length m for staged "
                        "BALANCED_1PH."
                    )
                sk = _as_f32(self.stage_kvar_total).ravel()
                if sk.size != m:
                    raise ValueError(
                        f"'stage_kvar_total' must have length m={m}, "
                        f"got {sk.size}."
                    )
            return

        # THREE_PHASE (supports nph = 1, 2, or 3)
        n = self.phase_spec.nph()
        has_scalar = self.kvar_total is not None
        has_ph = self.kvar_ph is not None
        has_staged = (
            self.stage_kvar_ph is not None
            or self.stage_enabled_ph is not None
            or (self.n_stages or 0) > 0
        )

        if not (has_scalar or has_ph or has_staged):
            raise ValueError(
                "THREE_PHASE requires 'kvar_ph' (len nph), 'kvar_total', "
                "or staged 'stage_kvar_ph'."
            )

        # Staged requirements (and optional inference)
        if has_staged:
            m = int(self.n_stages or 0)
            SK = None
            if self.stage_kvar_ph is not None:
                SK = _as_f32(self.stage_kvar_ph)
                if SK.ndim != 2 or SK.shape[0] != n:
                    raise ValueError(
                        f"'stage_kvar_ph' must have shape ({n}, m), "
                        f"got {SK.shape}."
                    )
                if m <= 0:
                    m = int(SK.shape[1])
            if m <= 0:
                raise ValueError(
                    "Per-phase staged config needs positive 'n_stages' or a "
                    f"({n}, m) 'stage_kvar_ph'."
                )
            if SK is None:
                raise ValueError(
                    "Provide 'stage_kvar_ph' with shape (nph, m) for staged "
                    "THREE_PHASE config."
                )
            if SK.shape[1] != m:
                raise ValueError(
                    f"'stage_kvar_ph' must have shape ({n}, {m}), "
                    f"got {SK.shape}."
                )

            if self.stage_enabled_ph is not None:
                SE = _as_f32(self.stage_enabled_ph)
                if SE.ndim != 2 or SE.shape != (n, m):
                    raise ValueError(
                        f"'stage_enabled_ph' must have shape ({n}, {m}), "
                        f"got {SE.shape}."
                    )

            # Infer per-phase totals if they were not provided
            if self.kvar_ph is None:
                self.kvar_ph = SK.sum(axis=1)

            # Persist inferred stage count
            if self.n_stages is None:
                self.n_stages = int(m)

        # Validate direct per-phase totals if provided
        if self.kvar_ph is not None:
            a = _as_f32(self.kvar_ph).ravel()
            if a.shape != (n,):
                raise ValueError(
                    f"'kvar_ph' must have shape ({n},), got {a.shape}."
                )

    def _infer_stage_count_(self) -> Optional[int]:
        if self.n_stages is not None:
            return int(max(0, int(self.n_stages)))
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.stage_kvar_total is not None:
                return int(_as_f32(self.stage_kvar_total).ravel().size)
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.stage_kvar_ph is not None:
                X = _as_f32(self.stage_kvar_ph)
                return int(X.shape[1]) if X.ndim == 2 else None
        return None

    def _ensure_shapes_(self) -> None:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.stage_kvar_total is not None:
                self.stage_kvar_total = _as_f32(self.stage_kvar_total).ravel()
            if self.stage_enabled is not None:
                self.stage_enabled = _as_f32(self.stage_enabled).ravel()
    
        if self.phase_model == PhaseModel.THREE_PHASE:
            n = self.phase_spec.nph()

            if self.kvar_ph is not None:
                a = _as_f32(self.kvar_ph).ravel()
                if a.shape != (n,):
                    raise ValueError(f"'kvar_ph' must have shape ({n},), got {a.shape}")
                self.kvar_ph = a

            def fix_2d(name: str, A: Optional[Array], m_eff: Optional[int]) -> Optional[Array]:
                if A is None:
                    return None
                X = _as_f32(A)
                if X.ndim != 2:
                    raise ValueError(f"{name} must be 2D (nph, m), got {X.shape}")
                if X.shape[0] != n:
                    raise ValueError(f"{name} first dim must be {n}, got {X.shape}")
                if m_eff is not None and X.shape[1] != m_eff:
                    out = np.zeros((n, m_eff), np.float32)
                    k = min(m_eff, X.shape[1])
                    if k:
                        out[:, :k] = X[:, :k]
                    X = out
                return X

            m_eff = self._infer_stage_count_()
            self.stage_enabled_ph = fix_2d("stage_enabled_ph", self.stage_enabled_ph, m_eff)
            self.stage_kvar_ph    = fix_2d("stage_kvar_ph",    self.stage_kvar_ph,    m_eff)

        m = self._infer_stage_count_()
        if m is not None and self.n_stages is None:
            self.n_stages = int(m)

    def vector(self) -> Array:
        self._ensure_shapes_()
        parts: List[Array] = []

        # Aggregate scalar (allowed in both)
        if self.kvar_total is not None:
            parts.append(np.array([float(self.kvar_total)], np.float32))

        # Aggregate staged (balanced 1φ only)
        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.stage_kvar_total is not None:
                parts.append(_as_f32(self.stage_kvar_total).ravel())
            if self.stage_enabled is not None:
                parts.append(_as_f32(self.stage_enabled).ravel())

        # Per-phase (3φ only)
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.kvar_ph is not None:
                parts.append(_as_f32(self.kvar_ph).ravel())
            if self.stage_kvar_ph is not None:
                parts.append(_as_f32(self.stage_kvar_ph).ravel())
            if self.stage_enabled_ph is not None:
                parts.append(_as_f32(self.stage_enabled_ph).ravel())

        return _cat_f32(parts)

    def names(self) -> List[str]:
        n: List[str] = []

        if self.kvar_total is not None:
            n.append("cap_kvar_total")

        if self.phase_model == PhaseModel.BALANCED_1PH:
            if self.stage_kvar_total is not None:
                m = len(_as_f32(self.stage_kvar_total))
                n += [f"cap_stage_kvar_{i}" for i in range(m)]
            if self.stage_enabled is not None:
                m = len(_as_f32(self.stage_enabled))
                n += [f"cap_stage_en_{i}" for i in range(m)]

        elif self.phase_model == PhaseModel.THREE_PHASE:
            if self.kvar_ph is not None:
                n += [f"cap_kvar_{ph}" for ph in self.phase_spec.phases]
            if self.stage_kvar_ph is not None:
                m = self.stage_kvar_ph.shape[1]
                for ph in self.phase_spec.phases:
                    n += [f"cap_stage_kvar_{ph}_{i}" for i in range(m)]
            if self.stage_enabled_ph is not None:
                m = self.stage_enabled_ph.shape[1]
                for ph in self.phase_spec.phases:
                    n += [f"cap_stage_en_{ph}_{i}" for i in range(m)]

        return n

    def to_dict(self) -> Dict:
        d = asdict(self)

        # numpy → list for JSON
        for k in (
            "stage_enabled",
            "stage_kvar_total",
            "kvar_ph",
            "stage_enabled_ph",
            "stage_kvar_ph",
        ):
            v = d.get(k)
            if isinstance(v, np.ndarray):
                d[k] = v.astype(np.float32).tolist()

        ps = d.pop("phase_spec", None)
        if ps is None:
            d["phase_spec"] = None
        elif isinstance(ps, dict):
            # asdict() already flattened it
            d["phase_spec"] = {
                "phases": ps.get("phases", "ABC"),
                "has_neutral": ps.get("has_neutral", True),
                "earth_bond": ps.get("earth_bond", True),
            }
        else:
            # handle raw PhaseSpec just in case
            d["phase_spec"] = {
                "phases": ps.phases,
                "has_neutral": ps.has_neutral,
                "earth_bond": ps.earth_bond,
            }

        pm = self.phase_model
        d["phase_model"] = pm.value if isinstance(pm, PhaseModel) else str(pm)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ShuntCapacitorBlock":
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
                psd.get("has_neutral", True),
                psd.get("earth_bond", True),
            )

        def arr(k: str) -> Optional[Array]:
            v = d.get(k)
            return None if v is None else _as_f32(v)

        return cls(
            phase_model=pm,
            phase_spec=ps,
            strict_checks=d.get("strict_checks", True),
            kvar_total=d.get("kvar_total"),
            n_stages=d.get("n_stages"),
            stage_enabled=arr("stage_enabled"),
            stage_kvar_total=arr("stage_kvar_total"),
            kvar_ph=arr("kvar_ph"),
            stage_enabled_ph=arr("stage_enabled_ph"),
            stage_kvar_ph=arr("stage_kvar_ph"),
        )



@provider()
@dataclass(slots=True)
class VoltVarCurve(FeatureProvider):
    # Context
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Shared curve (phase-agnostic)
    enabled: Optional[bool] = None
    v_points_pu: Optional[Array] = None     # (k,)
    q_points_pu: Optional[Array] = None     # (k,)

    # Per-phase optional
    enabled_ph: Optional[Array] = None      # (nph,)
    v_points_pu_ph: Optional[Array] = None  # (nph, k)
    q_points_pu_ph: Optional[Array] = None  # (nph, k)

    def __post_init__(self):
        if self.phase_model == PhaseModel.BALANCED_1PH and self.phase_spec.nph() != 1:
            ps = self.phase_spec
            self.phase_spec = PhaseSpec(ps.phases[0], ps.has_neutral, ps.earth_bond)

    def _ensure_shared_(self) -> None:
        if self.v_points_pu is None and self.q_points_pu is None:
            return
        v = None if self.v_points_pu is None else _as_f32(self.v_points_pu).ravel()
        q = None if self.q_points_pu is None else _as_f32(self.q_points_pu).ravel()
        if v is not None and q is not None and v.size != q.size:
            k = min(v.size, q.size)
            v, q = v[:k].astype(np.float32), q[:k].astype(np.float32)
        if v is not None and q is not None:
            order = np.argsort(v)
            if not np.all(order == np.arange(v.size)):
                v, q = v[order], q[order]
        if q is not None:
            q = np.clip(q, -1.0, 1.0)
        self.v_points_pu, self.q_points_pu = v, q

    def _ensure_per_phase_(self) -> None:
        n = self.phase_spec.nph()
        if self.enabled_ph is not None:
            e = _as_f32(self.enabled_ph).ravel()
            if e.shape != (n,):
                raise ValueError(f"enabled_ph must have shape ({n},), got {e.shape}")
            self.enabled_ph = np.where(e >= 0.5, 1.0, 0.0).astype(np.float32)

        def fix_2d(a: Optional[Array]) -> Optional[Array]:
            if a is None: return None
            A = _as_f32(a)
            if A.ndim != 2 or A.shape[0] != n:
                raise ValueError(f"expected 2D array with first dim {n}, got {A.shape}")
            return A

        V = fix_2d(self.v_points_pu_ph)
        Q = fix_2d(self.q_points_pu_ph)
        if V is not None and Q is not None:
            if V.shape != Q.shape:
                k = min(V.shape[1], Q.shape[1])
                V, Q = V[:, :k], np.clip(Q[:, :k], -1.0, 1.0)
            # sort each phase row by V
            V_sorted = np.empty_like(V)
            Q_sorted = np.empty_like(Q)
            for i in range(n):
                order = np.argsort(V[i])
                V_sorted[i] = V[i, order]
                Q_sorted[i] = Q[i, order]
            V, Q = V_sorted, Q_sorted
        elif Q is not None:
            Q = np.clip(Q, -1.0, 1.0)

        self.v_points_pu_ph, self.q_points_pu_ph = V, Q

    def vector(self) -> Array:
        self._ensure_shared_()
        self._ensure_per_phase_()
        parts: List[Array] = []

        # Shared first (balanced or three-phase)
        if self.enabled is not None:
            parts.append(np.array([1.0 if self.enabled else 0.0], np.float32))
        if self.v_points_pu is not None:
            parts.append(self.v_points_pu.ravel().astype(np.float32, copy=False))
        if self.q_points_pu is not None:
            parts.append(self.q_points_pu.ravel().astype(np.float32, copy=False))

        # Per-phase only when in THREE_PHASE and fields are provided
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.enabled_ph is not None:
                parts.append(self.enabled_ph.astype(np.float32, copy=False).ravel())
            if self.v_points_pu_ph is not None:
                parts.append(self.v_points_pu_ph.astype(np.float32, copy=False).ravel())
            if self.q_points_pu_ph is not None:
                parts.append(self.q_points_pu_ph.astype(np.float32, copy=False).ravel())

        return _cat_f32(parts)

    def names(self) -> List[str]:
        n: List[str] = []
        # Shared names
        if self.enabled is not None:
            n.append("vvar_enabled")
        if self.v_points_pu is not None:
            k = int(self.v_points_pu.size)
            n += [f"vvar_v_{i}" for i in range(k)]
        if self.q_points_pu is not None:
            k = int(self.q_points_pu.size)
            n += [f"vvar_q_{i}" for i in range(k)]

        # Per-phase names (only in three-phase export)
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.enabled_ph is not None:
                n += [f"vvar_enabled_{ph}" for ph in self.phase_spec.phases]
            if self.v_points_pu_ph is not None:
                k = int(self.v_points_pu_ph.shape[1])
                for ph in self.phase_spec.phases:
                    n += [f"vvar_v_{ph}_{i}" for i in range(k)]
            if self.q_points_pu_ph is not None:
                k = int(self.q_points_pu_ph.shape[1])
                for ph in self.phase_spec.phases:
                    n += [f"vvar_q_{ph}_{i}" for i in range(k)]
        return n

    def clamp_(self) -> None:
        self._ensure_shared_()
        self._ensure_per_phase_()

    def to_dict(self) -> Dict:
        d = asdict(self)
        # arrays -> lists
        for k in (
            "v_points_pu",
            "q_points_pu",
            "enabled_ph",
            "v_points_pu_ph",
            "q_points_pu_ph"
        ):
            if isinstance(d.get(k), np.ndarray):
                d[k] = d[k].astype(np.float32).tolist()
        ps = d.pop("phase_spec")
        d["phase_spec"] = {
            "phases": ps.phases,
            "has_neutral": ps.has_neutral,
            "earth_bond": ps.earth_bond,
        }
        d["phase_model"] = self.phase_model.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "VoltVarCurve":
        pm = d.get("phase_model", PhaseModel.BALANCED_1PH)
        pm = pm if isinstance(pm, PhaseModel) else PhaseModel(pm)
        psd = d.get("phase_spec", {"phases": "ABC", "has_neutral": True, "earth_bond": True})
        if isinstance(psd, PhaseSpec):
            ps = psd
        else:
            ps = PhaseSpec(
                psd.get("phases","ABC"),
                psd.get("has_neutral",True),
                psd.get("earth_bond",True)
            )

        def arr(k):
            v = d.get(k); return None if v is None else _as_f32(v)

        obj = cls(
            phase_model=pm, phase_spec=ps,
            enabled=d.get("enabled"),
            v_points_pu=arr("v_points_pu"),
            q_points_pu=arr("q_points_pu"),
            enabled_ph=arr("enabled_ph"),
            v_points_pu_ph=arr("v_points_pu_ph"),
            q_points_pu_ph=arr("q_points_pu_ph"),
        )
        obj.clamp_()
        return obj

    def to_phase_model(self, model: PhaseModel, spec: PhaseSpec) -> "VoltVarCurve":
        # Shared fields pass through
        if model == self.phase_model and spec.phases == self.phase_spec.phases:
            return self

        if model == PhaseModel.BALANCED_1PH:
            # Do not attempt to collapse per-phase curves into one; keep only shared fields.
            return VoltVarCurve(
                phase_model=PhaseModel.BALANCED_1PH,
                phase_spec=PhaseSpec(spec.phases[0], spec.has_neutral, spec.earth_bond),
                enabled=self.enabled,
                v_points_pu=self.v_points_pu,
                q_points_pu=self.q_points_pu,
                enabled_ph=None,
                v_points_pu_ph=None,
                q_points_pu_ph=None,
            )

        # THREE_PHASE target: remap per-phase matrices to new spec
        n = spec.nph()
        def remap1d(a: Optional[Array]) -> Optional[Array]:
            if a is None:
                return None

            out = np.zeros(n, np.float32)
            k = min(n, a.size); out[:k] = a[:k]
            return out

        def remap2d(A: Optional[Array]) -> Optional[Array]:
            if A is None: return None
            A = _as_f32(A); k = A.shape[1] if A.ndim == 2 else 0
            out = np.zeros((n, k), np.float32)
            r = min(n, A.shape[0]); out[:r, :k] = A[:r, :k]
            return out

        return VoltVarCurve(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=spec,
            enabled=self.enabled,
            v_points_pu=self.v_points_pu,
            q_points_pu=self.q_points_pu,
            enabled_ph=remap1d(self.enabled_ph),
            v_points_pu_ph=remap2d(self.v_points_pu_ph),
            q_points_pu_ph=remap2d(self.q_points_pu_ph),
        )


@provider()
@dataclass(slots=True)
class VoltVarCurve(FeatureProvider):
    """
    Volt/VAR droop, exported as features.

    Shared (phase-agnostic) curve:
      - enabled: Optional[bool]
      - v_points_pu: (k,)
      - q_points_pu: (k,)  (clamped to [-1, 1])
      If both V and Q are present with different lengths, we truncate to the
      common k and sort by V ascending.

    Per-phase (three-phase contexts):
      - enabled_ph:     (nph,)      in {0,1}
      - v_points_pu_ph: (nph, k)    each row a curve for a phase
      - q_points_pu_ph: (nph, k)    clamped to [-1, 1], row-wise sorted by V

    Emits only fields that are present (not None).
    Does NOT fabricate a balanced curve from per-phase curves when collapsing.
    """
    # Context
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)

    # Shared curve
    enabled: Optional[bool] = None
    v_points_pu: Optional[Array] = None
    q_points_pu: Optional[Array] = None

    # Per-phase curve (THREE_PHASE only)
    enabled_ph: Optional[Array] = None        # (nph,)
    v_points_pu_ph: Optional[Array] = None    # (nph, k)
    q_points_pu_ph: Optional[Array] = None    # (nph, k)

    def __post_init__(self):
        # Balanced must be single-phase; keep first listed phase.
        if (self.phase_model == PhaseModel.BALANCED_1PH
                and self.phase_spec.nph() != 1):
            ps = self.phase_spec
            self.phase_spec = PhaseSpec(
                ps.phases[0], ps.has_neutral, ps.earth_bond
            )

    def _ensure_shared_(self) -> None:
        """Normalize shared V/Q: truncate to common k, sort by V, clamp Q."""
        if self.v_points_pu is None and self.q_points_pu is None:
            return

        v = (None if self.v_points_pu is None
              else _as_f32(self.v_points_pu).ravel())
        q = (None if self.q_points_pu is None
              else _as_f32(self.q_points_pu).ravel())

        # Reconcile lengths: keep common part if both present.
        if v is not None and q is not None and v.size != q.size:
            k = min(v.size, q.size)
            v = v[:k].astype(np.float32)
            q = q[:k].astype(np.float32)

        # Sort by V per-unit if both present.
        if v is not None and q is not None:
            order = np.argsort(v)
            if not np.all(order == np.arange(v.size)):
                v = v[order]
                q = q[order]

        # Clamp Q into [-1, 1] if present.
        if q is not None:
            q = np.clip(q, -1.0, 1.0)

        self.v_points_pu = v
        self.q_points_pu = q

    def _ensure_per_phase_(self) -> None:
        """Normalize per-phase V/Q matrices: shape checks, sort rows, clamp."""
        if self.phase_model != PhaseModel.THREE_PHASE:
            return

        nph = self.phase_spec.nph()

        # enabled_ph
        if self.enabled_ph is not None:
            e = _as_f32(self.enabled_ph).ravel()
            if e.shape != (nph,):
                raise ValueError(
                    f"enabled_ph must have shape ({nph},), got {e.shape}"
                )
            self.enabled_ph = np.where(e >= 0.5, 1.0, 0.0).astype(np.float32)

        # v_points_pu_ph / q_points_pu_ph
        V = None if self.v_points_pu_ph is None else _as_f32(self.v_points_pu_ph)
        Q = None if self.q_points_pu_ph is None else _as_f32(self.q_points_pu_ph)

        def _check_2d(name: str, A: Array) -> None:
            if A.ndim != 2:
                raise ValueError(f"{name} must be 2D (nph, k), got {A.shape}")
            if A.shape[0] != nph:
                raise ValueError(
                    f"{name} first dim must be {nph}, got {A.shape}"
                )

        if V is not None:
            _check_2d("v_points_pu_ph", V)
        if Q is not None:
            _check_2d("q_points_pu_ph", Q)

        # Reconcile (nph, k): if both present, truncate to common k.
        if V is not None and Q is not None and V.shape[1] != Q.shape[1]:
            k = min(V.shape[1], Q.shape[1])
            V = V[:, :k].astype(np.float32)
            Q = Q[:, :k].astype(np.float32)

        # Row-wise sort by V if both present.
        if V is not None and Q is not None:
            V_sorted = np.empty_like(V)
            Q_sorted = np.empty_like(Q)
            for i in range(nph):
                order = np.argsort(V[i])
                if not np.all(order == np.arange(V.shape[1])):
                    V_sorted[i] = V[i, order]
                    Q_sorted[i] = Q[i, order]
                else:
                    V_sorted[i] = V[i]
                    Q_sorted[i] = Q[i]
            V, Q = V_sorted, Q_sorted

        # Clamp Q to [-1, 1]
        if Q is not None:
            Q = np.clip(Q, -1.0, 1.0)

        self.v_points_pu_ph = V
        self.q_points_pu_ph = Q

    def vector(self) -> Array:
        self._ensure_shared_()
        self._ensure_per_phase_()

        parts: List[Array] = []

        # Shared first
        if self.enabled is not None:
            parts.append(
                np.array([1.0 if self.enabled else 0.0], np.float32)
            )
        if self.v_points_pu is not None:
            parts.append(self.v_points_pu.astype(np.float32, copy=False).ravel())
        if self.q_points_pu is not None:
            parts.append(self.q_points_pu.astype(np.float32, copy=False).ravel())

        # Per-phase only for THREE_PHASE
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.enabled_ph is not None:
                parts.append(
                    self.enabled_ph.astype(np.float32, copy=False).ravel()
                )
            if self.v_points_pu_ph is not None:
                parts.append(
                    self.v_points_pu_ph.astype(np.float32, copy=False).ravel()
                )
            if self.q_points_pu_ph is not None:
                parts.append(
                    self.q_points_pu_ph.astype(np.float32, copy=False).ravel()
                )

        return _cat_f32(parts)

    def names(self) -> List[str]:
        n: List[str] = []

        # Shared names
        if self.enabled is not None:
            n.append("vvar_enabled")
        if self.v_points_pu is not None:
            k = int(_as_f32(self.v_points_pu).size)
            n += [f"vvar_v_{i}" for i in range(k)]
        if self.q_points_pu is not None:
            k = int(_as_f32(self.q_points_pu).size)
            n += [f"vvar_q_{i}" for i in range(k)]

        # Per-phase names
        if self.phase_model == PhaseModel.THREE_PHASE:
            if self.enabled_ph is not None:
                n += [f"vvar_enabled_{ph}" for ph in self.phase_spec.phases]
            if self.v_points_pu_ph is not None:
                k = int(self.v_points_pu_ph.shape[1])
                for ph in self.phase_spec.phases:
                    n += [f"vvar_v_{ph}_{i}" for i in range(k)]
            if self.q_points_pu_ph is not None:
                k = int(self.q_points_pu_ph.shape[1])
                for ph in self.phase_spec.phases:
                    n += [f"vvar_q_{ph}_{i}" for i in range(k)]
        return n

    def clamp_(self) -> None:
        # Just re-run normalization; it already clamps Q/binarizes enables.
        self._ensure_shared_()
        self._ensure_per_phase_()

    def to_dict(self) -> Dict:
        d = asdict(self)
        # arrays -> lists
        for k in (
            "v_points_pu",
            "q_points_pu",
            "enabled_ph",
            "v_points_pu_ph",
            "q_points_pu_ph",
        ):
            if isinstance(d.get(k), np.ndarray):
                d[k] = d[k].astype(np.float32).tolist()

        ps = d.pop("phase_spec")
        d["phase_spec"] = {
            "phases": ps.phases,
            "has_neutral": ps.has_neutral,
            "earth_bond": ps.earth_bond,
        }
        d["phase_model"] = self.phase_model.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "VoltVarCurve":
        pm = d.get("phase_model", PhaseModel.BALANCED_1PH)
        pm = pm if isinstance(pm, PhaseModel) else PhaseModel(pm)

        psd = d.get(
            "phase_spec",
            {"phases": "ABC", "has_neutral": True, "earth_bond": True},
        )
        ps = (psd if isinstance(psd, PhaseSpec) else PhaseSpec(
            psd.get("phases", "ABC"),
            psd.get("has_neutral", True),
            psd.get("earth_bond", True),
        ))

        def arr(k: str) -> Optional[Array]:
            v = d.get(k)
            return None if v is None else _as_f32(v)

        obj = cls(
            phase_model=pm,
            phase_spec=ps,
            enabled=d.get("enabled"),
            v_points_pu=arr("v_points_pu"),
            q_points_pu=arr("q_points_pu"),
            enabled_ph=arr("enabled_ph"),
            v_points_pu_ph=arr("v_points_pu_ph"),
            q_points_pu_ph=arr("q_points_pu_ph"),
        )
        # Normalize after load
        obj.clamp_()
        return obj

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
    ) -> "VoltVarCurve":
        # No change
        if model == self.phase_model and spec.phases == self.phase_spec.phases:
            return self

        # Collapse to balanced: keep only shared curve; do NOT try to
        # synthesize a balanced curve from per-phase curves.
        if model == PhaseModel.BALANCED_1PH:
            return VoltVarCurve(
                phase_model=PhaseModel.BALANCED_1PH,
                phase_spec=PhaseSpec(
                    spec.phases[0], spec.has_neutral, spec.earth_bond
                ),
                enabled=self.enabled,
                v_points_pu=self.v_points_pu,
                q_points_pu=self.q_points_pu,
                enabled_ph=None,
                v_points_pu_ph=None,
                q_points_pu_ph=None,
            )

        # Expand / remap to THREE_PHASE: carry shared curve as-is,
        # remap per-phase matrices to new nph via pad/truncate rows.
        nph_new = spec.nph()

        def remap_row(a: Optional[Array]) -> Optional[Array]:
            if a is None:
                return None
            a = _as_f32(a).ravel()
            out = np.zeros(nph_new, np.float32)
            k = min(nph_new, a.size)
            if k:
                out[:k] = a[:k]
            return out

        def remap_mat(A: Optional[Array]) -> Optional[Array]:
            if A is None:
                return None
            X = _as_f32(A)
            if X.ndim != 2:
                return None
            k = X.shape[1]
            out = np.zeros((nph_new, k), np.float32)
            r = min(nph_new, X.shape[0])
            if r and k:
                out[:r, :k] = X[:r, :k]
            return out

        return VoltVarCurve(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=spec,
            enabled=self.enabled,
            v_points_pu=self.v_points_pu,
            q_points_pu=self.q_points_pu,
            enabled_ph=remap_row(self.enabled_ph),
            v_points_pu_ph=remap_mat(self.v_points_pu_ph),
            q_points_pu_ph=remap_mat(self.q_points_pu_ph),
        )
