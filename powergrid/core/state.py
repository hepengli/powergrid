from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Type, Any, Iterable, Tuple
import numpy as np

from powergrid.core.utils.typing import Array, FeatureProvider
from powergrid.core.utils.registry import provider
from powergrid.core.utils.phase import PhaseModel, PhaseSpec


KNOWN_FEATURES: Dict[str, Type[FeatureProvider]] = {}


def _vec_names(feat: FeatureProvider) -> Tuple[np.ndarray, List[str]]:
    v = np.asarray(feat.vector(), np.float32).ravel()
    n = feat.names()
    if len(n) != v.size:
        raise ValueError(
            f"{feat.__class__.__name__}: names ({len(n)}) != vector size ({v.size})."
        )
    return v, n


@provider()
@dataclass(slots=True)
class DeviceState(FeatureProvider):
    """
    DeviceState — phase-owning container that aggregates multiple feature
    providers (StorageBlock, ElectricalBasePh, PhaseConnection, etc.) into a
    single feature vector and name list.

    Authoritative phase context:
      • DeviceState validates its own (phase_model, phase_spec).
      • It then overrides every child feature's phase_model/spec to match.
      • After override, each feature is asked to re-validate under the final context:
          - revalidate_after_context() if available (preferred)
          - else _validate_inputs_() or _ensure_shapes/_ensure_shapes_() if present
          - else defer errors to vector()/names()

    No implicit conversions:
      • We never call to_phase_model() on features.
      • Users must decide BALANCED_1PH vs THREE_PHASE; DeviceState enforces that
        choice across children.

    Vector & names:
      • vectors are concatenated in feature order; empty vectors are skipped.
      • names are concatenated in the same order; 1:1 parity enforced per feature.
      • prefix_names=True prepends '<ClassName>.' to each child’s names.
    """

    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = None  # None allowed for BALANCED_1PH

    features: List[FeatureProvider] = field(default_factory=list)
    prefix_names: bool = False

    def __post_init__(self):
        self._validate_phase_context_()
        self._apply_phase_context_to_features_()

    def _validate_phase_context_(self) -> None:
        if self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            n = self.phase_spec.nph()
            if n not in (1, 2, 3):
                raise ValueError("THREE_PHASE PhaseSpec must have 1, 2, or 3 phases.")
        elif self.phase_model == PhaseModel.BALANCED_1PH:
            # Allow None; if provided, must be single-phase
            if self.phase_spec is not None and self.phase_spec.nph() > 1:
                raise ValueError("BALANCED_1PH cannot use a multi-phase PhaseSpec.")
        else:
            raise ValueError(f"Unsupported phase model: {self.phase_model}")

    def _apply_phase_context_to_features_(self) -> None:
        """
        Override children with the authoritative phase context, then
        invoke their validators in the final context.
        """
        for f in self.features:
            # 1) Override phase_model if attribute exists
            if hasattr(f, "phase_model"):
                try:
                    setattr(f, "phase_model", self.phase_model)
                except Exception:
                    # If a feature refuses reassignment, let it fail later.
                    pass

            # 2) Override phase_spec if attribute exists
            if hasattr(f, "phase_spec"):
                try:
                    if self.phase_model == PhaseModel.BALANCED_1PH:
                        setattr(f, "phase_spec", None)
                    else:
                        setattr(f, "phase_spec", self.phase_spec)
                except Exception:
                    pass

            # 3) Re-validate under final context if the feature provides hooks
            #    Try in this order: revalidate_after_context, _validate_inputs_,
            #    _ensure_shapes, _ensure_shapes_
            for meth in ("revalidate_after_context",
                         "_validate_inputs_",
                         "_ensure_shapes",
                         "_ensure_shapes_"):
                if hasattr(f, meth) and callable(getattr(f, meth)):
                    getattr(f, meth)()
                    break  # run at most one

    def _iter_ready_features(self) -> Iterable[FeatureProvider]:
        for f in self.features:
            yield f
@dataclass(slots=False)  # Disable slots to allow __getattr__ and __setattr__
class DeviceState:
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)
    providers: List[FeatureProvider] = field(default_factory=list)

    def vector(self) -> Array:
        vecs: List[np.ndarray] = []
        for f in self._iter_ready_features():
            v, _ = _vec_names(f)
            if v.size:
                vecs.append(v)
        if not vecs:
            return np.zeros(0, np.float32)
        return np.concatenate(vecs, dtype=np.float32)

    def names(self) -> List[str]:
        out: List[str] = []
        for f in self._iter_ready_features():
            _, n = _vec_names(f)
            if self.prefix_names and n:
                pref = f.__class__.__name__ + "."
                n = [pref + s for s in n]
            out += n
        return out

    def clamp_(self) -> None:
        for f in self._iter_ready_features():
            if hasattr(f, "clamp_"):
                f.clamp_()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_model": (
                self.phase_model.value
                if isinstance(self.phase_model, PhaseModel) else str(self.phase_model)
            ),
            "phase_spec": (
                None if self.phase_spec is None else {
                    "phases": self.phase_spec.phases,
                    "has_neutral": self.phase_spec.has_neutral,
                    "earth_bond": self.phase_spec.earth_bond,
                }
            ),
            "prefix_names": self.prefix_names,
            "features": [
                {
                    "kind": f.__class__.__name__,
                    "payload": (
                        f.to_dict() if hasattr(f, "to_dict") else asdict(f)
                    ),
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        registry: Optional[Dict[str, Type[FeatureProvider]]] = None,
    ) -> "DeviceState":
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

        reg = registry or KNOWN_FEATURES
        feats: List[FeatureProvider] = []
        for item in d.get("features", []):
            kind = item.get("kind")
            payload = item.get("payload", {})
            cls_ = reg.get(kind)
            if cls_ is None:
                raise ValueError(
                    f"Unknown feature kind '{kind}'. Provide a registry mapping."
                )
            if hasattr(cls_, "
                       "):
                feats.append(cls_.from_dict(payload))  # type: ignore
            else:
                feats.append(cls_(**payload))          # type: ignore

        ds = cls(
            phase_model=pm,
            phase_spec=ps,
            features=feats,
            prefix_names=d.get("prefix_names", False),
        )
        # Defensive: apply again post-build
        ds._validate_phase_context_()
        ds._apply_phase_context_to_features_()
        return ds
        spec = spec or self.phase_spec
        new_provs: List[FeatureProvider] = []
        for p in self.providers:
            if hasattr(p, "to_phase_model"):
                new_provs.append(p.to_phase_model(model, spec, policy))
            else:
                new_provs.append(p)
        return DeviceState(phase_model=model, phase_spec=spec, providers=new_provs)


    def as_vector(self) -> np.ndarray:
        """Convert device state to flat numeric vector.

        Uses the provider-based architecture to construct the state vector.

        Returns:
            Float32 numpy array containing the state representation
        """
        return self.vector()

    def __getattr__(self, name: str):
        """Compatibility layer: access provider attributes directly.

        This allows backward compatibility with code that accesses state.P, state.Q, etc.
        """
        # Avoid recursion on internal attributes
        if name in ('phase_model', 'phase_spec', 'providers'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Alias mappings for backward compatibility
        alias_map = {
            'P': 'P_MW',
            'Q': 'Q_MVAr',
        }
        lookup_name = alias_map.get(name, name)

        # Search providers for the attribute
        providers_list = object.__getattribute__(self, 'providers')
        for provider in providers_list:
            if hasattr(provider, lookup_name):
                return getattr(provider, lookup_name)

        # Default values for commonly accessed attributes that may not exist
        defaults = {
            'on': 1.0,  # Devices without UC are always "on"
        }
        if name in defaults:
            return defaults[name]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        """Compatibility layer: set provider attributes directly."""
        # Handle dataclass fields normally
        if name in ('phase_model', 'phase_spec', 'providers'):
            object.__setattr__(self, name, value)
            return

        # Alias mappings for backward compatibility
        alias_map = {
            'P': 'P_MW',
            'Q': 'Q_MVAr',
        }
        lookup_name = alias_map.get(name, name)

        # Try to set on existing providers
        if hasattr(self, 'providers'):
            for provider in self.providers:
                if hasattr(provider, lookup_name):
                    setattr(provider, lookup_name, value)
                    return

        # If no provider has this attribute, set it as a normal attribute
        object.__setattr__(self, name, value)
