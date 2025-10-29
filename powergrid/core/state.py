from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np

from .typing import Array, FeatureProvider
from .registry import ProviderRegistry
from .utils import _as_f32, _one_hot, _pos_seq_voltage_mag_angle, _circ_mean

class PhaseModel(Enum):
    BALANCED_1PH = "balanced_1ph"
    THREE_PHASE = "three_phase"


class CollapsePolicy(Enum):
    """How to collapse 3φ → 1φ for voltage."""
    SUM_PQ_MEAN_V = "sum_pq_mean_v"       # P,Q sum; |V| mean; θ circular mean
    SUM_PQ_POSSEQ_V = "sum_pq_posseq_v"   # P,Q sum; positive-sequence |V|,∠ (needs 3 phases)


@dataclass(slots=True)
class PhaseSpec:
    phases: str = "ABC"
    has_neutral: bool = False
    earth_bond: bool = True

    def __post_init__(self):
        if not self.has_neutral and self.earth_bond:
            self.earth_bond = False  # or raise ValueError(...)

    def nph(self) -> int:
        return len(self.phases)

    def index(self, ph: str) -> int:
        return self.phases.index(ph)
    

@dataclass(slots=True)
class StatusFlags(FeatureProvider):
    online: Optional[bool] = None
    blocked: Optional[bool] = None

    def vector(self) -> Array:
        parts: List[Array] = []
        for b in (self.online, self.blocked):
            if b is not None:
                parts.append(np.array([1.0 if b else 0.0], np.float32))
        
        return (
            np.concatenate(parts, dtype=np.float32) 
            if parts 
            else np.zeros(0, np.float32)
        )

    def names(self) -> List[str]:
        n: List[str] = []
        if self.online is not None:
            n.append("online")
        if self.blocked is not None:
            n.append("blocked")
        return n

    def clamp_(self) -> None:
        pass

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "StatusFlags":
        return cls(**{k: d.get(k) for k in ("online", "blocked")})

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: PhaseSpec,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "StatusFlags":
        return self


@dataclass(slots=False)  # Disable slots to allow __getattr__ and __setattr__
class DeviceState:
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: PhaseSpec = field(default_factory=PhaseSpec)
    providers: List[FeatureProvider] = field(default_factory=list)

    def vector(self) -> Array:
        parts: List[Array] = []
        for p in self.providers:
            v = p.vector()
            if v.size:
                parts.append(v)
        
        return (
            np.concatenate(parts, dtype=np.float32) 
            if parts 
            else np.zeros(0, np.float32)
        )

    def names(self) -> List[str]:
        out: List[str] = []
        for p in self.providers:
            out.extend(p.names())
        return out

    def clamp_(self) -> "DeviceState":
        for p in self.providers:
            p.clamp_()
        return self

    def to_dict(self) -> Dict:
        return {
            "phase_model": self.phase_model.value,
            "phase_spec": {
                "phases": self.phase_spec.phases,
                "has_neutral": self.phase_spec.has_neutral,
                "earth_bond": self.phase_spec.earth_bond,
            },
            "providers": [
                {"type": p.__class__.__name__, "data": p.to_dict()} 
                for p in self.providers
            ],
        }

    @classmethod
    def from_dict(
        cls, 
        d: Dict, 
        type_map: Optional[Dict[str, type]] = None
    ) -> "DeviceState":
        pm = PhaseModel(d.get("phase_model", "balanced_1ph"))
        psd = d.get("phase_spec", {"phases": "ABC", "has_neutral": False, "earth_bond": True})
        ps = PhaseSpec(
            psd.get("phases", "ABC"),
            psd.get("has_neutral", False),
            psd.get("earth_bond", True),
        )

        # Local fallback only for providers defined in THIS file (e.g., StatusFlags).
        local = {"StatusFlags": StatusFlags}

        # Merge: explicit overrides > local fallback
        tmap = dict(local)
        if type_map:
            tmap.update(type_map)

        provs: List[FeatureProvider] = []
        for item in d.get("providers", []):
            typ_name = item["type"]
            data = item["data"]
            # Prefer explicit map, else ask the global registry
            typ = tmap.get(typ_name) or ProviderRegistry.get(typ_name)
            if typ is None:
                # Unknown provider type: skip or raise. Skipping is safe:
                # raise ValueError(f"Unknown provider type: {typ_name}")
                continue
            provs.append(typ.from_dict(data))

        return cls(phase_model=pm, phase_spec=ps, providers=provs)

    def to_phase_model(
        self,
        model: PhaseModel,
        spec: Optional[PhaseSpec] = None,
        policy: CollapsePolicy = CollapsePolicy.SUM_PQ_MEAN_V,
    ) -> "DeviceState":
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
