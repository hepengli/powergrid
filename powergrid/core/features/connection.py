# connection.py (only the changed/clarified parts)

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np

from powergrid.core.features.base import FeatureProvider
from powergrid.core.state import PhaseModel, PhaseSpec
from powergrid.utils.registry import provider

_CONN_SET = {"A", "B", "C", "AB", "BC", "CA", "ABC"}


@provider()
@dataclass(slots=True)
class PhaseConnection(FeatureProvider):
    phase_model: PhaseModel = PhaseModel.BALANCED_1PH
    phase_spec: Optional[PhaseSpec] = field(default_factory=PhaseSpec)
    connection: Optional[str] = None  # presence (1φ) or subset token (3φ)

    # Used only when a non-ON/OFF token is seen under BALANCED_1PH construction
    _pending_token: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Construction-time rules:
          • BALANCED_1PH:
              - Ignore spec.
              - Accept ON/OFF tokens; anything else is DEFERRED (saved to _pending_token).
          • THREE_PHASE:
              - REQUIRE a PhaseSpec.
              - If connection is provided, VALIDATE NOW and raise on bad tokens.
        """
        if self.phase_model == PhaseModel.BALANCED_1PH:
            self.phase_spec = None
            self._normalize_conn_balanced_defer_()
            return

        if self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            # STRICT: if a connection was provided, validate immediately (no deferral).
            if self.connection is not None:
                self._validate_conn_three_phase_(self.connection, self.phase_spec)
            return

        raise ValueError(f"Unsupported phase model: {self.phase_model}")

    # balanced normalization with DEFER
    def _normalize_conn_balanced_defer_(self) -> None:
        """BALANCED_1PH at init: accept ON/OFF; defer others (e.g., 'BC')."""
        if self.connection is None:
            return
        token = str(self.connection).strip().upper()
        off = {"", "OFF", "NONE", "FALSE", "0"}
        on  = {"ON", "TRUE", "1"}

        if token in off:
            self.connection = None
        elif token in on:
            self.connection = "ON"
        else:
            # Not a valid balanced token (maybe a 3φ subset) → defer
            self._pending_token = token
            self.connection = None  # names()/vector() will be empty until resolved

    def _validate_conn_three_phase_(self, conn: str, spec: PhaseSpec) -> None:
        """Strict 3ϕ validation (raises on invalid/outsider tokens)."""
        token = str(conn).strip().upper()
        if token not in _CONN_SET:
            raise ValueError(
                f"Unknown connection '{conn}'. Expected one of {_CONN_SET}."
            )
        if not set(token).issubset(set(spec.phases)):
            raise ValueError(
                f"Connection '{token}' not subset of PhaseSpec '{spec.phases}'."
            )
        # normalize on success
        self.connection = token

    # Called by DeviceState after it overwrites phase_model/spec
    def revalidate_after_context(self) -> None:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            # If still 1φ and had a deferred token, that's invalid now.
            if self._pending_token is not None:
                tok = self._pending_token
                self._pending_token = None
                raise ValueError(
                    f"Invalid connection token '{tok}' for BALANCED_1PH. "
                    "Use one of {'', 'OFF', 'NONE', 'FALSE', '0'} for off "
                    "or {'ON','TRUE','1'} for on."
                )
            # Re-check strictly if any new value was assigned
            self._normalize_conn_balanced_strict_()
            return

        if self.phase_model == PhaseModel.THREE_PHASE:
            if not isinstance(self.phase_spec, PhaseSpec):
                raise ValueError("THREE_PHASE requires a PhaseSpec.")
            # Promote a deferred token (e.g., 'BC') now that we are 3φ
            if self._pending_token is not None:
                tok = self._pending_token
                self._validate_conn_three_phase_(tok, self.phase_spec)
                self._pending_token = None
            elif self.connection is not None:
                # If a connection exists, validate it under the final spec
                self._validate_conn_three_phase_(self.connection, self.phase_spec)
            return

        raise ValueError(f"Unsupported phase model: {self.phase_model}")

    def _normalize_conn_balanced_strict_(self) -> None:
        if self.connection is None:
            return
        token = str(self.connection).strip().upper()
        if token in {"", "OFF", "NONE", "FALSE", "0"}:
            self.connection = None
        elif token in {"ON", "TRUE", "1"}:
            self.connection = "ON"
        else:
            raise ValueError(
                f"Invalid connection token '{self.connection}' for BALANCED_1PH. "
                "Use one of {'', 'OFF', 'NONE', 'FALSE', '0', 'ON', 'TRUE', '1'}."
            )

    def _mask(self) -> np.ndarray:
        n = self.phase_spec.nph()  # type: ignore
        m = np.zeros(n, np.float32)
        if self.connection is None:
            return m
        phs = self.phase_spec.phases  # type: ignore
        for i, p in enumerate(phs):
            if p in self.connection:
                m[i] = 1.0
        return m

    def vector(self) -> np.ndarray:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return np.array([1.0], np.float32) if self.connection else np.zeros(0, np.float32)
        if self.connection is None:
            return np.zeros(0, np.float32)
        return self._mask()

    def names(self) -> List[str]:
        if self.phase_model == PhaseModel.BALANCED_1PH:
            return ["conn_present"] if self.connection else []
        if self.connection is None:
            return []
        return [f"conn_{p}" for p in self.phase_spec.phases]  # type: ignore

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
        pm = self.phase_model
        d["phase_model"] = pm.value if isinstance(pm, PhaseModel) else str(pm)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PhaseConnection":
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

        return cls(
            phase_model=pm,
            phase_spec=ps,
            connection=d.get("connection"),
        )
