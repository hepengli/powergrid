from typing import Any, Optional, Dict
from builtins import float

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.policies import Policy
from powergrid.utils.cost import tap_change_cost
from powergrid.utils.safety import loading_over_pct


class Transformer(DeviceAgent):
    """On-load tap changer (OLTC) transformer — **controllable**, not passive.

    Discrete action selects tap index in [tap_min, tap_max]. Optional
    `tap_change_cost` applies per step moved to account for wear/operations.
    """

    def __init__(
        self,
        name: str,
        *,
        sn_mva: Optional[float] = None,
        tap_max: Optional[int] = None,
        tap_min: Optional[int] = None,
        dt: float = 1.0,
        tap_change_cost: float = 0.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        self.name = name
        self.sn_mva = sn_mva
        self.tap_max = tap_max
        self.tap_min = tap_min
        self.dt = float(dt)
        self.tap_change_cost = float(tap_change_cost)
        self._last_tap_position = self.state.tap_position

        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        if self.tap_max is not None and self.tap_min is not None:
            self.action.ncats = self.tap_max - self.tap_min + 1
            self.action.dim_d = 1
            self.action.sample()

    def set_device_state(self, config: Dict[str, Any]) -> None:
        self.state.loading_percentage = 0.0
        self.state.tap_position = self.tap_min if self.tap_min is not None else 0
        if self.tap_max is not None and self.tap_min is not None:
            self.state.tap_max = self.tap_max
            self.state.tap_min = self.tap_min

    def update_state(self) -> None:
        if self.tap_max is not None and self.tap_min is not None and self.action.d.size:
            self.state.tap_position = int(self.action.d[0]) + int(self.tap_min)

    def update_cost_safety(self, *, loading_percentage: float) -> None:
        # loading-derived safety
        self.state.loading_percentage = float(loading_percentage)
        self.safety = loading_over_pct(self.state.loading_percentage)
        # tap change cost
        delta = abs(self.state.tap_position - getattr(self, "_last_tap_position", self.state.tap_position))
        self.cost = tap_change_cost(delta, self.tap_change_cost)
        self._last_tap_position = self.state.tap_position

    def reset_device(self, rnd=None) -> None:
        self.state.loading_percentage = 0.0
        self.state.tap_position = self.tap_min if self.tap_min is not None else 0
        self._last_tap_position = self.state.tap_position
        self.cost = 0.0
        self.safety = 0.0