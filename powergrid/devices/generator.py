from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
from builtins import float

from ..agents.device_agent import DeviceAgent
from ..core.protocols import NoProtocol, Protocol
from ..core.policies import Policy
from ..utils.cost import cost_from_curve
from ..utils.safety import s_over_rating, pf_penalty

class DG(DeviceAgent):
    """Distributed Generator (optionally with unit commitment)."""

    def __init__(
        self,
        name: str,
        bus: str,
        min_p_mw: float,
        max_p_mw: float,
        *,
        min_q_mvar: float = np.nan,
        max_q_mvar: float = np.nan,
        sn_mva: float = np.nan,
        min_pf: Optional[float] = None,
        leading: Optional[bool] = None,
        cost_curve_coefs: Sequence[float] = (0.0, 0.0, 0.0),
        startup_time: Optional[int] = None,
        shutdown_time: Optional[int] = None,
        startup_cost: float = 0.0,
        shutdown_cost: float = 0.0,
        type: str = "fossil",
        dt: float = 1.0,
        # Base class args
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        self.type = type
        self.name = name
        self.bus = bus
        self.min_p_mw = float(min_p_mw)
        self.max_p_mw = float(max_p_mw)
        self.min_q_mvar = float(min_q_mvar)
        self.max_q_mvar = float(max_q_mvar)
        self.sn_mva = float(sn_mva)
        self.leading = leading
        self.cost_curve_coefs = list(cost_curve_coefs)
        self.dt = float(dt)
        self.min_pf = min_pf
        self.startup_time = None  # set later if UC enabled

        if not np.isnan(self.sn_mva):
            # capability curve at max P
            self.min_q_mvar = -float(np.sqrt(self.sn_mva**2 - self.max_p_mw**2))
            self.max_q_mvar = float(np.sqrt(self.sn_mva**2 - self.max_p_mw**2))

        if startup_time is not None:
            # enable UC fields on state
            self.startup_time = int(startup_time)
            self.shutdown_time = int(shutdown_time or 0)
            self.startup_cost = float(startup_cost)
            self.shutdown_cost = float(shutdown_cost)

        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_device_state(self):
        if self.startup_time is not None:
            self.state.Pmax = self.max_p_mw
            self.state.Pmin = self.min_p_mw
            self.state.Qmax = self.max_q_mvar
            self.state.Qmin = self.min_q_mvar
            self.state.shutting = 0
            self.state.starting = 0

    def set_action_space(self) -> None:
        # continuous
        if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
            low = [self.min_p_mw, self.min_q_mvar]
            high = [self.max_p_mw, self.max_q_mvar]
        else:
            low, high = [self.min_p_mw], [self.max_p_mw]
        self.action.range = np.array([low, high], dtype=np.float32)
        self.action.dim_c = len(low)
        self.action.sample()

        # UC discrete (0 = stop/keep off, 1 = start/keep on)
        if hasattr(self, "startup_time"):
            self.action.ncats = 2
            self.action.dim_d = 1
            if self.action.d.size == 0:
                self.action.sample()

    def update_state(self) -> None:
        # UC progression
        if hasattr(self.state, "shutting") and self.action.dim_d > 0:
            self._update_uc_state()

        # P/Q from continuous action
        if self.action.c.size == 2:
            self.state.P, self.state.Q = map(float, self.action.c)
        elif self.action.c.size == 1:
            self.state.P = float(self.action.c[0])

    def _update_uc_state(self) -> None:
        assert not (self.state.shutting and self.state.starting)
        if not (self.state.shutting or self.state.starting):
            self.uc_cost = 0.0

        a = int(self.action.d[0]) if self.action.d.size else 1
        # shutting
        if self.state.on and (a == 0):
            self.state.shutting += 1
            if self.state.shutting > self.shutdown_time:
                self.state.on = 0
                self.state.shutting = 0
                self.uc_cost = self.shutdown_cost
        # starting
        if (not self.state.on) and (a == 1):
            self.state.starting += 1
            if self.state.starting > self.startup_time:
                self.state.on = 1
                self.state.starting = 0
                self.uc_cost = self.startup_cost

    def update_cost_safety(self) -> None:
        P = float(getattr(self.state, "P", 0.0))
        cost = cost_from_curve(P, self.cost_curve_coefs)
        self.cost = (self.state.on * cost * self.dt) + getattr(self, "uc_cost", 0.0) * self.dt

        # safety via shared helpers
        safety = 0.0
        if self.action.dim_c > 1:
            safety += s_over_rating(self.state.P, self.state.Q, self.sn_mva)
            safety += pf_penalty(self.state.P, self.state.Q, self.min_pf)

        self.safety = safety * self.dt

    def reset_device(self, rnd=None) -> None:
        # reset P/Q
        if self.action.c.size == 2:
            self.state.P = 0.0
            self.state.Q = 0.0
        else:
            self.state.P = 0.0
        # UC fields
        if hasattr(self.state, "shutting"):
            self.state.shutting = 0
            self.state.starting = 0
            self.state.on = 1
        self.cost = 0.0
        self.safety = 0.0


class RES(DeviceAgent):
    """Renewable energy source (solar/wind)."""

    def __init__(
        self,
        name: str,
        bus: str,
        sn_mva: float,
        source: str,
        *,
        max_q_mvar: float = np.nan,
        min_q_mvar: float = np.nan,
        cost_curve_coefs = (0.0, 0.0, 0.0),
        dt: float = 1.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        
        assert source in {"solar", "wind"}
        self.type = source
        self.name = name
        self.bus = bus
        self.sn_mva = float(sn_mva)
        self.max_p_mw = float(sn_mva)
        self.min_p_mw = 0.0
        self.state.Pmax = self.max_p_mw
        self.state.Pmin = self.min_p_mw
        self.max_q_mvar = float(max_q_mvar)
        self.min_q_mvar = float(min_q_mvar)
        self.cost_curve_coefs = list(cost_curve_coefs)
        self.dt = float(dt)
        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        if not np.isnan(self.max_q_mvar):
            low, high = [self.min_q_mvar], [self.max_q_mvar]
            self.action.range = np.array([low, high], dtype=np.float32)
            self.action.dim_c = 1
            self.action.sample()
        else:
            self.action_callback = True

    def set_device_state(self) -> None:
        pass

    def update_state(self, *, scaling: Optional[float] = None) -> None:
        if scaling is not None:
            assert 0.0 <= scaling <= 1.0
            self.state.P = float(self.sn_mva * scaling)
        if self.action.c.size > 0:
            self.state.Q = float(self.action.c if np.isscalar(self.action.c) else self.action.c[0])

    def update_cost_safety(self) -> None:
        if self.action.c.size > 0:
            S = float(np.hypot(self.state.P, self.state.Q))
            self.safety = max(0.0, S - self.sn_mva) * self.dt
        else:
            self.safety = 0.0

    def reset_device(self, rnd=None) -> None:
        self.state.P = 0.0
        if self.action.c.size > 0:
            self.state.Q = 0.0
        self.cost = 0.0
        self.safety = 0.0