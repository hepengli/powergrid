"""Generator device implementations for power systems.

This module provides distributed generator (DG) and renewable energy source (RES)
implementations that extend DeviceAgent with power generation capabilities.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.utils.cost import cost_from_curve
from powergrid.utils.safety import pf_penalty, s_over_rating


class DG(DeviceAgent):
    """Distributed Generator with optional unit commitment.

    Attributes:
        type: Generator type (e.g., 'fossil', 'hydro')
        name: Generator name
        bus: Bus connection identifier
        min_p_mw: Minimum active power output
        max_p_mw: Maximum active power output
        min_q_mvar: Minimum reactive power
        max_q_mvar: Maximum reactive power
        sn_mva: Apparent power rating
        cost_curve_coefs: Cost curve coefficients (quadratic)
        startup_time: Timesteps required for startup (None = no UC)
        shutdown_time: Timesteps required for shutdown
        startup_cost: Cost incurred on startup
        shutdown_cost: Cost incurred on shutdown
    """

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
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        """Initialize distributed generator.

        Args:
            name: Generator identifier
            bus: Bus connection
            min_p_mw: Minimum active power
            max_p_mw: Maximum active power
            min_q_mvar: Minimum reactive power
            max_q_mvar: Maximum reactive power
            sn_mva: Apparent power rating (determines Q limits if provided)
            min_pf: Minimum power factor
            leading: Whether power factor is leading
            cost_curve_coefs: Quadratic cost curve coefficients [a, b, c]
            startup_time: Timesteps for startup (enables unit commitment)
            shutdown_time: Timesteps for shutdown
            startup_cost: Startup cost ($)
            shutdown_cost: Shutdown cost ($)
            type: Generator type identifier
            dt: Timestep duration (hours)
            policy: Control policy
            protocol: Coordination protocol
            device_config: Additional configuration
        """
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
            self.shutdown_time = int(shutdown_time) if shutdown_time is not None else 0
            self.startup_cost = float(startup_cost)
            self.shutdown_cost = float(shutdown_cost)
        else:
            self.startup_time = None
            self.shutdown_time = None

        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    # Initialization methods
    def set_device_state(self):
        """Initialize generator state including UC fields if enabled."""
        if self.startup_time is not None:
            self.state.Pmax = self.max_p_mw
            self.state.Pmin = self.min_p_mw
            self.state.Qmax = self.max_q_mvar
            self.state.Qmin = self.min_q_mvar
            self.state.shutting = 0
            self.state.starting = 0

    def set_action_space(self) -> None:
        """Define action space for P, Q control and optional UC."""
        # Continuous P, Q actions
        if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
            low = [self.min_p_mw, self.min_q_mvar]
            high = [self.max_p_mw, self.max_q_mvar]
        else:
            low, high = [self.min_p_mw], [self.max_p_mw]
        self.action.range = np.array([low, high], dtype=np.float32)
        self.action.dim_c = len(low)
        self.action.sample()

        # UC discrete action (0 = stop/keep off, 1 = start/keep on)
        if self.startup_time is not None:
            self.action.ncats = 2
            self.action.dim_d = 1
            if self.action.d.size == 0:
                self.action.sample()

    # State update methods
    def update_state(self) -> None:
        """Update generator state from actions."""
        # UC progression
        if hasattr(self.state, "shutting") and self.action.dim_d > 0:
            self._update_uc_state()

        # P/Q from continuous action
        if self.action.c.size == 2:
            self.state.P, self.state.Q = map(float, self.action.c)
        elif self.action.c.size == 1:
            self.state.P = float(self.action.c[0])

    def _update_uc_state(self) -> None:
        """Update unit commitment state machine."""
        assert not (self.state.shutting and self.state.starting)
        if not (self.state.shutting or self.state.starting):
            self.uc_cost = 0.0

        a = int(self.action.d[0]) if self.action.d.size else 1
        # Shutting down
        if self.state.on and (a == 0) and self.shutdown_time is not None:
            self.state.shutting = (self.state.shutting or 0) + 1
            if self.state.shutting > self.shutdown_time:
                self.state.on = 0
                self.state.shutting = 0
                self.uc_cost = self.shutdown_cost
        # Starting up
        if (not self.state.on) and (a == 1):
            self.state.starting = (self.state.starting or 0) + 1
            if self.state.starting > self.startup_time:
                self.state.on = 1
                self.state.starting = 0
                self.uc_cost = self.startup_cost

    def update_cost_safety(self) -> None:
        """Calculate cost and safety penalties."""
        P = float(getattr(self.state, "P", 0.0))
        cost = cost_from_curve(P, self.cost_curve_coefs)
        self.cost = (self.state.on * cost * self.dt) + getattr(self, "uc_cost", 0.0) * self.dt

        # Safety penalties
        safety = 0.0
        if self.action.dim_c > 1:
            safety += s_over_rating(self.state.P, self.state.Q, self.sn_mva)
            safety += pf_penalty(self.state.P, self.state.Q, self.min_pf)

        self.safety = safety * self.dt

    def reset_device(self, rnd=None) -> None:
        """Reset generator to initial state.

        Args:
            rnd: Random number generator (unused)
        """
        # Reset P/Q
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

    def __repr__(self) -> str:
        """Return string representation of the DG.

        Returns:
            String representation
        """
        return f"DG(name={self.name}, type={self.type}, bus={self.bus}, P=[{self.min_p_mw}, {self.max_p_mw}]MW)"


class RES(DeviceAgent):
    """Renewable Energy Source (solar/wind).

    RES devices have their active power determined by external scaling factors
    (solar irradiance, wind speed) rather than direct control.

    Attributes:
        type: Source type ('solar' or 'wind')
        name: Device name
        bus: Bus connection
        sn_mva: Apparent power rating
        max_p_mw: Maximum active power (equals sn_mva)
        max_q_mvar: Maximum reactive power
        min_q_mvar: Minimum reactive power
    """

    def __init__(
        self,
        name: str,
        bus: str,
        sn_mva: float,
        source: str,
        *,
        max_q_mvar: float = np.nan,
        min_q_mvar: float = np.nan,
        cost_curve_coefs=(0.0, 0.0, 0.0),
        dt: float = 1.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        """Initialize renewable energy source.

        Args:
            name: Device identifier
            bus: Bus connection
            sn_mva: Apparent power rating
            source: Source type ('solar' or 'wind')
            max_q_mvar: Maximum reactive power
            min_q_mvar: Minimum reactive power
            cost_curve_coefs: Cost curve coefficients
            dt: Timestep duration (hours)
            policy: Control policy
            protocol: Coordination protocol
            device_config: Additional configuration
        """
        assert source in {"solar", "wind"}
        self.type = source
        self.name = name
        self.bus = bus
        self.sn_mva = float(sn_mva)
        self.max_p_mw = float(sn_mva)
        self.min_p_mw = 0.0
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

    # Initialization methods
    def set_action_space(self) -> None:
        """Define action space for reactive power control only."""
        if not np.isnan(self.max_q_mvar):
            low, high = [self.min_q_mvar], [self.max_q_mvar]
            self.action.range = np.array([low, high], dtype=np.float32)
            self.action.dim_c = 1
            self.action.sample()
        else:
            # No Q control - P is determined externally by scaling
            # Set action_callback and create dummy action space
            self.action_callback = True
            # Create a dummy discrete action space with 1 option (no-op)
            self.action.ncats = 1
            self.action.dim_d = 1
            self.action.sample()

    def set_device_state(self) -> None:
        """Initialize RES state."""
        self.state.Pmax = self.max_p_mw
        self.state.Pmin = self.min_p_mw

    # State update methods
    def update_state(self, *, scaling: Optional[float] = None) -> None:
        """Update RES state from scaling factor and Q action.

        Args:
            scaling: Solar irradiance or wind speed scaling factor (0-1)
        """
        if scaling is not None:
            assert 0.0 <= scaling <= 1.0
            self.state.P = float(self.sn_mva * scaling)
        if self.action.c.size > 0:
            self.state.Q = float(self.action.c if np.isscalar(self.action.c) else self.action.c[0])

    def update_cost_safety(self) -> None:
        """Calculate safety penalty for apparent power exceeding rating."""
        if self.action.c.size > 0:
            S = float(np.hypot(self.state.P, self.state.Q))
            self.safety = max(0.0, S - self.sn_mva) * self.dt
        else:
            self.safety = 0.0

    def reset_device(self, rnd=None) -> None:
        """Reset RES to initial state.

        Args:
            rnd: Random number generator (unused)
        """
        self.state.P = 0.0
        if self.action.c.size > 0:
            self.state.Q = 0.0
        self.cost = 0.0
        self.safety = 0.0

    def __repr__(self) -> str:
        """Return string representation of the RES.

        Returns:
            String representation
        """
        return f"RES(name={self.name}, type={self.type}, bus={self.bus}, sn_mva={self.sn_mva}MVA)"
