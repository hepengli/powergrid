from typing import Any, Optional, Dict

import numpy as np

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.policies import Policy
from powergrid.devices.features.storage import StorageBlock
from powergrid.devices.features.electrical import ElectricalBasePh
from powergrid.utils.cost import cost_from_curve
from powergrid.utils.safety import s_over_rating, soc_bounds_penalty


class ESS(DeviceAgent):
    """Energy Storage System with SOC and optional Q support."""

    def __init__(
        self,
        name: str,
        bus: int,
        min_p_mw: float,
        max_p_mw: float,
        capacity: float,
        *,
        max_e_mwh: Optional[float] = None,
        min_e_mwh: float = 0.0,
        init_soc: float = 0.5,
        min_q_mvar: float = np.nan,
        max_q_mvar: float = np.nan,
        sn_mva: float = np.nan,
        ch_eff: float = 0.98,
        dsc_eff: float = 0.98,
        cost_curve_coefs = (0.0, 0.0, 0.0),
        dt: float = 1.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        self.type = "ESS"
        self.name = name
        self.bus = bus
        self.capacity = float(capacity)
        self.min_p_mw = float(min_p_mw)
        self.max_p_mw = float(max_p_mw)
        self.max_e_mwh = float(max_e_mwh if max_e_mwh is not None else capacity)
        self.min_e_mwh = float(min_e_mwh)
        assert 0.0 <= self.min_e_mwh < self.max_e_mwh <= self.capacity
        self.min_q_mvar = float(min_q_mvar)
        self.max_q_mvar = float(max_q_mvar)
        self.sn_mva = float(sn_mva)
        self.dt = float(dt)
        self.ch_eff = float(ch_eff)
        self.dsc_eff = float(dsc_eff)
        self.cost_curve_coefs = list(cost_curve_coefs)
        self.init_soc = float(init_soc)
        self.min_soc = self.min_e_mwh / self.capacity
        self.max_soc = self.max_e_mwh / self.capacity

        if not np.isnan(self.sn_mva):
            self.min_q_mvar = -float(np.sqrt(self.sn_mva**2 - self.max_p_mw**2))
            self.max_q_mvar = float(np.sqrt(self.sn_mva**2 - self.max_p_mw**2))

        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_device_state(self, config: Dict[str, Any]) -> None:
        # Initialize state with StorageBlock and ElectricalBasePh features
        storage_block = StorageBlock(
            soc=self.init_soc,
            soc_min=self.min_soc,
            soc_max=self.max_soc,
            e_capacity_MWh=self.capacity,
            p_ch_max_MW=self.max_p_mw,
            p_dis_max_MW=-self.min_p_mw,  # Note: discharge is positive
            eta_ch=self.ch_eff,
            eta_dis=self.dsc_eff,
        )
        electrical_block = ElectricalBasePh(
            P_MW=0.0,
            Q_MVAr=0.0 if not np.isnan(self.max_q_mvar) else None,
        )
        self.state.features = [storage_block, electrical_block]

    def set_action_space(self) -> None:
        if not np.isnan(self.sn_mva) or not np.isnan(self.max_q_mvar):
            low = [self.min_p_mw, self.min_q_mvar]
            high = [self.max_p_mw, self.max_q_mvar]
        else:
            low, high = [self.min_p_mw], [self.max_p_mw]
        self.action.range = np.array([low, high], dtype=np.float32)
        self.action.dim_c = len(low)
        self.action.sample()

    def update_state(self) -> None:
        # Get features
        storage_block = self.storage_block
        electrical_block = self.electrical_block

        # Update P and Q from action
        if self.action.c.size > 1:
            P, Q = map(float, self.action.c)
            electrical_block.P_MW = P
            electrical_block.Q_MVAr = Q
        else:
            P = float(self.action.c[0])
            electrical_block.P_MW = P

        # SOC dynamics (P>=0 charging; P<0 discharging). SOC is fraction of *capacity*.
        if P >= 0:
            storage_block.soc += P * self.ch_eff * self.dt / self.capacity
        else:
            storage_block.soc += P / self.dsc_eff * self.dt / self.capacity

    def update_cost_safety(self) -> None:
        # Get features
        storage_block = self.storage_block
        electrical_block = self.electrical_block

        P = float(electrical_block.P_MW or 0.0)
        cost = cost_from_curve(P, self.cost_curve_coefs)
        # Note: we don't have 'on' state in new architecture, assume always on
        self.cost = cost * self.dt

        s = 0.0
        s += s_over_rating(
            P,
            float(electrical_block.Q_MVAr or 0.0),
            None if np.isnan(self.sn_mva) else self.sn_mva
        )
        s += soc_bounds_penalty(storage_block.soc, self.min_soc, self.max_soc)
        self.safety = s * self.dt

    def feasible_action(self) -> None:
        # Get storage provider
        storage_block = self.storage_block

        # compute instantaneous feasible P based on available energy windows
        max_dsc_power = (storage_block.soc - self.min_soc) * self.capacity * self.dsc_eff / self.dt
        max_dsc_power = min(max_dsc_power, -self.min_p_mw)  # note: discharging => negative P allowed up to -min_p

        max_ch_power = (self.max_soc - storage_block.soc) * self.capacity / self.ch_eff / self.dt
        max_ch_power = min(max_ch_power, self.max_p_mw)

        low, high = -max_dsc_power, max_ch_power
        if self.action.c.size > 1:
            low = np.array([low, self.min_q_mvar], dtype=np.float32)
            high = np.array([high, self.max_q_mvar], dtype=np.float32)
        self.action.c = np.clip(self.action.c, low, high)

    def reset_device(self, *, rnd=None, init_soc: Optional[float] = None) -> None:
        rnd = np.random if rnd is None else rnd

        # Get features
        storage_block = self.storage_block
        electrical_block = self.electrical_block

        # Reset SOC
        storage_block.soc = float(init_soc) if init_soc is not None else float(
            rnd.uniform(self.min_soc, self.max_soc)
        )

        # Reset P and Q
        electrical_block.P_MW = 0.0
        if self.action.c.size > 1:
            electrical_block.Q_MVAr = 0.0

        self.cost = 0.0
        self.safety = 0.0

    @property
    def storage_block(self) -> StorageBlock:
        """Get the StorageBlock provider from state."""
        for provider in self.state.features:
            if isinstance(provider, StorageBlock):
                return provider
        raise ValueError("StorageBlock provider not found in state")

    @property
    def electrical_block(self) -> ElectricalBasePh:
        """Get the ElectricalBasePh provider from state."""
        for provider in self.state.features:
            if isinstance(provider, ElectricalBasePh):
                return provider
        raise ValueError("ElectricalBasePh provider not found in state")

    def __repr__(self) -> str:
        """Return string representation of the ESS.

        Returns:
            String representation
        """
        return f"ESS(name={self.name}, bus={self.bus}, capacity={self.capacity}MWh, device_states=[{self.storage_block}, {self.electrical_block}])"