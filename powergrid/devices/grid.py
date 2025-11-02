from builtins import float
from typing import Any, Dict, Optional

from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.policies import Policy
from powergrid.utils.cost import energy_cost

class Grid(DeviceAgent):
    """Main grid interface. Convention: P>0 buys, P<0 sells."""

    def __init__(
        self, 
        name: str, 
        bus: int, 
        sn_mva: float, 
        *, 
        sell_discount: float = 1.0, 
        dt: float = 1.0,
        # Base class args
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
    ) -> None:
        self.type = "GRID"
        self.name = name
        self.bus = bus
        self.sn_mva = float(sn_mva)
        self.sell_discount = float(sell_discount)
        self.dt = float(dt)
        self.action_callback = True
        super().__init__(
            agent_id=name,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        pass

    def set_device_state(self, config: Dict[str, Any]) -> None:
        self.state.P = 0.0
        self.state.Q = 0.0
        self.state.price = 0.0

    def update_state(self, *, price: float = None, P: float = None, Q: float = None) -> None:
        if P is not None:
            self.state.P = float(P)
        if Q is not None:
            self.state.Q = float(Q)
        if price is not None:
            self.state.price = float(price)

    def update_cost_safety(self) -> None:
        cost = energy_cost(
            self.state.P,
            self.state.price,
            self.sell_discount,
        )
        self.cost = cost * self.dt
        self.safety = 0.0

    def reset_device(self, rnd=None) -> None:
        self.state.P = 0.0
        self.state.Q = 0.0
        self.state.price = 0.0
        self.cost = 0.0
        self.safety = 0.0