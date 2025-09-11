from .base import Device

from ..utils.cost import energy_cost

class Grid(Device):
    """Main grid interface. Convention: P>0 buys, P<0 sells."""

    def __init__(self, name: str, bus: int, sn_mva: float, *, sell_discount: float = 1.0, dt: float = 1.0):
        super().__init__()
        self.type = "GRID"
        self.name = name
        self.bus = bus
        self.sn_mva = float(sn_mva)
        self.sell_discount = float(sell_discount)
        self.dt = float(dt)
        self.action_callback = True
        self.state.P = 0.0
        self.state.Q = 0.0
        self.state.price = 0.0

    def set_action_space(self) -> None:
        return None

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

    def reset(self, rnd=None) -> None:
        self.state.P = 0.0
        self.state.Q = 0.0
        self.state.price = 0.0
        self.cost = 0.0
        self.safety = 0.0