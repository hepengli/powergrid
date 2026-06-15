# tests/test_grid.py
import pytest

# Adjust this import to your package path if needed:
from powergrid.devices.grid import Grid


def test_grid_init_defaults_and_set_action_space_noop():
    g = Grid(name="main", bus=1, sn_mva=100.0)
    assert g.type == "GRID"
    assert g.name == "main"
    assert g.bus == 1
    assert g.sn_mva == 100.0
    assert g.sell_discount == 1.0
    assert g.dt == 1.0
    assert g.action_callback is True
    # initial state
    assert g.state.P == 0.0
    assert g.state.Q == 0.0
    assert g.state.price == 0.0
    # set_action_space returns None and should not change anything
    assert g.set_action_space() is None
    assert g.action_callback is True


def test_update_state_individual_fields():
    g = Grid(name="main", bus=1, sn_mva=100.0)
    # only price
    g.update_state(price=50.0)
    assert g.state.price == 50.0
    assert g.state.P == 0.0 and g.state.Q == 0.0
    # only P
    g.update_state(P=10.0)
    assert g.state.P == 10.0 and g.state.price == 50.0
    # only Q
    g.update_state(Q=-3.0)
    assert g.state.Q == -3.0


@pytest.mark.parametrize(
    "P, price, dt, expected_cost",
    [
        # buying (P>0): cost = P * price * dt
        (10.0, 100.0, 1.0, 1000.0),
        (5.0,  80.0, 0.5, 200.0),
        # zero price => zero cost regardless of P
        (10.0, 0.0,  1.0, 0.0),
    ],
)
def test_grid_cost_buying(P, price, dt, expected_cost):
    g = Grid(name="main", bus=1, sn_mva=100.0, dt=dt, sell_discount=0.8)
    g.update_state(P=P, price=price)
    g.update_cost_safety()
    assert g.cost == pytest.approx(expected_cost, rel=1e-6)
    assert g.safety == 0.0  # always zero by design


@pytest.mark.parametrize(
    "P, price, dt, disc, expected_cost",
    [
        # selling (P<0): cost = P * price * sell_discount * dt  (negative => credit)
        (-10.0, 100.0, 1.0, 1.0, -1000.0),
        (-10.0, 100.0, 1.0, 0.8, -800.0),   # discounted export price
        (-5.0,   50.0, 0.5, 0.5, -62.5),    # dt & discount both apply
    ],
)
def test_grid_cost_selling_with_discount(P, price, dt, disc, expected_cost):
    g = Grid(name="main", bus=1, sn_mva=100.0, dt=dt, sell_discount=disc)
    g.update_state(P=P, price=price)
    g.update_cost_safety()
    assert g.cost == pytest.approx(expected_cost, rel=1e-6)
    assert g.safety == 0.0


def test_reset_zeros_everything():
    g = Grid(name="main", bus=1, sn_mva=100.0, dt=2.0, sell_discount=0.9)
    g.update_state(P=10.0, price=100.0, Q=2.0)
    g.update_cost_safety()
    assert g.cost != 0.0
    g.reset()
    assert g.state.P == 0.0
    assert g.state.Q == 0.0
    assert g.state.price == 0.0
    assert g.cost == 0.0
    assert g.safety == 0.0
