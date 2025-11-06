# tests/test_generator_res.py
import math
import numpy as np
import pytest

# Adjust this import to match your package path if needed:
from powergrid.devices.generator import DG, RES


def test_dg_action_dims_continuous_only():
    # No Q capability -> 1D continuous action
    dg = DG(name="g1", bus=1, min_p_mw=0.0, max_p_mw=10.0, sn_mva=np.nan, max_q_mvar=np.nan)
    assert dg.action.dim_c == 1
    assert dg.action.c.size == 1


def test_dg_action_dims_with_q():
    # With S rating -> 2D (P, Q)
    dg = DG(name="g2", bus=1, min_p_mw=0.0, max_p_mw=10.0, sn_mva=12.0)
    assert dg.action.dim_c == 2


def test_dg_cost_quadratic_and_no_safety_when_only_P():
    # quadratic: a=0.01, b=1.0, c=2.0
    dg = DG(name="g3", bus=1, min_p_mw=0.0, max_p_mw=10.0, cost_curve_coefs=(0.01, 1.0, 2.0))
    dg.action.c = np.array([5.0], dtype=np.float32)  # P=5
    dg.update_state()
    dg.update_cost_safety()
    expected = (0.01 * 25 + 1.0 * 5 + 2.0) * dg.dt
    assert pytest.approx(dg.cost, rel=1e-6) == expected
    # safety requires Q dimension; here dim_c==1 so stays 0
    assert dg.safety == 0.0


def test_dg_cost_piecewise_linear():
    # Knots: (0,0) -> (10,50). At P=6 => 30.
    dg = DG(name="g4", bus=1, min_p_mw=0.0, max_p_mw=10.0, cost_curve_coefs=(0.0, 0.0, 10.0, 50.0))
    dg.action.c = np.array([6.0], dtype=np.float32)
    dg.update_state()
    dg.update_cost_safety()
    assert pytest.approx(dg.cost, rel=1e-6) == 30.0 * dg.dt


def test_dg_pf_and_overrating_safety():
    # Give Q capability and min_pf to trigger PF penalty term.
    dg = DG(
        name="g5",
        bus=1,
        min_p_mw=0.0,
        max_p_mw=10.0,
        sn_mva=10.0,     # enables (P,Q)
        min_pf=0.95,
    )
    # Choose P, Q such that S == sn_mva -> no S-overrating; PF = 0.8 -> penalty 0.15
    dg.action.c = np.array([8.0, 6.0], dtype=np.float32)  # P=8, Q=6 -> S=10
    dg.update_state()
    dg.update_cost_safety()
    assert pytest.approx(dg.safety, rel=1e-6) == 0.15  # 0.95 - 0.8
    # If P=10, Q=10 -> S≈14.142; overrating term positive
    dg.action.c = np.array([10.0, 10.0], dtype=np.float32)
    dg.update_state()
    dg.update_cost_safety()
    S = math.hypot(10.0, 10.0)
    over = max(0.0, S - 10.0)  # DG adds raw (S - sn_mva) here
    pf_pen = max(0.0, 0.95 - abs(10.0 / S))
    assert pytest.approx(dg.safety, rel=1e-6) == over + pf_pen


def test_dg_pf_penalty_zero_when_S_zero():
    dg = DG(name="g6", bus=1, min_p_mw=0.0, max_p_mw=10.0, sn_mva=10.0, min_pf=0.9)
    dg.action.c = np.array([0.0, 0.0], dtype=np.float32)
    dg.update_state()
    dg.update_cost_safety()
    # S==0 => pf_penalty should be 0 (by design to avoid division by zero)
    assert dg.safety == 0.0


def test_dg_unit_commitment_shutdown_and_startup_costs():
    dg = DG(
        name="g7",
        bus=1,
        min_p_mw=0.0,
        max_p_mw=10.0,
        startup_time=1,
        shutdown_time=1,
        startup_cost=5.0,
        shutdown_cost=3.0,
    )
    # Ensure P doesn't contribute to variable cost
    dg.action.c = np.array([0.0], dtype=np.float32)

    # Start from on=1 (reset behavior)
    assert dg.state.on == 1
    # Command shutdown (action.d == 0) twice to cross the >shutdown_time threshold
    dg.action.d = np.array([0], dtype=np.int32)
    dg.update_state()  # shutting=1
    dg.update_state()  # shutting=2 -> on=0, uc_cost=shutdown_cost
    dg.update_cost_safety()
    assert dg.state.on == 0
    assert pytest.approx(dg.cost, rel=1e-6) == 3.0 * dg.dt

    # Now command startup twice
    dg.action.d = np.array([1], dtype=np.int32)
    dg.update_state()  # starting=1
    dg.update_state()  # starting=2 -> on=1, uc_cost=startup_cost
    dg.update_cost_safety()
    assert dg.state.on == 1
    # cost includes startup UC cost; variable cost is 0 since P=0
    assert pytest.approx(dg.cost, rel=1e-6) == 5.0 * dg.dt


def test_res_scaling_and_q_and_safety():
    res = RES(name="r1", bus=1, sn_mva=5.0, source="solar", max_q_mvar=2.0, min_q_mvar=-2.0)
    # Set P via scaling
    res.update_state(scaling=0.6)  # P = 3.0
    assert pytest.approx(res.state.P, rel=1e-6) == 3.0
    # Set Q via action: handle both scalar or vector gracefully
    res.action.c = np.array([0.5], dtype=np.float32)
    res.update_state()  # applies Q
    assert pytest.approx(res.state.Q, rel=1e-6) == 0.5

    # Safety: S <= sn_mva -> 0
    res.update_cost_safety()
    assert res.safety == 0.0

    # Push beyond nameplate to trigger safety
    res.action.c = np.array([4.0], dtype=np.float32)  # Q=4
    res.update_state()
    res.update_cost_safety()
    S = math.hypot(res.state.P, res.state.Q)
    assert pytest.approx(res.safety, rel=1e-6) == max(0.0, S - res.sn_mva)


def test_res_reset_zeros_state():
    res = RES(name="r2", bus=1, sn_mva=5.0, source="wind", max_q_mvar=1.0, min_q_mvar=-1.0)
    res.action.c = np.array([0.25], dtype=np.float32)
    res.update_state(scaling=0.5)
    assert res.state.P != 0.0
    res.reset()
    assert res.state.P == 0.0
    if res.action.c.size > 0:
        assert res.state.Q == 0.0
    assert res.cost == 0.0
    assert res.safety == 0.0
