# tests/test_storage_ess.py
import math
import numpy as np
import pytest

# Adjust import path if needed:
from powergrid.devices.storage import ESS


def test_ess_action_space_dims_without_q():
    ess = ESS(
        name="b0", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, sn_mva=np.nan, max_q_mvar=np.nan
    )
    assert ess.action.dim_c == 1
    assert ess.action.c.size == 1
    assert ess.action.range.shape == (2, 1)


def test_ess_action_space_dims_with_q():
    # With either sn_mva or max_q_mvar defined -> (P, Q)
    ess = ESS(
        name="b1", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, sn_mva=10.0
    )
    assert ess.action.dim_c == 2
    assert ess.action.c.size == 2
    assert ess.action.range.shape == (2, 2)
    # sn_mva implies q limits derived from capability
    assert not np.isnan(ess.min_q_mvar)
    assert not np.isnan(ess.max_q_mvar)


def test_ess_soc_dynamics_charge_then_discharge():
    ess = ESS(
        name="b2", bus=1,
        min_p_mw=-10.0, max_p_mw=10.0,
        capacity=100.0, init_soc=0.5, ch_eff=0.98, dsc_eff=0.98, dt=1.0
    )
    # Charge at +10 MW for 1h
    ess.action.c = np.array([10.0], dtype=np.float32)
    ess.update_state()
    expected_soc_after_charge = 0.5 + 10.0 * 0.98 * 1.0 / 100.0
    assert pytest.approx(ess.state.soc, rel=1e-6) == expected_soc_after_charge
    # Discharge at -10 MW for 1h
    ess.action.c = np.array([-10.0], dtype=np.float32)
    ess.update_state()
    expected_soc_after_discharge = expected_soc_after_charge + (-10.0) / 0.98 * 1.0 / 100.0
    assert pytest.approx(ess.state.soc, rel=1e-6) == expected_soc_after_discharge


def test_ess_cost_quadratic_absP_term():
    ess = ESS(
        name="b3", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, init_soc=0.5,
        dt=2.0,  # check dt scaling
        # cost = a*|P| + b*P + c
        # choose a=0.1, b=0.0, c=0.0
        # at P=-5 => 0.1*5 = 0.5; cost = 0.5 * dt = 1.0
        cost_curve_coefs=(0.1, 0.0, 0.0),
    )
    ess.action.c = np.array([-5.0], dtype=np.float32)
    ess.update_state()
    ess.update_cost_safety()
    assert pytest.approx(ess.cost, rel=1e-6) == 1.0


def test_ess_safety_soc_bounds_and_overrating():
    # Tight SOC window to trigger bound penalties, and small sn_mva to trigger S-over-rating
    ess = ESS(
        name="b4", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, min_e_mwh=10.0, max_e_mwh=90.0, init_soc=0.5,
        sn_mva=10.0, dt=1.0
    )
    # Set P,Q to exceed sn_mva
    ess.action.c = np.array([8.0, 8.0], dtype=np.float32)  # S ≈ 11.314
    ess.update_state()
    # Push SOC above max to trigger SOC penalty
    ess.state.soc = ess.max_soc + 0.05  # 0.9 + 0.05 = 0.95
    ess.update_cost_safety()

    S = math.hypot(ess.state.P, ess.state.Q)
    s_over = max(0.0, S - ess.sn_mva) / ess.sn_mva  # normalized by nameplate
    soc_pen = (ess.state.soc - ess.max_soc)  # 0.05
    expected = (s_over + soc_pen) * ess.dt
    assert pytest.approx(ess.safety, rel=1e-6) == expected


def test_ess_feasible_action_clamps_power_by_soc_window():
    # min/max state of charge imply instantaneous feasible P bounds
    ess = ESS(
        name="b5", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, min_e_mwh=10.0, max_e_mwh=90.0, init_soc=0.1,  # at lower bound
        ch_eff=0.98, dsc_eff=0.98, dt=1.0
    )
    # Try to discharge when at min SOC -> should clamp up to >= 0
    ess.action.c = np.array([-5.0], dtype=np.float32)
    ess.feasible_action()
    assert ess.action.c[0] >= 0.0  # cannot go negative

    # Move to max SOC and try to charge -> should clamp to <= 0
    ess.state.soc = ess.max_soc  # 0.9
    ess.action.c = np.array([5.0], dtype=np.float32)
    ess.feasible_action()
    assert ess.action.c[0] <= 0.0  # cannot go positive


def test_ess_feasible_action_clips_q_when_available():
    ess = ESS(
        name="b6", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, init_soc=0.5,
        min_q_mvar=-1.0, max_q_mvar=1.0,  # enables Q dimension
        sn_mva=np.nan,  # leave NaN; Q range alone still triggers 2D action
        dt=1.0,
    )
    # set a P, and an out-of-range Q which should be clipped to [-1,1]
    ess.action.c = np.array([2.0, 3.5], dtype=np.float32)
    ess.feasible_action()
    assert ess.action.c.shape == (2,)
    assert ess.action.c[1] == pytest.approx(1.0, rel=1e-6)

    ess.action.c = np.array([2.0, -5.0], dtype=np.float32)
    ess.feasible_action()
    assert ess.action.c[1] == pytest.approx(-1.0, rel=1e-6)


def test_ess_reset_sets_soc_and_zeros_power():
    ess = ESS(
        name="b7", bus=1,
        min_p_mw=-5.0, max_p_mw=5.0,
        capacity=100.0, min_e_mwh=10.0, max_e_mwh=90.0, init_soc=0.5,
        min_q_mvar=np.nan, max_q_mvar=np.nan
    )
    # Explicit init_soc
    ess.reset(init_soc=0.7)
    assert ess.state.soc == 0.7
    assert ess.state.P == 0.0
    # Random init -> within [min_soc, max_soc]
    ess.reset()
    assert ess.min_soc <= ess.state.soc <= ess.max_soc
    assert ess.state.P == 0.0
