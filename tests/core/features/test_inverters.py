import numpy as np
import pytest

from powergrid.core.utils.phase import PhaseModel, PhaseSpec
from powergrid.core.features import InverterBasedSource


def _val(names, vec, key):
    """Helper to pull a scalar by name from (names, vector)."""
    if key not in names:
        raise KeyError(f"{key} not in names: {names}")
    return float(vec[names.index(key)])


def _has(names, key):
    return key in names


# --------- GOOD CASES ---------

def test_balanced_fixed_q_minimal_ok():
    """Balanced 1φ with fixed Q setpoint; p,q should show up and be correct."""
    inv = InverterBasedSource(
        phase_model=PhaseModel.BALANCED_1PH,
        p_set_MW=3.0,
        q_set_MVAr=-1.0,  # inductive
    )
    names, vec = inv.names(), inv.vector()
    # Must include core instantaneous outputs
    assert _has(names, "p_MW")
    assert _has(names, "q_MVAr")
    assert _val(names, vec, "p_MW") == pytest.approx(3.0, rel=1e-6)
    assert _val(names, vec, "q_MVAr") == pytest.approx(-1.0, rel=1e-6)
    # If a clip flag is present, it should be 0 here (no limits configured)
    if _has(names, "q_clip_flag"):
        assert _val(names, vec, "q_clip_flag") == pytest.approx(0.0, abs=1e-6)


def test_three_phase_subset_even_alloc_ok():
    """3φ with subset phases 'AC', even allocation with expand_phases=True."""
    inv = InverterBasedSource(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("AC"),
        p_set_MW=3.0,
        q_set_MVAr=-1.0,
        expand_phases=True,
    )
    names, vec = inv.names(), inv.vector()
    # Core scalars present
    assert _has(names, "p_MW") and _has(names, "q_MVAr")
    assert _val(names, vec, "p_MW") == pytest.approx(3.0, rel=1e-6)
    assert _val(names, vec, "q_MVAr") == pytest.approx(-1.0, rel=1e-6)
    # Per-phase appended for A and C only
    for ph in "ABC":
        expect_in = ph in "AC"
        assert _has(names, f"p_MW_{ph}") is expect_in
        assert _has(names, f"q_MVAr_{ph}") is expect_in
    # Even split across two phases: 1.5 MW each, -0.5 MVAr each
    assert _val(names, vec, "p_MW_A") == pytest.approx(1.5, rel=1e-6)
    assert _val(names, vec, "p_MW_C") == pytest.approx(1.5, rel=1e-6)
    assert _val(names, vec, "q_MVAr_A") == pytest.approx(-0.5, rel=1e-6)
    assert _val(names, vec, "q_MVAr_C") == pytest.approx(-0.5, rel=1e-6)


def test_qset_with_limits_clips_and_flags():
    """
    3φ fixed-Q where capability/limits force clipping.
    Set small symmetrical Q limits so -1.5 MVAr → clipped to -1.0 MVAr.
    """
    inv = InverterBasedSource(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        s_rated_MVA=5.0,        # plenty to allow the limit-based clip
        p_set_MW=3.0,
        q_set_MVAr=-1.5,
        q_min_MVAr=-1.0,
        q_max_MVAr=+1.0,
        expand_phases=False,
    )
    names, vec = inv.names(), inv.vector()
    assert _val(names, vec, "q_MVAr") == pytest.approx(-1.0, rel=1e-6)
    assert _val(names, vec, "q_clip_flag") == pytest.approx(1.0, abs=1e-6)


def test_pf_control_inductive_alloc_ok():
    """
    3φ with PF control on subset 'BC'.
    Q should be derived from P and PF, negative for inductive, allocated by fraction.
    """
    inv = InverterBasedSource(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        alloc_frac_ph=[0.6, 0.4],
        s_rated_MVA=8.0,
        p_MW=4.5,              # instantaneous real power present
        ctrl_mode="pf_set",
        pf_target=0.95,
        pf_leading=False,      # inductive => negative Q
        q_min_MVAr=-2.0,
        q_max_MVAr=+2.0,
        expand_phases=True,
        include_derived=False,
        all_params=True,
    )
    names, vec = inv.names(), inv.vector()
    # PF → Q magnitude = P * sqrt(1/pf^2 - 1)
    P = 4.5
    pf = 0.95
    q_mag = P * np.sqrt(1.0 / (pf * pf) - 1.0)
    q_expected = -q_mag  # inductive
    # Core check
    assert _val(names, vec, "p_MW") == pytest.approx(P, rel=1e-6)
    assert _val(names, vec, "q_MVAr") == pytest.approx(q_expected, rel=1e-6)
    # Per-phase sum must match aggregate
    q_parts = [
        _val(names, vec, "q_MVAr_B"),
        _val(names, vec, "q_MVAr_C"),
    ]
    assert np.sum(q_parts) == pytest.approx(q_expected, rel=1e-6)
    # Allocation consistency (proportional to alloc_frac_ph)
    p_parts = [
        _val(names, vec, "p_MW_B"),
        _val(names, vec, "p_MW_C"),
    ]
    assert p_parts[0] / np.sum(p_parts) == pytest.approx(0.6, rel=1e-6)
    assert p_parts[1] / np.sum(p_parts) == pytest.approx(0.4, rel=1e-6)


def test_volt_var_interpolates_and_no_clip():
    """
    Volt-VAR with V between (v1,v2): interpolate q_cmd linearly,
    then q_inst follows q_cmd (no clip here).
    """
    inv = InverterBasedSource(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        s_rated_MVA=3.0,
        p_set_MW=2.2,
        ctrl_mode="volt_var",
        vv_v1_pu=0.97, vv_v2_pu=1.03,
        vv_q1_MVAr=+0.5, vv_q2_MVAr=-0.5,
        expand_phases=False,
    )
    names, vec = inv.names(), inv.vector(V_pu_for_vv=1.02)
    # Interpolated q_cmd at 1.02 pu:
    # fraction = (1.02 - 0.97) / (1.03 - 0.97) = 0.833333...
    # q_cmd = +0.5 + (-1.0)*0.833333.. = -0.333333..
    q_cmd = 0.5 + (-1.0) * ((1.02 - 0.97) / (1.03 - 0.97))
    assert _val(names, vec, "q_MVAr") == pytest.approx(q_cmd, rel=1e-6)
    if _has(names, "q_clip_flag"):
        assert _val(names, vec, "q_clip_flag") == pytest.approx(0.0, abs=1e-6)


# --------- BAD CASES (sanity) ---------

def test_three_phase_requires_valid_spec():
    with pytest.raises(ValueError):
        _ = InverterBasedSource(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=None,  # invalid
            p_set_MW=1.0,
            q_set_MVAr=0.1,
        )


def test_three_phase_subset_allocation_len_mismatch():
    with pytest.raises(ValueError):
        _ = InverterBasedSource(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("AC"),
            alloc_frac_ph=[1.0, 0.0, 0.0],  # length 3 vs nph=2
            p_set_MW=1.0,
        )


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    # You can pass pytest args like -q, -v, etc.
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))