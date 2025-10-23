import json
import numpy as np
import pytest

from powergrid.core.features import ElectricalBasePh
from powergrid.core.state import PhaseModel, PhaseSpec


def _assert_vec_names_consistent(b: ElectricalBasePh):
    v = b.vector().ravel()
    n = b.names()
    assert isinstance(n, list)
    assert v.ndim == 1
    assert len(v) == len(n), f"{len(v)} vs {len(n)}"


# ------------- GOOD: BALANCED 1φ -------------

def test_balanced_scalar_only():
    b = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=1.2, V_pu=0.98,
    )
    assert b.phase_spec is None
    _assert_vec_names_consistent(b)


def test_balanced_any_one_scalar_is_ok():
    b = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        theta_rad=0.1,
    )
    _assert_vec_names_consistent(b)


# ------------- BAD: BALANCED 1φ -------------

def test_balanced_requires_at_least_one_scalar():
    with pytest.raises(ValueError, match="requires at least one"):
        ElectricalBasePh(phase_model=PhaseModel.BALANCED_1PH)


def test_balanced_forbids_per_phase():
    with pytest.raises(ValueError, match="forbids per-phase"):
        ElectricalBasePh(
            phase_model=PhaseModel.BALANCED_1PH,
            P_MW_ph=[1.0, 1.0, 1.0],
        )


def test_balanced_neutral_forbidden():
    with pytest.raises(ValueError, match="Neutral telemetry not allowed"):
        ElectricalBasePh(
            phase_model=PhaseModel.BALANCED_1PH,
            P_MW=0.5,
            I_neutral_A=10.0,
        )


# ------------- GOOD: THREE-PHASE -------------

def test_three_phase_per_phase_ok():
    b = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        P_MW_ph=[0.5, 0.6, 0.7],
    )
    assert b.phase_spec.nph() == 3
    assert b.P_MW_ph.shape == (3,)
    _assert_vec_names_consistent(b)


def test_three_phase_neutral_allowed_when_spec_says_so():
    b = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC", has_neutral=True, earth_bond=True),
        V_pu_ph=[1.0, 0.99, 1.01],
        I_neutral_A=5.0,
        Vn_earth_V=2.0,
    )
    _assert_vec_names_consistent(b)
    assert b.I_neutral_A == 5.0
    assert b.Vn_earth_V == 2.0


# ------------- BAD: THREE-PHASE -------------

def test_three_phase_requires_spec_and_at_least_one_array():
    with pytest.raises(ValueError, match="requires a PhaseSpec"):
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=None,
            P_MW_ph=[0.5, 0.6, 0.7],
        )
    with pytest.raises(ValueError, match="requires at least one per-phase"):
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
        )


def test_three_phase_forbids_scalars():
    with pytest.raises(ValueError, match="forbids scalar fields"):
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            P_MW=1.0,
        )


def test_three_phase_array_shapes_checked():
    with pytest.raises(ValueError, match="shape"):
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            Q_MVAr_ph=[0.1, 0.2],  # wrong length
        )


def test_three_phase_neutral_needs_has_neutral_true():
    with pytest.raises(ValueError, match="Neutral telemetry requires"):
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC", has_neutral=False, earth_bond=True),
            V_pu_ph=[1.0, 1.0, 1.0],
            I_neutral_A=1.0,
        )


# ------------- ROUND-TRIP (to_dict/from_dict) -------------

def test_roundtrip_balanced():
    b0 = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=1.2, Q_MVAr=0.3, V_pu=0.99, theta_rad=0.05,
    )
    _assert_vec_names_consistent(b0)

    d = b0.to_dict()
    s = json.dumps(d)          # JSON-safe
    d2 = json.loads(s)
    b1 = ElectricalBasePh.from_dict(d2)

    _assert_vec_names_consistent(b1)
    assert b1.phase_spec is None
    assert np.allclose(b0.vector(), b1.vector())


def test_roundtrip_three_phase():
    b0 = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        P_MW_ph=[0.5, 0.6, 0.7],
        Q_MVAr_ph=[0.1, 0.2, 0.3],
        V_pu_ph=[1.0, 0.99, 1.01],
        theta_rad_ph=[0.0, -0.05, 0.03],
    )
    _assert_vec_names_consistent(b0)

    d = b0.to_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    b1 = ElectricalBasePh.from_dict(d2)

    _assert_vec_names_consistent(b1)
    assert b1.phase_spec.nph() == 3
    assert np.allclose(b0.vector(), b1.vector())


# ------------- names() coverage -------------

def test_names_balanced_fields_present_only():
    b = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=1.0, V_pu=0.99,
    )
    names = b.names()
    assert "P_MW" in names
    assert "V_pu" in names
    assert "Q_MVAr" not in names
    assert "theta_rad" not in names


def test_names_three_phase_per_phase_only():
    b = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        V_pu_ph=[1.0, 1.0, 1.0],
    )
    names = b.names()
    assert "V_pu_A" in names and "V_pu_B" in names and "V_pu_C" in names
    assert "P_MW" not in names and "theta_rad" not in names


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    # You can pass pytest args like -q, -v, etc.
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))