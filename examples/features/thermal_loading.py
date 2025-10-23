import json
import numpy as np
import pytest

from powergrid.core.features import ThermalLoading
from powergrid.core.state import PhaseModel, PhaseSpec


def _assert_vec_names_consistent(b: ThermalLoading):
    v = b.vector().ravel()
    n = b.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"len(vector)={len(v)} vs len(names)={len(n)}"


# ----------------- GOOD: BALANCED -----------------

def test_balanced_scalar_ok():
    b = ThermalLoading(
        phase_model=PhaseModel.BALANCED_1PH,
        loading_percentage=87.5,
    )
    _assert_vec_names_consistent(b)
    assert np.allclose(b.vector(), [0.875])
    assert b.names() == ["loading_frac"]


# ----------------- BAD: BALANCED -----------------

def test_balanced_requires_scalar():
    with pytest.raises(ValueError, match="requires 'loading_percentage'"):
        ThermalLoading(phase_model=PhaseModel.BALANCED_1PH)

def test_balanced_forbids_per_phase():
    with pytest.raises(ValueError, match="forbids per-phase"):
        ThermalLoading(
            phase_model=PhaseModel.BALANCED_1PH,
            loading_percentage_ph=[90.0, 95.0, 100.0],
        )


# ----------------- GOOD: THREE-PHASE (subset specs) -----------------

def test_three_phase_full_spec_ok():
    b = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        loading_percentage_ph=[80.0, 90.0, 100.0],
    )
    _assert_vec_names_consistent(b)
    assert np.allclose(b.vector(), [0.8, 0.9, 1.0])
    assert set(b.names()) == {"loading_frac_A","loading_frac_B","loading_frac_C"}

def test_three_phase_two_phase_spec_ok():
    b = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        loading_percentage_ph=[55.0, 65.0],
    )
    _assert_vec_names_consistent(b)
    assert np.allclose(b.vector(), [0.55, 0.65])
    assert set(b.names()) == {"loading_frac_B","loading_frac_C"}

def test_three_phase_one_phase_spec_ok():
    b = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("A"),
        loading_percentage_ph=[120.0],
    )
    _assert_vec_names_consistent(b)
    assert np.allclose(b.vector(), [1.2])
    assert b.names() == ["loading_frac_A"]


# ----------------- BAD: THREE-PHASE -----------------

def test_three_phase_requires_per_phase_array():
    with pytest.raises(ValueError, match="requires 'loading_percentage_ph'"):
        ThermalLoading(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
        )

def test_three_phase_forbids_scalar():
    with pytest.raises(ValueError, match="forbids scalar"):
        ThermalLoading(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            loading_percentage=50.0,
        )

def test_three_phase_shape_mismatch():
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        ThermalLoading(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("BC"),
            loading_percentage_ph=[80.0, 90.0, 100.0],  # too long
        )


# ----------------- clamp_ behavior -----------------

def test_clamp_balanced_and_three_phase():
    b1 = ThermalLoading(
        phase_model=PhaseModel.BALANCED_1PH,
        loading_percentage=250.0,  # will clamp to 200
    )
    b1.clamp_()
    assert b1.loading_percentage == 200.0
    assert np.allclose(b1.vector(), [2.0])

    b2 = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        loading_percentage_ph=[-10.0, 50.0, 250.0],  # clamp to 0..200
    )
    b2.clamp_()
    assert np.allclose(b2.loading_percentage_ph, [0.0, 50.0, 200.0])
    assert np.allclose(b2.vector(), [0.0, 0.5, 2.0])


# ----------------- round-trip serialization -----------------

def test_roundtrip_balanced():
    b0 = ThermalLoading(
        phase_model=PhaseModel.BALANCED_1PH,
        loading_percentage=73.0,
    )
    d = b0.to_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    b1 = ThermalLoading.from_dict(d2)
    _assert_vec_names_consistent(b1)
    assert np.allclose(b0.vector(), b1.vector())
    assert b1.phase_spec is None

def test_roundtrip_three_phase_subset():
    b0 = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        loading_percentage_ph=[88.0, 92.0],
    )
    d = b0.to_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    b1 = ThermalLoading.from_dict(d2)
    _assert_vec_names_consistent(b1)
    assert b1.phase_spec.nph() == 2
    assert np.allclose(b0.vector(), b1.vector())


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
