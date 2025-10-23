import json
import numpy as np
import pytest

from powergrid.core.features import TapChangerPh
from powergrid.core.utils.phase import PhaseModel, PhaseSpec


def _assert_vec_names_consistent(b: TapChangerPh):
    v = b.vector().ravel()
    n = b.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"{len(v)} vs {len(n)}"


# --------------- GOOD: BALANCED ----------------

def test_balanced_onehot_ok():
    b = TapChangerPh(
        phase_model=PhaseModel.BALANCED_1PH,
        tap_min=-8, 
        tap_max=8,
        tap_position=0,
        one_hot=True,
    )
    _assert_vec_names_consistent(b)
    # 17 positions → one-hot length 17
    assert b.vector().size == 17
    assert b.names()[0] == "tap_-8"
    assert b.names()[-1] == "tap_8"


def test_balanced_scalar_norm_ok():
    b = TapChangerPh(
        phase_model=PhaseModel.BALANCED_1PH,
        tap_min=0,
        tap_max=10,
        tap_position=5,
        one_hot=False,
    )
    _assert_vec_names_consistent(b)
    # Implementation uses (pos - min) / (max - min) => 5/10 = 0.5
    assert np.allclose(b.vector(), [0.5])


# --------------- GOOD: THREE-PHASE (subset) ----

def test_three_phase_full_spec_onehot_ok():
    b = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        tap_min=-3, 
        tap_max=3,
        tap_pos_ph=[-1, 0, 2],
        one_hot=True,
    )
    _assert_vec_names_consistent(b)
    # 7 steps per phase → 3 * 7 outputs
    assert b.vector().size == 21
    # name set includes tap_A_*, tap_B_*, tap_C_*
    assert "tap_A_-3" in b.names()
    assert "tap_C_3" in b.names()


def test_three_phase_two_phase_norm_ok():
    b = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        tap_min=0,
        tap_max=4,
        tap_pos_ph=[1, 4],
        one_hot=False,  # <- normalized per-phase (2 scalars)
    )
    v = b.vector()
    n = b.names()
    _assert_vec_names_consistent(b)
    assert v.size == 2
    assert set(n) == {"tap_B_pos_norm", "tap_C_pos_norm"}
    assert np.all((v >= 0.0) & (v <= 1.0))


# --------------- BAD: RANGE / SHAPES ------------

def test_missing_tap_range_rejected():
    with pytest.raises(ValueError, match="Provide 'tap_min' and 'tap_max'"):
        TapChangerPh(
            phase_model=PhaseModel.BALANCED_1PH,
            tap_position=0,
        )


def test_inverted_range_rejected():
    with pytest.raises(ValueError, match="must be ≥ 'tap_min'"):
        TapChangerPh(
            phase_model=PhaseModel.BALANCED_1PH,
            tap_min=5, 
            tap_max=2,
            tap_position=3,
        )


def test_zero_steps_rejected():
    # Inverted range triggers "'tap_max' must be ≥ 'tap_min'." in current impl
    with pytest.raises(ValueError, match=r"must be ≥ 'tap_min'"):
        TapChangerPh(
            phase_model=PhaseModel.BALANCED_1PH,
            tap_min=2,
            tap_max=1,
            tap_position=2,
        )


# --------------- BAD: MODEL / FIELD MISMATCH ----

def test_balanced_forbids_per_phase():
    with pytest.raises(ValueError, match="forbids 'tap_pos_ph'"):
        TapChangerPh(
            phase_model=PhaseModel.BALANCED_1PH,
            tap_min=-2, 
            tap_max=2,
            tap_position=0,
            tap_pos_ph=[0, 0, 0],
        )


def test_balanced_requires_scalar():
    with pytest.raises(ValueError, match="requires 'tap_position'"):
        TapChangerPh(
            phase_model=PhaseModel.BALANCED_1PH,
            tap_min=0, 
            tap_max=2,
        )


def test_three_phase_requires_per_phase():
    with pytest.raises(ValueError, match="requires 'tap_pos_ph'"):
        TapChangerPh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            tap_min=-1, 
            tap_max=1,
        )


def test_three_phase_forbids_scalar():
    with pytest.raises(ValueError, match="forbids 'tap_position'"):
        TapChangerPh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            tap_min=-1, 
            tap_max=1,
            tap_position=0,
            tap_pos_ph=[0, 0, 0],
        )


def test_three_phase_shape_mismatch():
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        TapChangerPh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("BC"),
            tap_min=0, 
            tap_max=4,
            tap_pos_ph=[1, 2, 3],  # too long for 2-phase spec
        )


# --------------- clamp_ behavior -----------------

def test_clamp_bounds_and_positions():
    b = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("AB"),
        tap_min=0, 
        tap_max=4,
        tap_pos_ph=[-3, 9],  # will be clamped to [0, 4]
    )
    b.clamp_()
    assert np.all(b.tap_pos_ph == np.array([0, 4], dtype=np.int32))


def test_clamp_balanced_position():
    b = TapChangerPh(
        phase_model=PhaseModel.BALANCED_1PH,
        tap_min=-2, 
        tap_max=2,
        tap_position=7,
    )
    b.clamp_()
    assert b.tap_position == 2


# --------------- round-trip I/O -------------------

def test_roundtrip_balanced():
    b0 = TapChangerPh(
        phase_model=PhaseModel.BALANCED_1PH,
        tap_min=-3, 
        tap_max=3,
        tap_position=1,
        one_hot=True,
    )
    d = b0.to_dict()
    s = json.dumps(d)
    b1 = TapChangerPh.from_dict(json.loads(s))
    _assert_vec_names_consistent(b1)
    assert np.allclose(b0.vector(), b1.vector())
    assert b1.phase_spec is None


def test_roundtrip_three_phase_subset():
    b0 = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        tap_min=0, 
        tap_max=2,
        tap_pos_ph=[0, 2],
        one_hot=False,
    )
    d = b0.to_dict()
    s = json.dumps(d)
    b1 = TapChangerPh.from_dict(json.loads(s))
    _assert_vec_names_consistent(b1)
    assert b1.phase_spec.nph() == 2
    assert np.allclose(b0.vector(), b1.vector())


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
