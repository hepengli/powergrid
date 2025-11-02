import pytest, json
import numpy as np

from powergrid.core.features import ShuntCapacitorBlock
from powergrid.utils.phase import PhaseModel, PhaseSpec


def _assert_vector_names_consistent(b: ShuntCapacitorBlock):
    v = b.vector()
    n = b.names()
    # Each scalar contributes 1, each array contributes its size.
    # We can only check length equality in a generic way.
    assert isinstance(n, list)
    assert v.ndim == 1
    assert len(v) == len(n), f"len(vector)={len(v)} vs len(names)={len(n)}"


# ------------------------------
# Good (should construct cleanly)
# ------------------------------

def test_good_balanced_scalar_only():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.BALANCED_1PH,
        phase_spec=PhaseSpec("ABC"),  # ignored
        kvar_total=300.0,
    )
    assert b.phase_spec is None
    assert b.kvar_total == 300.0
    assert b.n_stages in (None, 0)


def test_good_balanced_staged_match_sum():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.BALANCED_1PH,
        kvar_total=300.0,
        n_stages=3,
        stage_kvar_total=[100.0, 100.0, 100.0],
    )
    assert b.n_stages == 3
    assert b.stage_kvar_total.shape == (3,)
    assert np.isclose(b.stage_kvar_total.sum(), 300.0)


def test_good_three_phase_per_phase_only():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_ph=[100.0, 150.0, 200.0],
    )
    assert b.phase_spec.nph() == 3
    assert b.kvar_ph.shape == (3,)
    assert np.isclose(b.kvar_ph.sum(), 450.0)


def test_good_three_phase_total_matches_per_phase():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_total=450.0,
        kvar_ph=[100.0, 150.0, 200.0],
    )
    assert np.isclose(b.kvar_total, 450.0)
    assert np.isclose(b.kvar_ph.sum(), 450.0)


def test_good_three_phase_staged_per_phase():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_ph=[120.0, 120.0, 120.0],
        n_stages=2,
        stage_kvar_ph=[
            [60.0, 60.0],
            [60.0, 60.0],
            [60.0, 60.0],
        ],
    )
    assert b.n_stages == 2
    assert b.stage_kvar_ph.shape == (3, 2)
    assert np.allclose(b.stage_kvar_ph.sum(axis=1), [120, 120, 120])


# -----------------------------
# Bad (should raise ValueError)
# -----------------------------

def test_bad_balanced_per_phase_forbidden():
    with pytest.raises(ValueError, match="forbids per-phase"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.BALANCED_1PH,
            kvar_ph=[50.0, 50.0, 50.0],
        )


def test_bad_balanced_staged_missing_vec():
    # current implementation says "Provide 'stage_kvar_total' ..."
    with pytest.raises(ValueError, match=r"Provide .*stage_kvar_total"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.BALANCED_1PH,
            kvar_total=100.0,
            n_stages=2,
            # stage_kvar_total missing
        )


def test_bad_balanced_stage_len_mismatch():
    # current implementation says "'stage_kvar_total' must have length m=3, got 2."
    with pytest.raises(ValueError, match=r"must have length m=3"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.BALANCED_1PH,
            n_stages=3,
            stage_kvar_total=[50.0, 50.0],  # too short
        )


def test_bad_balanced_total_vs_sum_conflict():
    with pytest.raises(ValueError, match="Conflict: kvar_total"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.BALANCED_1PH,
            kvar_total=120.0,
            n_stages=2,
            stage_kvar_total=[50.0, 50.0],  # sums to 100
        )


def test_bad_three_phase_needs_spec():
    with pytest.raises(ValueError, match="requires a PhaseSpec"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=None,
            kvar_total=100.0,
        )


def test_bad_three_phase_spec_must_be_3ph():
    # We now ALLOW 1φ/2φ subset specs in THREE_PHASE (unbalanced modeling).
    # The correct failure here is a shape mismatch with the provided per-phase vector.
    with pytest.raises(ValueError, match=r"'kvar_ph' must have shape \(1,"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("A"),
            kvar_ph=[10.0, 10.0, 10.0],   # wrong length for a 1-phase spec
        )


def test_bad_three_phase_needs_total_or_ph():
    # current message lists all acceptable options explicitly
    with pytest.raises(ValueError, match=r"requires .*kvar_ph.*kvar_total.*stage_kvar_ph"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
        )


def test_bad_three_phase_kvar_ph_shape():
    with pytest.raises(ValueError, match="kvar_ph.*shape"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            kvar_ph=[100.0, 200.0],  # wrong length
        )


def test_bad_three_phase_aggregate_staged_forbidden():
    with pytest.raises(ValueError, match="forbids aggregate staged"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            stage_kvar_total=[50.0, 50.0],
        )


def test_bad_three_phase_total_vs_sum_conflict():
    with pytest.raises(ValueError, match="kvar_total=.*≠.*sum"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            kvar_total=500.0,
            kvar_ph=[100.0, 100.0, 200.0],  # sums to 400
        )


def test_bad_three_phase_staged_missing_matrix():
    # current implementation says "Provide 'stage_kvar_ph' with shape (nph, m) ..."
    with pytest.raises(ValueError, match=r"Provide .*'stage_kvar_ph'.*shape"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            kvar_ph=[120.0, 120.0, 120.0],
            n_stages=2,
            # stage_kvar_ph missing
        )


def test_bad_three_phase_staged_shape_mismatch():
    with pytest.raises(ValueError, match="stage_kvar_ph.*shape"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            kvar_total=240.0,
            n_stages=2,
            stage_kvar_ph=[
                [60.0, 60.0],
                [60.0, 60.0],
                # missing third row
            ],
        )


def test_bad_three_phase_enabled_shape_mismatch():
    with pytest.raises(ValueError, match="stage_enabled_ph.*shape"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            kvar_ph=[120.0, 120.0, 120.0],
            n_stages=2,
            stage_kvar_ph=[
                [60.0, 60.0],
                [60.0, 60.0],
                [60.0, 60.0],
            ],
            stage_enabled_ph=[  # wrong shape (2,2)
                [1, 1],
                [1, 0],
            ],
        )


# -------------------------------
# Balanced one-phase round trips
# -------------------------------

def test_roundtrip_balanced_scalar_only():
    b0 = ShuntCapacitorBlock(
        phase_model=PhaseModel.BALANCED_1PH,
        phase_spec=PhaseSpec("ABC"),  # ignored
        kvar_total=300.0,
    )
    _assert_vector_names_consistent(b0)
    assert b0.phase_spec is None

    d = b0.to_dict()
    json.dumps(d)  # must be JSON-serializable

    b1 = ShuntCapacitorBlock.from_dict(d)
    _assert_vector_names_consistent(b1)

    v0 = b0.vector()
    v1 = b1.vector()
    assert np.allclose(v0, v1)
    assert b1.phase_spec is None
    assert b1.kvar_total == 300.0


def test_roundtrip_balanced_staged():
    b0 = ShuntCapacitorBlock(
        phase_model=PhaseModel.BALANCED_1PH,
        kvar_total=300.0,
        n_stages=3,
        stage_kvar_total=[100.0, 100.0, 100.0],
    )
    _assert_vector_names_consistent(b0)
    assert b0.n_stages == 3

    d = b0.to_dict()
    json.dumps(d)

    b1 = ShuntCapacitorBlock.from_dict(d)
    _assert_vector_names_consistent(b1)

    assert b1.n_stages == 3
    assert np.allclose(b1.stage_kvar_total, [100, 100, 100])
    assert np.isclose(b1.kvar_total, 300.0)

    v0 = b0.vector()
    v1 = b1.vector()
    assert np.allclose(v0, v1)


# -------------------------------
# Three-phase round trips
# -------------------------------

def test_roundtrip_three_phase_per_phase_only():
    b0 = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_ph=[100.0, 150.0, 200.0],
    )
    _assert_vector_names_consistent(b0)
    assert b0.phase_spec.nph() == 3

    d = b0.to_dict()
    json.dumps(d)

    b1 = ShuntCapacitorBlock.from_dict(d)
    _assert_vector_names_consistent(b1)

    assert b1.phase_spec.nph() == 3
    assert np.allclose(b1.kvar_ph, [100, 150, 200])

    v0 = b0.vector()
    v1 = b1.vector()
    assert np.allclose(v0, v1)


def test_roundtrip_three_phase_total_and_per_phase_agree():
    # both scalar total and per-phase allowed if they match
    b0 = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_total=450.0,
        kvar_ph=[100.0, 150.0, 200.0],
    )
    _assert_vector_names_consistent(b0)

    d = b0.to_dict()
    json.dumps(d)

    b1 = ShuntCapacitorBlock.from_dict(d)
    _assert_vector_names_consistent(b1)

    assert np.isclose(b1.kvar_total, 450.0)
    assert np.allclose(b1.kvar_ph, [100, 150, 200])

    v0 = b0.vector()
    v1 = b1.vector()
    assert np.allclose(v0, v1)


def test_roundtrip_three_phase_staged():
    b0 = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        kvar_ph=[120.0, 120.0, 120.0],
        n_stages=2,
        stage_kvar_ph=[
            [60.0, 60.0],
            [60.0, 60.0],
            [60.0, 60.0],
        ],
        # include enabled to exercise serialization of 2D arrays
        stage_enabled_ph=[
            [1, 0],
            [1, 1],
            [0, 1],
        ],
    )
    _assert_vector_names_consistent(b0)
    assert b0.n_stages == 2
    assert b0.stage_kvar_ph.shape == (3, 2)

    d = b0.to_dict()
    json.dumps(d)

    b1 = ShuntCapacitorBlock.from_dict(d)
    _assert_vector_names_consistent(b1)

    assert b1.n_stages == 2
    assert b1.stage_kvar_ph.shape == (3, 2)
    assert np.allclose(b1.stage_kvar_ph.sum(axis=1), [120, 120, 120])
    assert np.array_equal(b1.stage_enabled_ph.shape, (3, 2))

    v0 = b0.vector()
    v1 = b1.vector()
    assert np.allclose(v0, v1)


# -------------------------------
# Edge / inference behavior
# -------------------------------

def test_infer_n_stages_balanced_from_vector():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.BALANCED_1PH,
        stage_kvar_total=[50.0, 75.0, 25.0],
    )
    assert b.n_stages == 3
    _assert_vector_names_consistent(b)


def test_infer_n_stages_three_phase_from_matrix():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        stage_kvar_ph=[
            [10.0, 20.0],
            [10.0, 20.0],
            [10.0, 20.0],
        ],
    )
    assert b.n_stages == 2
    _assert_vector_names_consistent(b)

def test_cap_three_phase_two_phase_spec_ok():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        kvar_ph=[100.0, 150.0],
    )
    assert b.kvar_ph.shape == (2,)
    assert "cap_kvar_B" in b.names()
    assert "cap_kvar_C" in b.names()

def test_cap_three_phase_one_phase_staged_ok():
    b = ShuntCapacitorBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("A"),
        n_stages=2,
        stage_kvar_ph=[[50.0, 25.0]],  # shape (1,2)
    )
    assert b.stage_kvar_ph.shape == (1, 2)
    assert np.allclose(b.kvar_ph, [75.0])

def test_cap_three_phase_subset_shape_mismatch():
    with pytest.raises(ValueError, match=r"shape \(2,"):
        ShuntCapacitorBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("BC"),
            kvar_ph=[100.0, 100.0, 100.0],  # too long for 2-phase
        )


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
