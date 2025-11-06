import numpy as np
import pytest

# NOTE: use the canonical import path for these types
from powergrid.utils.phase import PhaseModel, PhaseSpec


def test_default_is_three_phase_with_neutral_and_bond():
    ps = PhaseSpec()
    assert ps.phases == "ABC"
    assert ps.has_neutral is True
    assert ps.earth_bond is True
    assert ps.nph() == 3


def test_no_neutral_forces_no_earth_bond():
    ps = PhaseSpec("ABC", has_neutral=False, earth_bond=True)
    assert ps.has_neutral is False
    # earth_bond must be forced False if there is no neutral
    assert ps.earth_bond is False


def test_sanitizes_phases_case_duplicates_and_order():
    # lower-case, scrambled, duplicates -> canonical "ABC" ordering of subset
    ps = PhaseSpec("bca")
    assert ps.phases == "ABC"

    ps2 = PhaseSpec("AAB")
    # "AAB" -> subset AB, then canonical order "AB"
    assert ps2.phases == "AB"

    ps3 = PhaseSpec("D")  # non-ABC letters ignored -> fallback to "A"
    assert ps3.phases == "A"
    assert ps3.nph() == 1


def test_index_and_nph():
    ps = PhaseSpec("ABC")
    assert ps.index("B") == 1
    assert ps.nph() == 3


def test_normalized_for_balanced_picks_single_phase():
    ps = PhaseSpec("ABC", has_neutral=True, earth_bond=True)
    ps1 = ps.normalized_for_model(PhaseModel.BALANCED_1PH)
    assert ps1.phases == "A"  # pick first
    assert ps1.has_neutral is True
    assert ps1.earth_bond is True

    # THREE_PHASE leaves subset as-is
    ps2 = PhaseSpec("AC").normalized_for_model(PhaseModel.THREE_PHASE)
    assert ps2.phases == "AC"


def test_index_map_to_and_align_array_simple_reorder_and_fill():
    src = PhaseSpec("AC")
    dst = PhaseSpec("ABC")
    # Expect mapping A->0, C->2
    si, di = src.index_map_to(dst)
    np.testing.assert_array_equal(si, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(di, np.array([0, 2], dtype=np.int32))

    arr = np.array([10.0, 20.0], np.float32)  # values for A,C
    aligned = src.align_array(arr, dst, fill=0.0)
    np.testing.assert_allclose(aligned, np.array([10.0, 0.0, 20.0], np.float32))


def test_align_array_drops_missing_dest_phases():
    src = PhaseSpec("ABC")
    dst = PhaseSpec("AB")
    arr = np.array([1.0, 2.0, 3.0], np.float32)
    aligned = src.align_array(arr, dst, fill=-9.0)
    # C is dropped
    np.testing.assert_allclose(aligned, np.array([1.0, 2.0], np.float32))


def test_align_array_handles_no_overlap_gracefully():
    # (contrived) source has only A, dest only BC -> result is all fill
    src = PhaseSpec("A")
    dst = PhaseSpec("BC")
    arr = np.array([7.0], np.float32)
    aligned = src.align_array(arr, dst, fill=0.0)
    np.testing.assert_allclose(aligned, np.array([0.0, 0.0], np.float32))


def test_align_array_type_and_shape_check():
    src = PhaseSpec("AB")
    dst = PhaseSpec("ABC")
    # wrong length should raise
    with pytest.raises(ValueError):
        src.align_array(np.array([1.0, 2.0, 3.0], np.float32), dst)

    # correct length works and dtype respected
    aligned = src.align_array(
        np.array([5.0, 6.0], np.float32), dst, fill=-1.0, dtype=np.float32
    )
    assert aligned.dtype == np.float32
    np.testing.assert_allclose(aligned, np.array([5.0, 6.0, -1.0], np.float32))


def test_to_dict_from_dict_roundtrip():
    ps0 = PhaseSpec("AB", has_neutral=True, earth_bond=True)
    d = ps0.to_dict()
    ps1 = PhaseSpec.from_dict(d)
    assert ps1.phases == "AB"
    assert ps1.has_neutral is True
    assert ps1.earth_bond is True


def test_three_phase_subset_is_valid_in_unbalanced_context():
    # Using THREE_PHASE with a single-phase spec is a common unbalanced pattern
    ps = PhaseSpec("A", has_neutral=True, earth_bond=True)
    keep = ps.normalized_for_model(PhaseModel.THREE_PHASE)
    assert keep.phases == "A"


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
