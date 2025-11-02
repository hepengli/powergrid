import json
import numpy as np
import pytest

from powergrid.core.features import StorageBlock
from powergrid.core.state import PhaseModel, PhaseSpec


def _assert_vec_names_consistent(b: StorageBlock):
    v = b.vector().ravel()
    n = b.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"len(vector)={len(v)} vs len(names)={len(n)}"


# ----------------- GOOD EXAMPLES -----------------

def test_good_minimal_soc_only():
    b = StorageBlock(soc=0.5)
    _assert_vec_names_consistent(b)
    assert "soc_frac" in b.names()


def test_good_capacity_and_power_limits():
    b = StorageBlock(
        soc=0.6,
        e_capacity_MWh=8.0,
        p_ch_max_MW=2.0,
        p_dis_max_MW=3.0,
    )
    _assert_vec_names_consistent(b)
    ns = b.names()
    assert "e_capacity_MWh" in ns and "p_dis_max_MW" in ns


def test_good_with_derived_features():
    b = StorageBlock(
        soc=0.4,
        soc_min=0.1,
        soc_max=0.9,
        e_capacity_MWh=10.0,
        p_ch_max_MW=5.0,
        p_dis_max_MW=5.0,
        eta_ch=0.9,
        eta_dis=0.95,
        include_derived=True,
    )
    names = b.names()
    v = b.vector()
    for nm in ["headroom_up_MWh", "headroom_down_MWh", "ttf_h", "tte_h"]:
        assert nm in names
    _assert_vec_names_consistent(b)
    assert len(v) == len(names)


def test_good_three_phase_context_subset_spec():
    b = StorageBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        e_capacity_MWh=12.0,
    )
    _assert_vec_names_consistent(b)
    assert b.phase_spec.nph() == 2


def test_good_reserves_in_order():
    b = StorageBlock(
        soc=0.5,
        reserve_min_frac=0.1,
        reserve_max_frac=0.3,
    )
    _assert_vec_names_consistent(b)
    assert "reserve_min_frac" in b.names()
    assert "reserve_max_frac" in b.names()


def test_good_roundtrip_serialization():
    b0 = StorageBlock(
        soc=0.5,
        e_capacity_MWh=10.0,
        include_derived=True,
    )
    d = b0.to_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    b1 = StorageBlock.from_dict(d2)
    _assert_vec_names_consistent(b1)
    # Names may include derived depending on feasibility; lengths match.
    assert len(b0.names()) == len(b1.names())
    assert np.allclose(b0.vector(), b1.vector())


# ----------------- BAD EXAMPLES -----------------

def test_bad_empty_block_rejected():
    with pytest.raises(ValueError, match="requires at least one"):
        StorageBlock()


def test_bad_reserve_ordering_rejected():
    with pytest.raises(ValueError, match="reserve_max_frac"):
        StorageBlock(
            soc=0.4,
            reserve_min_frac=0.8,
            reserve_max_frac=0.2,
        )


def test_bad_three_phase_missing_spec():
    with pytest.raises(ValueError, match="requires a PhaseSpec"):
        StorageBlock(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=None,
            soc=0.4,
        )


def test_bad_unsupported_phase_model():
    with pytest.raises(ValueError, match="Unsupported phase model"):
        StorageBlock(phase_model="WEIRD_PHASE", soc=0.4)  # type: ignore


def test_bad_all_fields_none_same_as_empty():
    with pytest.raises(ValueError, match="requires at least one"):
        StorageBlock(
            soc=None,
            e_capacity_MWh=None,
            p_ch_max_MW=None,
        )


# ----------------- CLAMP & BOUNDS -----------------

def test_clamp_fraction_and_nonnegatives():
    b = StorageBlock(
        soc=-0.2,
        soc_min=0.7,
        soc_max=0.6,           # reversed; clamp_ swaps
        e_capacity_MWh=-5.0,   # -> 0.0
        p_ch_max_MW=-2.0,      # -> 0.0
        degradation_frac=1.5,  # -> 1.0
    )
    # __post_init__ already calls clamp_(), but call again to be explicit
    b.clamp_()
    assert 0.0 <= b.soc <= 1.0
    assert b.soc_min <= b.soc_max
    assert b.e_capacity_MWh == 0.0
    assert b.p_ch_max_MW == 0.0
    assert 0.0 <= b.degradation_frac <= 1.0


def test_clamp_reserve_ordering_swap():
    b = StorageBlock(
        soc=0.55,
        reserve_min_frac=0.8,
        reserve_max_frac=0.9,
    )
    # valid as-is; ensure clamp does not break
    b.clamp_()
    assert b.reserve_min_frac <= b.reserve_max_frac


def test_vector_names_alignment_minimal():
    b = StorageBlock(soc=0.33)
    _assert_vec_names_consistent(b)
    assert b.names() == ["soc_frac"]


def test_vector_names_with_many_fields():
    b = StorageBlock(
        soc=0.25, soc_min=0.1, soc_max=0.9,
        e_capacity_MWh=8.0,
        p_ch_max_MW=2.0, p_dis_max_MW=3.0,
        eta_ch=0.9, eta_dis=0.95, soh_frac=0.8,
        reserve_min_frac=0.2, reserve_max_frac=0.9,
        cycle_throughput_MWh=12.0, degradation_frac=0.1,
        include_derived=True,
    )
    names = b.names()
    v = b.vector()
    _assert_vec_names_consistent(b)
    assert "soc_frac" in names and "tte_h" in names
    assert len(v) == len(names)


def test_three_phase_settings():
    # Single-phase connection on a 3φ bus (phase B), even split (only one phase)
    sb1 = StorageBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("B"),
        e_capacity_MWh=10.0, p_ch_max_MW=3.0, p_dis_max_MW=3.0, soc=0.5,
    )
    assert sb1.connected_phases() == "B"
    assert sb1.get_phase_allocation().tolist() == [1.0]

    # Two-phase connection (BC) with explicit 70/30 split
    sb2 = StorageBlock(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        alloc_frac_ph=[0.7, 0.3],
        e_capacity_MWh=12.0, p_ch_max_MW=6.0, p_dis_max_MW=6.0, soc=0.4,
    )
    assert sb2.connected_phases() == "BC"
    alloc = sb2.get_phase_allocation()
    assert np.isclose(np.sum(alloc), 1.0)  # sums to 1.0 after normalization
    # allocation matches provided split after normalization
    np.testing.assert_allclose(alloc, np.array([0.7, 0.3], dtype=np.float32))


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
