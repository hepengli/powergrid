import json
import numpy as np
import pytest

from powergrid.core.utils.phase import PhaseModel, PhaseSpec
from powergrid.core.features import PhaseConnection


def _assert_vec_names_consistent(x: PhaseConnection):
    v = x.vector().ravel()
    n = x.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"{len(v)} vs {len(n)}"

# ---------- GOOD ----------

def test_three_phase_full_abc_ok():
    pc = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        connection="BC",
    )
    _assert_vec_names_consistent(pc)
    assert pc.names() == ["conn_A", "conn_B", "conn_C"]
    assert np.allclose(pc.vector(), [0.0, 1.0, 1.0])

def test_three_phase_subset_spec_ok():
    pc = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        connection="B",
    )
    _assert_vec_names_consistent(pc)
    assert pc.names() == ["conn_B", "conn_C"]
    assert np.allclose(pc.vector(), [1.0, 0.0])

def test_three_phase_none_connection_ok():
    pc = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        connection=None,
    )
    assert pc.names() == []
    assert np.allclose(pc.vector(), [0.0, 0.0, 0.0])

def test_roundtrip_three_phase():
    pc0 = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        connection="BC",
    )
    d = pc0.to_dict()
    pc1 = PhaseConnection.from_dict(json.loads(json.dumps(d)))
    _assert_vec_names_consistent(pc1)
    assert pc1.phase_spec.nph() == 2
    assert pc1.connection == "BC"

def test_balanced_presence_only_vector_and_names_empty_when_none():
    pc = PhaseConnection(
        phase_model=PhaseModel.BALANCED_1PH,
        connection=None
    )
    assert pc.names() == []
    assert pc.vector().size == 0  # empty vector

def test_three_phase_subset_ok_and_aligns():
    pc = PhaseConnection(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("BC"),
        connection="B",
    )
    assert pc.names() == ["conn_B", "conn_C"]
    assert np.allclose(pc.vector(), [1.0, 0.0])

# ---------- BAD ----------

def test_unknown_connection_rejected():
    with pytest.raises(ValueError, match="Unknown connection"):
        PhaseConnection(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC"),
            connection="BD",  # invalid token
        )

def test_conn_not_subset_of_spec_rejected():
    with pytest.raises(ValueError, match="not subset"):
        PhaseConnection(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("BC"),
            connection="ABC",  # includes A but spec lacks A
        )

def test_missing_spec_in_three_phase_rejected():
    with pytest.raises(ValueError, match="requires a PhaseSpec"):
        PhaseConnection(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=None,
            connection="A",
        )

def test_unsupported_phase_model_rejected():
    with pytest.raises(ValueError, match="Unsupported phase model"):
        PhaseConnection(phase_model="weird", connection="A")  # type: ignore


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    # You can pass pytest args like -q, -v, etc.
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))