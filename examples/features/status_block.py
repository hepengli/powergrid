import numpy as np
import pytest

from powergrid.core.features import StatusBlock


def _assert_vec_names_consistent(x: StatusBlock):
    v = x.vector().ravel()
    n = x.names()
    assert v.ndim == 1
    assert len(v) == len(n), f"{len(v)} vs {len(n)}"


def _get(names, vec, key):
    assert key in names, f"{key} not found in names"
    return float(vec[names.index(key)])


# ---------------- GOOD CASES ----------------

def test_simple_generator_online():
    gen1 = StatusBlock(
        online=True,
        blocked=False,
        state="online",
        states_vocab=["off", "online", "fault"],
    )
    names, vec = gen1.names(), gen1.vector()
    _assert_vec_names_consistent(gen1)
    assert ("online" in names) and ("blocked" in names)
    assert _get(names, vec, "online") == pytest.approx(1.0)
    assert _get(names, vec, "blocked") == pytest.approx(0.0)
    # state encoding is optional unless emit_state_one_hot or emit_state_index set
    # So we only check consistency here.


def test_diesel_with_startup_shutdown_one_hot():
    gen2 = StatusBlock(
        online=None,
        blocked=False,
        state="startup",
        states_vocab=["off", "startup", "online", "shutdown", "fault"],
        t_in_state_s=12.0,
        t_to_next_s=48.0,
        progress_frac=0.2,
        emit_state_one_hot=True,
    )
    names, vec = gen2.names(), gen2.vector()
    _assert_vec_names_consistent(gen2)

    # time/progress fields present
    for k in ("t_in_state_s", "t_to_next_s", "progress_frac"):
        assert k in names

    # one-hot present and valid (STATUSBLOCK uses 'state_<token>' names)
    one_hot_names = [f"state_{s}" for s in gen2.states_vocab]
    for k in one_hot_names:
        assert k in names
    one_hot_vals = np.array([_get(names, vec, k) for k in one_hot_names])
    assert np.isclose(one_hot_vals.sum(), 1.0)
    # index of "startup" is 1
    assert one_hot_vals[1] == pytest.approx(1.0)


def test_ev_lifecycle_one_hot():
    ev = StatusBlock(
        online=True,
        blocked=False,
        state="charging",
        states_vocab=["idle", "waiting", "charging", "driving", "fault"],
        t_in_state_s=3600.0,
        t_to_next_s=1800.0,
        progress_frac=0.67,
        emit_state_one_hot=True,
    )
    names, vec = ev.names(), ev.vector()
    _assert_vec_names_consistent(ev)

    assert _get(names, vec, "online") == pytest.approx(1.0)
    assert _get(names, vec, "blocked") == pytest.approx(0.0)

    # charging should be the one-hot '1'
    idx = ev.states_vocab.index("charging")
    # STATUSBLOCK uses 'state_<token>' for one-hot labels
    one_hot = np.array([_get(names, vec, f"state_{s}") for s in ev.states_vocab])
    assert one_hot[idx] == pytest.approx(1.0)


def test_thermal_storage_progress_and_times():
    ts = StatusBlock(
        online=True,
        blocked=None,
        state="discharging",
        states_vocab=["idle", "charging", "discharging", "full", "empty"],
        progress_frac=0.4,
        t_in_state_s=600.0,
    )
    names, vec = ts.names(), ts.vector()
    _assert_vec_names_consistent(ts)
    assert "progress_frac" in names
    assert "t_in_state_s" in names
    # 'blocked' is None => not emitted
    assert ("blocked" not in names)


def test_pv_fault_blocked_flag_and_times():
    pv = StatusBlock(
        online=False,
        blocked=True,
        state="fault",
        states_vocab=["off", "starting", "online", "fault", "recovering"],
        t_in_state_s=120.0,
        t_to_next_s=30.0,
        progress_frac=0.8,
    )
    names, vec = pv.names(), pv.vector()
    _assert_vec_names_consistent(pv)
    assert _get(names, vec, "online") == pytest.approx(0.0)
    assert _get(names, vec, "blocked") == pytest.approx(1.0)
    for k in ("t_in_state_s", "t_to_next_s", "progress_frac"):
        assert k in names


def test_battery_idle_minimal():
    bat = StatusBlock(
        online=True,
        blocked=False,
        state="idle",
        states_vocab=["idle", "charging", "discharging", "fault"],
        progress_frac=None,
    )
    names, vec = bat.names(), bat.vector()
    _assert_vec_names_consistent(bat)
    # progress omitted
    assert "progress_frac" not in names
    # online/blocked present
    assert _get(names, vec, "online") == pytest.approx(1.0)
    assert _get(names, vec, "blocked") == pytest.approx(0.0)


def test_with_state_index_export():
    b = StatusBlock(
        state="waiting",
        states_vocab=["idle", "waiting", "charging", "driving"],
        emit_state_one_hot=False,
        emit_state_index=True,
    )
    names, vec = b.names(), b.vector()
    _assert_vec_names_consistent(b)
    assert names == ["state_idx"]
    assert vec.size == 1
    assert vec[0] == pytest.approx(b.states_vocab.index("waiting"))


# ---------------- BAD CASES ----------------

def test_invalid_vocab_duplicates_raises():
    with pytest.raises(ValueError):
        _ = StatusBlock(state="on", states_vocab=["on", "on"])  # duplicates


def test_state_not_in_vocab_raises():
    with pytest.raises(ValueError):
        _ = StatusBlock(state="active", states_vocab=["on", "off"])


def test_invalid_progress_fraction_raises():
    with pytest.raises(ValueError):
        _ = StatusBlock(state="on", states_vocab=["on", "off"], progress_frac=1.5)


if __name__ == "__main__":
    import sys
    print(f"Running {__file__} as standalone test module...\n")
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
