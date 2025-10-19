import numpy as np
import pytest

from powergrid.core.state import (
    PhaseModel, 
    PhaseSpec, 
    CollapsePolicy, 
    StatusFlags, 
    DeviceState,
)

from powergrid.core.providers import (
    ElectricalBasePh, 
    TapChangerPh, 
    ThermalLoading,
    StorageBlock,
    PhaseConnection,
    PowerAllocation,
)

@pytest.fixture
def spec_abc():      # 3φ, no neutral
    return PhaseSpec("ABC", has_neutral=False)

@pytest.fixture
def spec_abc_n():   # 3φ, with neutral
    return PhaseSpec("ABC", has_neutral=True)


spec_abc = PhaseSpec("ABC", has_neutral=False)
spec_abc_n = PhaseSpec("ABC", has_neutral=True)


def test_balanced_vector_and_names():
    eb = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=1.2, Q_MVAr=0.1, V_pu=0.99, theta_rad=0.05
    )
    v = eb.vector()
    n = eb.names()

    np.testing.assert_allclose(v, np.array([1.2, 0.1, 0.99, 0.05], np.float32))
    assert n == ["P_MW", "Q_MVAr", "V_pu", "theta_rad"]


def test_three_phase_vector_and_names_no_neutral(spec_abc):
    eb = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_abc,
        P_MW_ph=np.array([1.0, 2.0, 3.0], np.float32),
        Q_MVAr_ph=np.array([0.1, 0.2, 0.3], np.float32),
        V_pu_ph=np.array([0.98, 1.00, 1.02], np.float32),
        theta_rad_ph=np.array([0.0, 0.1, -0.1], np.float32),
    )
    v = eb.vector()
    n = eb.names()

    expected = np.concatenate([
        np.array([1.0, 2.0, 3.0], np.float32),
        np.array([0.1, 0.2, 0.3], np.float32),
        np.array([0.98, 1.00, 1.02], np.float32),
        np.array([0.0,  0.1, -0.1], np.float32),
    ])
    np.testing.assert_allclose(v, expected)
    assert n == [
        "P_MW_A","P_MW_B","P_MW_C",
        "Q_MVAr_A","Q_MVAr_B","Q_MVAr_C",
        "V_pu_A","V_pu_B","V_pu_C",
        "theta_rad_A","theta_rad_B","theta_rad_C",
    ]


def test_three_phase_with_neutral_features(spec_abc_n):
    eb = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_abc_n,
        P_MW_ph=np.array([0.5, 0.5, 0.5], np.float32),
        Q_MVAr_ph=np.array([0.0, 0.0, 0.0], np.float32),
        V_pu_ph=np.array([1.0, 1.0, 1.0], np.float32),
        theta_rad_ph=np.array([0.0, 0.0, 0.0], np.float32),
        I_neutral_A=12.3, Vn_earth_V=1.5
    )
    v = eb.vector()
    n = eb.names()

    assert n[-2:] == ["I_neutral_A", "Vn_earth_V"]
    np.testing.assert_allclose(v[-2:], np.array([12.3, 1.5], np.float32))


def test_expand_balanced_to_three_phase_with_neutral(spec_abc_n):
    eb = ElectricalBasePh(
        phase_model=PhaseModel.BALANCED_1PH,
        P_MW=3.0, Q_MVAr=0.9, V_pu=1.01, theta_rad=0.05
    )
    eb3 = eb.to_phase_model(PhaseModel.THREE_PHASE, spec_abc_n)
    v = eb3.vector()
    # Expect replication across phases; neutral telemetry omitted (None)
    exp = np.array([3.0,3.0,3.0, 0.9,0.9,0.9, 1.01,1.01,1.01, 0.05,0.05,0.05], np.float32)
    np.testing.assert_allclose(v[:12], exp)   # ignore any tail


def test_collapse_three_phase_to_balanced_mean_and_posseq(spec_abc):
    Vmag = np.array([0.98, 1.00, 1.02], np.float32)
    ang  = np.array([0.0, -2*np.pi/3, +2*np.pi/3], np.float32)

    eb3 = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_abc,
        P_MW_ph=np.array([1.0, 2.0, 3.0], np.float32),
        Q_MVAr_ph=np.array([0.1, 0.2, 0.3], np.float32),
        V_pu_ph=Vmag, theta_rad_ph=ang
    )
    # Mean policy
    eb_mean = eb3.to_phase_model(
        PhaseModel.BALANCED_1PH, PhaseSpec("A"), policy=CollapsePolicy.SUM_PQ_MEAN_V
    )
    # Positive-sequence policy
    eb_pos  = eb3.to_phase_model(
        PhaseModel.BALANCED_1PH, PhaseSpec("A"), policy=CollapsePolicy.SUM_PQ_POSSEQ_V
    )

    v_mean = eb_mean.vector()
    v_pos  = eb_pos.vector()

    # P,Q summed
    np.testing.assert_allclose(v_mean[:2], np.array([6.0, 0.6], np.float32))
    np.testing.assert_allclose(v_pos[:2],  np.array([6.0, 0.6], np.float32))

    # Magnitudes near ~1.0; angles should be reasonable (no wrap artifacts)
    assert 0.99 <= v_mean[2] <= 1.01
    assert 0.99 <= v_pos[2]  <= 1.01
    # Angles:
    # For POSSEQ, angle should be small.
    assert abs(v_pos[3]) < 0.11

    # For MEAN policy, the circular mean is ill-defined when resultant≈0,
    # so accept either ~0 or ~π (wrap to (-π, π]).
    theta = (v_mean[3] + np.pi) % (2*np.pi) - np.pi
    assert (abs(theta) < 0.11) or (abs(abs(theta) - np.pi) < 0.11)


def test_collapse_angle_circular_mean_wrap(spec_abc):
    # Angles near wrap-around: 179°, -179°, 180°≈-π
    ang_deg = np.array([179.0, -179.0, 180.0], np.float32)
    ang = np.deg2rad(ang_deg).astype(np.float32)
    Vmag = np.array([1.0, 1.0, 1.0], np.float32)

    eb3 = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE, 
        phase_spec=spec_abc,
        P_MW_ph=np.zeros(3, np.float32),
        Q_MVAr_ph=np.zeros(3, np.float32),
        V_pu_ph=Vmag, 
        theta_rad_ph=ang
    )
    eb = eb3.to_phase_model(PhaseModel.BALANCED_1PH, PhaseSpec("A"))
    _, _, _, theta = eb.vector()
    # bring to principal branch (-π, π]
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    assert abs(abs(theta) - np.pi) < 1e-3


# -------------------- TapChangerPh tests --------------------
def test_tap_balanced_one_hot_names_and_vector():
    tap = TapChangerPh(
        phase_model=PhaseModel.BALANCED_1PH,
        tap_min=-2, tap_max=2, tap_position=0, one_hot=True
    )
    v = tap.vector()
    names = tap.names()
    # steps: -2, -1, 0, +1, +2 → index 2 is hot
    np.testing.assert_array_equal(v, np.array([0, 0, 1, 0, 0], np.float32))
    assert names == ["tap_-2", "tap_-1", "tap_0", "tap_1", "tap_2"]


def test_tap_three_phase_one_hot_concat(spec_abc):
    tap = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_abc,
        tap_min=0, tap_max=2, one_hot=True,
        tap_pos_ph=np.array([0, 1, 2], np.int32)
    )
    v = tap.vector()
    names = tap.names()
    expected = np.concatenate([
        np.array([1, 0, 0], np.float32),  # A at 0
        np.array([0, 1, 0], np.float32),  # B at 1
        np.array([0, 0, 1], np.float32),  # C at 2
    ])
    np.testing.assert_array_equal(v, expected)
    assert names == ["tap_A_0","tap_A_1","tap_A_2","tap_B_0","tap_B_1","tap_B_2","tap_C_0","tap_C_1","tap_C_2"]


def test_tap_clamp_and_phase_conversion(spec_abc):
    tap3 = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE, phase_spec=spec_abc,
        tap_min=-1, tap_max=1, tap_pos_ph=np.array([-3, 0, 5], np.int32)
    )
    tap3.clamp_()
    np.testing.assert_array_equal(tap3.tap_pos_ph, np.array([-1, 0, 1], np.int32))

    # Collapse to balanced → median across phases = 0
    tap1 = tap3.to_phase_model(PhaseModel.BALANCED_1PH, PhaseSpec("A"))
    v = tap1.vector()
    np.testing.assert_array_equal(v, np.array([0, 1, 0], np.float32))  # one-hot at 0


# -------------------- ThermalLoading test --------------------
def test_thermal_aggregate_only():
    tl = ThermalLoading(loading_percentage=80.0)
    v, n = tl.vector(), tl.names()
    np.testing.assert_allclose(v, np.array([0.8], np.float32))
    assert n == ["loading_frac"]

def test_thermal_per_phase_and_collapse():
    tl = ThermalLoading(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=PhaseSpec("ABC"),
        loading_percentage_ph=np.array([80.0, 65.0, 105.0], np.float32),
    )
    v3, n3 = tl.vector(), tl.names()
    np.testing.assert_allclose(v3, np.array([0.8, 0.65, 1.05], np.float32))
    assert n3 == ["loading_frac_A", "loading_frac_B", "loading_frac_C"]

    tl1 = tl.to_phase_model(PhaseModel.BALANCED_1PH, PhaseSpec("A"))
    v1, n1 = tl1.vector(), tl1.names()
    np.testing.assert_allclose(v1, np.array([1.05], np.float32))  # max-of-phases
    assert n1 == ["loading_frac"]

def test_thermal_expand_from_aggregate():
    tl = ThermalLoading(
        phase_model=PhaseModel.BALANCED_1PH, 
        phase_spec=PhaseSpec("A"),
        loading_percentage=90.0
    )
    tl3 = tl.to_phase_model(PhaseModel.THREE_PHASE, PhaseSpec("ABC"))
    v3, n3 = tl3.vector(), tl3.names()
    np.testing.assert_allclose(v3, np.array([0.9, 0.9, 0.9], np.float32))
    assert n3 == ["loading_frac_A", "loading_frac_B", "loading_frac_C"]


# -------------------- StorageBlock --------------------
sb = StorageBlock(
    soc=0.4, soc_min=0.1, soc_max=0.95,
    e_capacity_MWh=2.0, p_ch_max_MW=1.0, p_dis_max_MW=1.5,
    eta_ch=0.96, eta_dis=0.95, soh_frac=0.92,
    include_derived=True
)
vec, names = sb.vector(), sb.names()

# -------------------- DeviceState: concat, names, conversion --------------------
def test_device_state_concat_and_names():
    ds = DeviceState(
        phase_model=PhaseModel.BALANCED_1PH,
        providers=[
            ElectricalBasePh(phase_model=PhaseModel.BALANCED_1PH, P_MW=1.0, V_pu=0.99),
            ThermalLoading(loading_percentage=80.0),
            StatusFlags(online=True, blocked=False),
            TapChangerPh(phase_model=PhaseModel.BALANCED_1PH, tap_min=0, tap_max=1, tap_position=1),
        ]
    )
    vec = ds.vector()
    names = ds.names()
    # [P, V] + [loading 0.8] + [flags 1,0] + [tap one-hot 0,1]
    np.testing.assert_allclose(vec, np.array([1.0, 0.99, 0.8, 1.0, 0.0, 0.0, 1.0], np.float32))
    assert names == ["P_MW","V_pu","loading_frac","online","blocked","tap_0","tap_1"]


def test_device_state_phase_conversion_roundtrip():
    ds1 = DeviceState(
        phase_model=PhaseModel.BALANCED_1PH,
        providers=[
            ElectricalBasePh(phase_model=PhaseModel.BALANCED_1PH, P_MW=2.0, Q_MVAr=0.5, V_pu=1.02, theta_rad=0.0),
            TapChangerPh(phase_model=PhaseModel.BALANCED_1PH, tap_min=-2, tap_max=2, tap_position=1),
        ]
    )
    spec3 = PhaseSpec("ABC", has_neutral=True)
    ds3 = ds1.to_phase_model(PhaseModel.THREE_PHASE, spec3)
    assert ds3.phase_model == PhaseModel.THREE_PHASE
    ds1b = ds3.to_phase_model(PhaseModel.BALANCED_1PH, PhaseSpec("A"))
    assert ds1b.phase_model == PhaseModel.BALANCED_1PH
    assert ds1b.vector().ndim == 1
    assert len(ds1b.names()) == ds1b.vector().shape[0]

# ----------DeviceState with neutral and connection/provider tests --------------------
ds = DeviceState(
    phase_model=PhaseModel.THREE_PHASE,
    phase_spec=PhaseSpec("ABC", has_neutral=True),
    providers=[
        StorageBlock(soc=0.5, e_capacity_MWh=2.0),           # scalar storage state
        PhaseConnection(connection="A"),                     # wired to phase A
        PowerAllocation(weights_ph=np.array([1.0, 0.0, 0.0], np.float32)),  # all power on A
        ElectricalBasePh(                                    # terminal telemetry/setpoints
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC", has_neutral=True),
            P_MW_ph=np.array([0.0, 0.0, 0.0], np.float32),
            Q_MVAr_ph=np.array([0.0, 0.0, 0.0], np.float32),
        ),
    ]
)

# aggregate load (positive = consumption)
P_MW, Q_MVAr = 0.8, 0.15

alloc = np.array([1.0, 0.0, 0.0], np.float32)  # all on B
ds = DeviceState(
    phase_model=PhaseModel.THREE_PHASE,
    phase_spec=PhaseSpec("ABC", has_neutral=True),
    providers=[
        PhaseConnection(connection="A"),
        PowerAllocation(weights_ph=alloc),
        ElectricalBasePh(
            phase_model=PhaseModel.THREE_PHASE,
            phase_spec=PhaseSpec("ABC", has_neutral=True),
            P_MW_ph=alloc * P_MW,
            Q_MVAr_ph=alloc * Q_MVAr,
        ),
    ],
)

# -------------------- Serialization --------------------
def test_serialization_roundtrip_with_neutral():
    spec = PhaseSpec("ABC", has_neutral=True)
    eb3 = ElectricalBasePh(
        phase_model=PhaseModel.THREE_PHASE, 
        phase_spec=spec,
        P_MW_ph=np.array([0.7, 0.8, 0.9], np.float32),
        Q_MVAr_ph=np.array([0.0, 0.1, 0.2], np.float32),
        V_pu_ph=np.array([1.00, 0.99, 1.01], np.float32),
        theta_rad_ph=np.array([0.0, 0.05, -0.05], np.float32),
        I_neutral_A=5.0,
        Vn_earth_V=2.0
    )
    tap = TapChangerPh(
        phase_model=PhaseModel.THREE_PHASE,
        phase_spec=spec, 
        tap_min=0,
        tap_max=2,
        tap_pos_ph=np.array([1, 1, 2], np.int32)
    )
    ds = DeviceState(
        phase_model=PhaseModel.THREE_PHASE, 
        phase_spec=spec,
        providers=[eb3, tap, StatusFlags(online=True)]
    )
    d = ds.to_dict()
    ds2 = DeviceState.from_dict(d)  # default type_map covers these providers
    np.testing.assert_allclose(ds.vector(), ds2.vector())
    assert ds.names() == ds2.names()
