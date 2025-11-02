import numpy as np
from microgrid.core import DG, ESS, MainGrid, Network, RES, Substation
from microgrid.read_data import read_data
from pypower.idx_brch import (
    BR_STATUS,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    T_BUS,
    TAP,
)
from pypower.idx_bus import BUS_TYPE, GS, PD, PQ, QD, REF, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QMAX, QMIN, QG, VG


def case_lvmg():
    """Power flow data for BENCHMARK LOW VOLTAGE MICROGRID NETWORK.
    Please see L{caseformat} for details on the case file format.

    Based on data from “A BENCHMARK LOW VOLTAGE MICROGRID NETWORK”.

    @return: Power flow data for BENCHMARK LOW VOLTAGE MICROGRID NETWORK.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.array([
        [1,  3, 0,   0,   0, 0, 1, 1, 0, 20., 1, 1.1, 0.9], # | B0
        [2,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B1
        [3,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P1
        [4,  1, 12,  5.8, 0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B2
        [5,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P2
        [6,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B3
        [7,  1, 60,  29.1,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B4
        [8,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P3
        [9,  1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P4
        [10, 1, 42,  20.3,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B5
        [11, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P5
        [12, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P6
        [13, 1, 12,  5.8, 0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B6
        [14, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P7
        [15, 1, 40,  19.4,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B7
        [16, 1, 70,  52.5,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B8
        [17, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P8
        [18, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P9
        [19, 1, 18,  11.2,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B9
        [20, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B10
        [21, 1, 25,  15.5,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B11
        [22, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P10
        [23, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P11
        [24, 1, 20,  12.4,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B12
        [25, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B13
        [26, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P12
        [27, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P13
        [28, 1, 20,  12.4,0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B14
        [29, 1, 0,   0,   0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | P14
        [30, 1, 15,  9.3, 0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9], # | B15
    ], dtype=np.float64)

    ppc['bus'][:,PD] *= 0.001
    ppc['bus'][:,QD] *= 0.001

    # generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2, \
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.array([
	    [1, 0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], #
	    [6,	0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 60kW/380kWh Battery
        [7, 0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 50kW/62.5KVA DG1
        [15,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 40kW/50kVA DG2
	    [10,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 10kW Wind Turbine
	    [20,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 10kW Wind Turbine
        [10,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 10kW Solar Panel
        [13,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 10kW Solar Panel
        [25,0, 0, 0, 0, 1, 1, 1, 100, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 10kW Solar Panel
    ], dtype=np.float64)

    # branch data
    # I_base = baseMVA / V_LV_base = 100(MVA) / 0.4kV;  Z_base = V_LV_base / I_base = 0.4 * 1000 / (100 000/0.4) = 0.0016(ohms)
    # Transformer: 
    #       Z = R + jX (p.u. in rating)
    #       S_base = rating / 3 = 400kVA / 3 = 133.3333kVA; 
    #       V_LV_LN_base = 0.4kV / sqrt(3);
    #       I_base = S_base / V_LV_LN_base = 577.3503 A;
    #       Z_base_rating = V_LV_LN_base / I_base = 0.4 ohms; Z_real = Z * Z_base_rating = 0.004 + j0.016; 
    #       Z_pu = Z_real / Z_base = (0.004 + j0.016) / 0.0016 = 2.5 + j10;
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.array([
        [1,  2,  2.5,     10    ,  0,  300,  300,  300,  0,  0,  1, -360, 360], # | B0 -> B1
        [2,  3,  6.21250, 1.8156,  0,  200,  200,  200,  0,  0,  1, -360, 360], # | B1 -> P1
        [3,  4,  69.1875, 1.7625,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P1 -> B2
        [3,  5,  6.21250, 1.8156,  0,  150,  150,  150,  0,  0,  1, -360, 360], # | P1 -> P2
        [5,  6,  25.8750, 1.5375,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P2 -> B3
        [5,  7,  48.0281, 7.0875,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P2 -> B4
        [5,  8,  6.21250, 1.8156,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P2 -> P3
        [8,  9,  6.21250, 1.8156,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P3 -> P4
        [9,  10, 16.3313, 1.5187,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P4 -> B5
        [9,  11, 6.21250, 1.8156,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P4 -> P5
        [11, 12, 6.21250, 1.8156,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P5 -> P6
        [12, 13, 9.18750, 1.7625,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P6 -> B6
        [12, 14, 6.21250, 1.8156,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P6 -> P7
        [14, 15, 25.8750, 1.5375,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P7 -> B7
        [2,  16, 33,      8.8750,  0,  150,  150,  150,  0,  0,  1, -360, 360], # | B1 -> B8
        [2,  17, 7.44370, 5.2313,  0,  150,  150,  150,  0,  0,  1, -360, 360], # | B1 -> P8
        [17, 18, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P8 -> P9
        [18, 19, 10.7625, 5.5125,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P9 -> B9
        [19, 20, 10.7625, 5.5125,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | B9 -> B10
        [18, 21, 7.44370, 5.2313,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P9 -> B11
        [18, 22, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P9 -> P10
        [22, 23, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P10 -> P11
        [23, 24, 11.4188, 2.9812,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P11 -> B12
        [24, 25, 11.4188, 2.9812,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | B12 -> B13
        [23, 26, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P11 -> P12
        [26, 27, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P12 -> P13
        [27, 28, 10.7625, 5.5125,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P13 -> B14
        [27, 29, 7.44370, 5.2313,  0,  100,  100,  100,  0,  0,  1, -360, 360], # | P13 -> P14
        [29, 30, 11.4188, 2.9812,  0,  50,   50,   50,   0,  0,  1, -360, 360], # | P14 -> B15
    ], dtype=np.float64)

    ppc["branch"][:,RATE_A] *= 0.01
    ppc["branch"][:,RATE_B] *= 0.01
    ppc["branch"][:,RATE_C] *= 0.01

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = np.array([
        [1, 1]
    ], dtype=np.float64)

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.array([
        [2, 0, 0, 3, 0,   1,     0],
        [2, 0, 0, 3, 0,   0,     0],
        [2, 0, 0, 3, 100, 30.4,  1.3011], #$/MW^2  $/MW  $
        [2, 0, 0, 3, 100, 51.6,  0.4615], #$/MW^2  $/MW  $
        [2, 0, 0, 3, 34., 42.54, 1.1744], # $/MW^2  $/MW  $
        [2, 0, 0, 3, 34., 42.54, 1.1744], # $/MW^2  $/MW  $
        [2, 0, 0, 3, 28., 35.65, 1.5057], # $/MW^2  $/MW  $
        [2, 0, 0, 3, 34., 42.54, 1.1744], # $/MW^2  $/MW  $
        [2, 0, 0, 3, 28., 35.65, 1.5057], # $/MW^2  $/MW  $
    ], dtype=np.float64)

    ppc["pf"] = ppc["bus"][:,2] / (np.sqrt(np.square(ppc["bus"][:,2]) + np.square(ppc["bus"][:,3])) + 1e-10)

    return ppc

class Scenario(object):
    def make_world(self):
        train = True
        world = Network()
        world.train = train
        world.data = read_data(train)
        world.ppc = case_lvmg()
        DG_1 = DG('DG1', bus=7, Pmin=0, Pmax=40, rating=None, ramp_up=40, ramp_down=40, cost_curve_coefs=[0.01, 7.16, 4.615])
        DG_2 = DG('DG2', bus=15, Pmin=0, Pmax=50, rating=None, ramp_up=50, ramp_down=50, cost_curve_coefs=[0.01, 5.04, 11.011])
        ESS_1 = ESS('ESS1', bus=6, capacity=380.0, rating=60, max_discharging_power=60, min_soc=0.2, max_soc=1)
        ESS_2 = ESS('ESS2', bus=25, capacity=500.0, rating=100, max_discharging_power=100, min_soc=0.15, max_soc=1)
        CL_1 = DG('CL1', bus=7, Pmin=-20, Pmax=0, leading=False, cost_curve_coefs=[-10, 20, 0, 70])
        CL_2 = DG('CL2', bus=16, Pmin=-20, Pmax=0, leading=False, cost_curve_coefs=[-10, 100, 0, 120])
        PV_1 = RES('PV1', source='SOLAR', bus=10, share=1/3)
        PV_2 = RES('PV2', source='SOLAR', bus=13, share=1/3)
        PV_3 = RES('PV3', source='SOLAR', bus=25, share=1/3)
        WIND_1 = RES('WIND_1', source='WIND', bus=10, share=1/2)
        WIND_2 = RES('WIND_2', source='WIND', bus=20, share=1/2)
        GRID = MainGrid('GRID', bus=1)
        SUB = Substation('SUB', fbus=1, tbus=2, rating=300)
        world.agents = [GRID, DG_1, DG_2, ESS_1, ESS_2, CL_1, CL_2, PV_1, PV_2, PV_3, WIND_1, WIND_2, SUB]
        # make initial conditions
        self.reset_world(world, np.random)
        # define communication topology
        num_agents = len(world.policy_agents)
        world.comm_matrix = toeplitz(
            [1]+[0]*(num_agents-2), 
            [1,-1]+[0]*(num_agents-2)
        ).astype(np.float32)
        world.comm_matrix = np.vstack([
            world.comm_matrix,
            np.array([[-1]+[0]*(num_agents-2)+[1]]),
        ]).astype(np.float32)
        # shared rewards
        world.collaborative = False

        return world

    def reset_world(self, world, np_random):
        world.reset()

    def benchmark_data(self, agent, world):
        pass

    def reward(self, agent, world):
        reward, safety = 0, 0
        for agent in world.agents:
            reward -= agent.cost
            safety += agent.safety

            assert agent.cost is not np.nan
            assert agent.safety is not np.nan

        safety += world.undervoltage.sum() * 1e3
        safety += world.overvoltage.sum() * 1e3

        return reward - 0.1 * safety

    def observation(self, agent, world):
        internal_state = []
        for agent in world.dg_agents:
            # internal_state.append(np.arange(2)==agent.state.uc)
            # internal_state.append(np.arange(agent.startup_time+1)==agent.starting)
            # internal_state.append(np.arange(agent.shutdown_time+1)==agent.shutting)
            internal_state.append(np.array([agent.state.P])*1e-3)
        for agent in world.ess_agents:
            internal_state.append(np.array([agent.state.soc]))
        internal_state = np.hstack(internal_state)

        netload = np.sum(world.his['load'], axis=1) - np.sum(world.his['res'], axis=1)
        price = np.array(world.his['price'])
        external_state = np.hstack([np.real(netload)*1e-3, price*0.1])

        return np.hstack([internal_state, external_state]).astype('float32')
