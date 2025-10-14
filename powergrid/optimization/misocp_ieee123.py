import numpy as np
import pandapower as pp
from pyscipopt import Model, quicksum
from powergrid.networks.ieee123 import IEEE123Bus
from matplotlib import pyplot as plt
from pandapower.pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN, BUS_TYPE, NONE, VM, VA, \
    CID, CZD, bus_cols, REF
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, \
    BR_R_ASYM, BR_X_ASYM, branch_cols

def read_data():
    import pickle, os
    # Update path to point to data directory from new location
    f = open(os.path.join(os.path.dirname(__file__), '../../data','data2018-2020.pkl'), 'rb')
    data = pickle.load(f)
    f.close()
    return data

dataset = read_data()
price = dataset['test']['price'].reshape([366,24])
load_scale = dataset['test']['load'].reshape([366,24])
solar_scale = dataset['test']['solar'].reshape([366,24])
wind_scale = dataset['test']['wind'].reshape([366,24])

net = IEEE123Bus()

pp.runpp(net)
net.sgen.scaling = 0
ppc = net._ppc
eps = np.finfo(np.float32).eps

n_buses = ppc['bus'].shape[0]
n_lines = ppc['branch'].shape[0]

id2bus = dict(zip(net.bus.index.tolist(), net.bus['name'].values))
bus2id = dict(zip(net.bus['name'].values, net.bus.index.tolist()))

bus2vr = {}
for idx, trafo in net.trafo.iterrows():
    if 'Regulator' in trafo['name']:
        hv, lv = trafo[['hv_bus','lv_bus']].values
        bus2vr[hv, lv] = idx

bus2cap = {}
bus2cap = dict(zip(net.shunt['bus'].values, net.shunt['bus'].index))

r = ppc['branch'][:,BR_R].real
x = ppc['branch'][:,BR_X].real
z = np.sqrt(r**2 + x**2)
l_ub = np.array(net.line.max_i_ka.tolist()+[1.0, 0.23, 0.53, 0.53, 1.0, 1.0, 1.0, 1.0, 1.0]) * 4.16
f_bus = ppc['branch'][:,F_BUS].real[:n_lines]
t_bus = ppc['branch'][:,T_BUS].real[:n_lines]

pd = ppc['bus'][:,PD]
qd = ppc['bus'][:,QD]

vmax = 1.05
vmin = 0.95

time_steps = 24
cost, schedues = [], {}
es1_0 = es2_0 = es3_0 = 0.5
for d in range(366):
    model = Model("IEEE34")
    # Define gen variables
    PG, QG = {}, {}
    DG1_cost, DG2_cost, DG3_cost, DG4_cost, DG5_cost, GRID_cost = {}, {}, {}, {}, {}, {}
    for t in range(time_steps):
        PG[t, bus2id['Bus 150']] = P_GRID_T = model.addVar(lb=-5., ub=5., vtype="C", name="PG_Bus_150_{}".format(t))
        QG[t, bus2id['Bus 150']] = Q_GRID_T = model.addVar(lb=-5., ub=5., vtype="C", name="QG_Bus 150_{}".format(t))
        PG[t, bus2id['Bus 24']]  = P_DG1_T = model.addVar(lb=0, ub=0.66, vtype="C", name="PG_Bus 24_{}".format(t))
        QG[t, bus2id['Bus 24']]  = Q_DG1_T = model.addVar(lb=-0.5, ub=0.5, vtype="C", name="QG_Bus 24_{}".format(t))
        # PG[t, bus2id['Bus 41']]  = P_DG2_T = model.addVar(lb=0, ub=0.66, vtype="C", name="PG_Bus 41_{}".format(t))
        # QG[t, bus2id['Bus 41']]  = Q_DG2_T = model.addVar(lb=-0.5, ub=0.5, vtype="C", name="QG_Bus 41_{}".format(t))
        PG[t, bus2id['Bus 94']]  = P_DG3_T = model.addVar(lb=0, ub=0.5, vtype="C", name="PG_Bus 94_{}".format(t))
        QG[t, bus2id['Bus 94']]  = Q_DG3_T = model.addVar(lb=-0.375, ub=0.375, vtype="C", name="QG_Bus 94_{}".format(t))
        PG[t, bus2id['Bus 71']]  = P_DG4_T = model.addVar(lb=0, ub=0.5, vtype="C", name="PG_Bus 71_{}".format(t))
        QG[t, bus2id['Bus 71']]  = Q_DG4_T = model.addVar(lb=-0.375, ub=0.375, vtype="C", name="QG_Bus 71_{}".format(t))
        # PG[t, bus2id['Bus 114']] = P_DG5_T = model.addVar(lb=0, ub=0.4, vtype="C", name="PG_Bus 114_{}".format(t))
        # QG[t, bus2id['Bus 114']] = Q_DG5_T = model.addVar(lb=-0.3, ub=0.3, vtype="C", name="QG_Bus 114_{}".format(t))
        PG[t, bus2id['Bus 22']]  = 0.1 * solar_scale[d,t]
        PG[t, bus2id['Bus 250']] = 0.1 * solar_scale[d,t]
        PG[t, bus2id['Bus 43']]  = 0.1 * solar_scale[d,t]
        PG[t, bus2id['Bus 450']] = 0.1 * solar_scale[d,t]
        PG[t, bus2id['Bus 39']]  = 0.1 * solar_scale[d,t]
        PG[t, bus2id['Bus 4']]   = 0.1 * wind_scale[d,t]
        PG[t, bus2id['Bus 59']]  = 0.1 * wind_scale[d,t]
        PG[t, bus2id['Bus 46']]  = 0.1 * wind_scale[d,t]
        PG[t, bus2id['Bus 75']]  = 0.1 * wind_scale[d,t]
        PG[t, bus2id['Bus 83']]  = 0.1 * wind_scale[d,t]
        model.addCons(P_GRID_T**2 + Q_GRID_T**2 <= 5**2, name="Sub_capacity_cons_{}".format(t))
        model.addCons(P_DG1_T/np.tan(np.arccos(-np.sqrt(1-0.8**2))) <= Q_DG1_T, name="DG1_min_pf_cons_{}".format(t))
        model.addCons(P_DG1_T/np.tan(np.arccos(np.sqrt(1-0.8**2))) >= Q_DG1_T, name="DG1_max_pf_cons_{}".format(t))
        # model.addCons(P_DG2_T/np.tan(np.arccos(-np.sqrt(1-0.8**2))) <= Q_DG2_T, name="DG2_min_pf_cons_{}".format(t))
        # model.addCons(P_DG2_T/np.tan(np.arccos(np.sqrt(1-0.8**2))) >= Q_DG2_T, name="DG2_max_pf_cons_{}".format(t))
        model.addCons(P_DG3_T/np.tan(np.arccos(-np.sqrt(1-0.8**2))) <= Q_DG3_T, name="DG3_min_pf_cons_{}".format(t))
        model.addCons(P_DG3_T/np.tan(np.arccos(np.sqrt(1-0.8**2))) >= Q_DG3_T, name="DG3_max_pf_cons_{}".format(t))
        model.addCons(P_DG4_T/np.tan(np.arccos(-np.sqrt(1-0.8**2))) <= Q_DG4_T, name="DG4_min_pf_cons_{}".format(t))
        model.addCons(P_DG4_T/np.tan(np.arccos(np.sqrt(1-0.8**2))) >= Q_DG4_T, name="DG4_max_pf_cons_{}".format(t))
        # model.addCons(P_DG5_T/np.tan(np.arccos(-np.sqrt(1-0.8**2))) <= Q_DG5_T, name="DG5_min_pf_cons_{}".format(t))
        # model.addCons(P_DG5_T/np.tan(np.arccos(np.sqrt(1-0.8**2))) >= Q_DG5_T, name="DG5_max_pf_cons_{}".format(t))
        # Define cost
        a1, b1, c1 = 100, 51.6, 0.4615
        # a2, b2, c2 = 100, 51.6, 0.4615
        a3, b3, c3 = 100, 72.4, 0.5011
        a4, b4, c4 = 100, 72.4, 0.5011
        # a5, b5, c5 = 100, 81.6, 0.3011
        DG1_cost[t] = model.addVar(vtype="C", name="DG1_cost_{}".format(t))
        # DG2_cost[t] = model.addVar(vtype="C", name="DG2_cost_{}".format(t))
        DG3_cost[t] = model.addVar(vtype="C", name="DG3_cost_{}".format(t))
        DG4_cost[t] = model.addVar(vtype="C", name="DG4_cost_{}".format(t))
        # DG5_cost[t] = model.addVar(vtype="C", name="DG5_cost_{}".format(t))
        GRID_cost[t] = model.addVar(vtype="C", name="GRID_cost_{}".format(t))
        model.addCons(DG1_cost[t] == a1*(P_DG1_T**2) + b1*P_DG1_T + c1, name="DG1_cost_cons_{}".format(t)) # ($)
        # model.addCons(DG2_cost[t] == a2*(P_DG2_T**2) + b2*P_DG2_T + c2, name="DG2_cost_cons_{}".format(t)) # ($)
        model.addCons(DG3_cost[t] == a3*(P_DG3_T**2) + b3*P_DG3_T + c3, name="DG3_cost_cons_{}".format(t)) # ($)
        model.addCons(DG4_cost[t] == a4*(P_DG4_T**2) + b4*P_DG4_T + c4, name="DG4_cost_cons_{}".format(t)) # ($)
        # model.addCons(DG5_cost[t] == a5*(P_DG5_T**2) + b5*P_DG5_T + c5, name="DG5_cost_cons_{}".format(t)) # ($)
        model.addCons(GRID_cost[t] == price[d,t] * P_GRID_T, name="GRID_cost_cons_{}".format(t))
    # # Define storage variables
    eta_ch, eta_dch = 0.98, 0.98
    max_es1, min_es1, max_p1 = 2.0, 0.2, 0.5
    max_es2, min_es2, max_p2 = 2.0, 0.2, 0.5
    # max_es3, min_es3, max_p3 = 1.0, 0.1, 0.25
    PS_CH, PS_DCH, BS, ES = {}, {}, {}, {}
    for t in range(time_steps):
        # ESS 1
        PS_CH[t,bus2id['Bus 20']] = PS1_CH_T = model.addVar(vtype="C", name="PS1_CH_{}".format(t))
        PS_DCH[t,bus2id['Bus 20']] = PS1_DCH_T = model.addVar(vtype="C", name="PS1_DCH_{}".format(t))
        BS[t,bus2id['Bus 20']] = BS1_T = model.addVar(vtype="B", name="BS1_{}".format(t))
        ES[t,bus2id['Bus 20']] = ES1_T = model.addVar(lb=min_es1, ub=max_es1, vtype="C", name="ES1_{}".format(t))
        # ESS 2
        PS_CH[t,bus2id['Bus 56']] = PS2_CH_T = model.addVar(vtype="C", name="PS2_CH_{}".format(t))
        PS_DCH[t,bus2id['Bus 56']] = PS2_DCH_T = model.addVar(vtype="C", name="PS2_DCH_{}".format(t))
        BS[t,bus2id['Bus 56']] = BS2_T = model.addVar(vtype="B", name="BS_2{}".format(t))
        ES[t,bus2id['Bus 56']] = ES2_T = model.addVar(lb=min_es2, ub=max_es2, vtype="C", name="ES2_{}".format(t))
        # # ESS 3
        # PS_CH[t,bus2id['Bus 113']] = PS3_CH_T = model.addVar(vtype="C", name="PS3_CH_{}".format(t))
        # PS_DCH[t,bus2id['Bus 113']] = PS3_DCH_T = model.addVar(vtype="C", name="PS3_DCH_{}".format(t))
        # BS[t,bus2id['Bus 113']] = BS3_T = model.addVar(vtype="B", name="BS_3{}".format(t))
        # ES[t,bus2id['Bus 113']] = ES3_T = model.addVar(lb=min_es3, ub=max_es3, vtype="C", name="ES3_{}".format(t))
        # ESS 1
        model.addCons(PS1_CH_T <= max_p1 * BS1_T)
        model.addCons(PS1_CH_T >= 0.0 * BS1_T)
        model.addCons(PS1_DCH_T <= max_p1 * (1 - BS1_T))
        model.addCons(PS1_DCH_T >= 0.0 * (1 - BS1_T))
        # ESS 2
        model.addCons(PS2_CH_T <= max_p2 * BS2_T)
        model.addCons(PS2_CH_T >= 0.0 * BS2_T)
        model.addCons(PS2_DCH_T <= max_p2 * (1 - BS2_T))
        model.addCons(PS2_DCH_T >= 0.0 * (1 - BS2_T))
        # # ESS 3
        # model.addCons(PS3_CH_T <= max_p3 * BS3_T)
        # model.addCons(PS3_CH_T >= 0.0 * BS3_T)
        # model.addCons(PS3_DCH_T <= max_p3 * (1 - BS3_T))
        # model.addCons(PS3_DCH_T >= 0.0 * (1 - BS3_T))
        if t == 0:
            model.addCons(ES1_T == es1_0 + PS1_CH_T*eta_ch - PS1_DCH_T/eta_ch)
            model.addCons(ES2_T == es2_0 + PS2_CH_T*eta_ch - PS2_DCH_T/eta_ch)
            # model.addCons(ES3_T == es3_0 + PS3_CH_T*eta_ch - PS3_DCH_T/eta_ch)
        else:
            model.addCons(ES1_T == ES[t-1,bus2id['Bus 20']] + PS1_CH_T*eta_ch - PS1_DCH_T/eta_ch)
            model.addCons(ES2_T == ES[t-1,bus2id['Bus 56']] + PS2_CH_T*eta_ch - PS2_DCH_T/eta_ch)
            # model.addCons(ES3_T == ES[t-1,bus2id['Bus 113']] + PS3_CH_T*eta_ch - PS3_DCH_T/eta_ch)
    # Define variables I^{2}
    I_SR, P_IJ, Q_IJ = {}, {}, {}
    for t in range(time_steps):
        for k in range(n_lines):
            I_SR[t,k] = model.addVar(lb=0, ub=l_ub[k]**2, vtype="C", name="I_SR_{}_{}".format(k,t))
            P_IJ[t,k] = model.addVar(lb=-10, ub=10, vtype="C", name="P_IJ_{}_{}".format(k,t))
            Q_IJ[t,k] = model.addVar(lb=-10, ub=10, vtype="C", name="Q_IJ_{}_{}".format(k,t))
    # Define variables V^{2}
    V_SR = {}
    for t in range(time_steps):
        for i in range(n_buses):
            if ppc['bus'][i,BUS_TYPE] == 3: # reference bus
                V_SR[t,i] = 1.0 # model.addVar(lb=1-eps, ub=1+eps, vtype="C", name="V_SR_{}".format(i))
            else:
                V_SR[t,i] = model.addVar(lb=vmin**2, ub=vmax**2, vtype="C", name="V_SR_{}_{}".format(i,t))
    # Define VR variables
    VR_TAP = {}
    for t in range(time_steps):
        for (fbus,tbus), v in bus2vr.items():
            lb, ub = net.trafo.loc[v].tap_min, net.trafo.loc[v].tap_max
            VR_TAP[t,v] = model.addVar(lb=lb, ub=ub, vtype="I", name=net.trafo.loc[v]['name']+"_{}".format(t))
    # Define shunt capacitor variables
    CAP = {}
    for t in range(time_steps):
        for b, c in bus2cap.items():
            lb, ub = 0, net.shunt.loc[c].max_step
            CAP[t,c] = model.addVar(lb=lb, ub=ub, vtype="I", name=net.shunt.loc[c]['name']+"_{}".format(t))
    # Define dist flow constraints
    for t in range(time_steps):
        for i in range(n_buses):
            # load flow from upstream nodes
            SUM_P_KI, SUM_Q_KI = 0.0, 0.0
            for k in np.where(t_bus==i)[0]:
                SUM_P_KI += P_IJ[t,k] - r[k]*I_SR[t,k]
                SUM_Q_KI += Q_IJ[t,k] - x[k]*I_SR[t,k]
            # load flow from downstream nodes
            SUM_P_IJ, SUM_Q_IJ = 0.0, 0.0
            for j in np.where(f_bus==i)[0]:
                SUM_P_IJ += P_IJ[t,j]
                SUM_Q_IJ += Q_IJ[t,j]
            # power injection into node i
            PG_I = PG.get((t,i)) or 0.0
            QG_I = QG.get((t,i)) or 0.0
            PS_I = PS_DCH.get((t,i)) or 0.0
            PS_I -= PS_CH.get((t,i)) or 0.0
            CAP_I = CAP[t,bus2cap[i]]*net.shunt.loc[bus2cap[i]].q_mvar if bus2cap.get(i) is not None else 0.0
            model.addCons(PG_I + PS_I - pd[i]*load_scale[d,t] == SUM_P_IJ - SUM_P_KI)
            model.addCons(QG_I - CAP_I - qd[i]*load_scale[d,t] == SUM_Q_IJ - SUM_Q_KI)
    # Define dist flow constraints
    for t in range(time_steps):
        for ij in range(n_lines):
            i = int(f_bus[ij])
            j = int(t_bus[ij])
            if bus2vr.get((i,j)):
                v = bus2vr.get((i,j))
                tap_step_percent = net.trafo.loc[v].tap_step_percent
                tap_neutral = net.trafo.loc[v].tap_neutral
                tap_step = (VR_TAP[t,v] - tap_neutral) * tap_step_percent / 100
                ratio = 1 + tap_step if net.trafo.loc[v].tap_side == 'lv' else 1 / (1 + tap_step)
            else:
                ratio = 1
            model.addCons(V_SR[t,j]*(ratio*ratio) == V_SR[t,i] - 2*(r[ij]*P_IJ[t,ij] + x[ij]*Q_IJ[t,ij]) + z[ij]*z[ij]*I_SR[t,ij])
            model.addCons(I_SR[t,ij]*V_SR[t,i] >= P_IJ[t,ij]*P_IJ[t,ij] + Q_IJ[t,ij]*Q_IJ[t,ij])
    model.setObjective(quicksum(DG1_cost[t]+DG3_cost[t]+DG4_cost[t]+GRID_cost[t] for t in range(time_steps)))
    model.hideOutput()
    model.setRealParam('limits/time', 60)
    model.optimize()
    obj = model.getObjVal()
    sol = model.getBestSol()
    print('Day {} is done!'.format(d))
    cost.append(obj)
    v_pu = np.empty([time_steps, n_buses])
    for t in range(time_steps):
        for i in range(n_buses-1):
            v_pu[t,i] = np.sqrt(sol[V_SR[t,i]])
    i_pu = np.empty([time_steps, n_lines])
    for t in range(time_steps):
        for ij in range(1, n_lines):
            i_pu[t,ij] = np.sqrt(sol[I_SR[t,ij]])
    cap1 = np.empty([time_steps])
    for t in range(time_steps):
        cap1[t] = sol[CAP[t,0]]
    cap2 = np.empty([time_steps])
    for t in range(time_steps):
        cap2[t] = sol[CAP[t,1]]
    # VR 1
    vr_tap1 = np.empty([time_steps])
    for t in range(time_steps):
        vr_tap1[t] = sol[VR_TAP[t,0]]
    # VR 2
    vr_tap2 = np.empty([time_steps])
    for t in range(time_steps):
        vr_tap2[t] = sol[VR_TAP[t,1]]
    # VR 3
    vr_tap3 = np.empty([time_steps])
    for t in range(time_steps):
        vr_tap3[t] = sol[VR_TAP[t,2]]
    # VR 4
    vr_tap4 = np.empty([time_steps])
    for t in range(time_steps):
        vr_tap4[t] = sol[VR_TAP[t,2]]
    # DG 1
    p_dg1, q_dg1 = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p_dg1[t] = sol[PG[t,bus2id['Bus 24']]]
        q_dg1[t] = sol[QG[t,bus2id['Bus 24']]]
    # # DG 2
    # p_dg2, q_dg2 = np.empty([time_steps]), np.empty([time_steps])
    # for t in range(time_steps):
    #     p_dg2[t] = sol[PG[t,bus2id['Bus 41']]]
    #     q_dg2[t] = sol[QG[t,bus2id['Bus 41']]]
    # DG 3
    p_dg3, q_dg3 = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p_dg3[t] = sol[PG[t,bus2id['Bus 94']]]
        q_dg3[t] = sol[QG[t,bus2id['Bus 94']]]
    # DG 4
    p_dg4, q_dg4 = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p_dg4[t] = sol[PG[t,bus2id['Bus 71']]]
        q_dg4[t] = sol[QG[t,bus2id['Bus 71']]]
    # # DG 5
    # p_dg5, q_dg5 = np.empty([time_steps]), np.empty([time_steps])
    # for t in range(time_steps):
    #     p_dg5[t] = sol[PG[t,bus2id['Bus 114']]]
    #     q_dg5[t] = sol[QG[t,bus2id['Bus 114']]]
    # Grid
    p_grid, q_grid = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p_grid[t] = sol[PG[t,bus2id['Bus 150']]]
        q_grid[t] = sol[QG[t,bus2id['Bus 150']]]
    # ESS 1
    p1_s, e1_s = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p1_s[t] = sol[PS_DCH[t,bus2id['Bus 20']]] - sol[PS_CH[t,bus2id['Bus 20']]]
        e1_s[t] = sol[ES[t,bus2id['Bus 20']]]
    es1_0 = e1_s[-1]
    # ESS 2
    p2_s, e2_s = np.empty([time_steps]), np.empty([time_steps])
    for t in range(time_steps):
        p2_s[t] = sol[PS_DCH[t,bus2id['Bus 56']]] - sol[PS_CH[t,bus2id['Bus 56']]]
        e2_s[t] = sol[ES[t,bus2id['Bus 56']]]
    es2_0 = e2_s[-1]
    # # ESS 3
    # p3_s, e3_s = np.empty([time_steps]), np.empty([time_steps])
    # for t in range(time_steps):
    #     p3_s[t] = sol[PS_DCH[t,bus2id['Bus 113']]] - sol[PS_CH[t,bus2id['Bus 113']]]
    #     e3_s[t] = sol[ES[t,bus2id['Bus 113']]]
    # es3_0 = e3_s[-1]
    # renew
    p_renew = []
    for t in range(time_steps):
        p = PG[t,bus2id['Bus 22']]  * solar_scale[d,t] + \
            PG[t,bus2id['Bus 250']] * solar_scale[d,t] + \
            PG[t,bus2id['Bus 43']]  * solar_scale[d,t] + \
            PG[t,bus2id['Bus 450']] * solar_scale[d,t] + \
            PG[t,bus2id['Bus 39']]  * solar_scale[d,t] + \
            PG[t,bus2id['Bus 4']]   * wind_scale[d,t] + \
            PG[t,bus2id['Bus 59']]  * wind_scale[d,t] + \
            PG[t,bus2id['Bus 46']]  * wind_scale[d,t] + \
            PG[t,bus2id['Bus 75']]  * wind_scale[d,t] + \
            PG[t,bus2id['Bus 83']]  * wind_scale[d,t]
        p_renew.append(p)
    for t in range(time_steps):
        print(p_grid[t]+p_dg1[t]+p_dg3[t]+p_dg4[t]+p_renew[t]+p1_s[t]+p2_s[t], np.sum(pd * load_scale[d,t]))
        print(q_grid[t]+q_dg1[t]+q_dg3[t]+q_dg4[t]-cap1[t]*(-0.3)-cap2[t]*(-0.3), np.sum(qd * load_scale[d,t]))
    # dg1_cost, dg2_cost, grid_cost = 0, 0, 0
    # for t in range(time_steps):
    #     dg1_cost += a1 * (p_dg1[t]**2) + b1 * p_dg1[t] + c1
    #     dg2_cost += a2 * (p_dg2[t]**2) + b2 * p_dg2[t] + c2
    #     grid_cost += price[d,t] * p_grid[t]
    # print(dg1_cost+dg2_cost+grid_cost)
    schedues[d] = {
        'grid_P': p_grid, 
        'grid_Q': q_grid, 
        'DG1_P': p_dg1, 
        'DG1_Q': q_dg1, 
        'DG3_P': p_dg3, 
        'DG3_Q': q_dg3, 
        'DG4_P': p_dg4, 
        'DG4_Q': q_dg4, 
        'ESS1_E': e1_s, 
        'ESS1_P': p1_s, 
        'ESS2_E': e2_s, 
        'ESS2_P': p2_s, 
        'VR1': vr_tap1,
        'VR2': vr_tap2,
        'VR3': vr_tap3,
        'VR4': vr_tap4,
        'SCB1': cap1,
        'SCB2': cap2,
        'voltage': v_pu, 
        'current': i_pu, 
    }

    print(cost)

# import pickle, os
# save_path = '/home/lihepeng/Documents/Github/learning2opDN/results/test/misocp_ieee123.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump({'cost': cost, 'schedues': schedues}, f, protocol=pickle.HIGHEST_PROTOCOL)
# print('done')

# import pickle, os
# save_path = '/home/lihepeng/Documents/Github/learning2opDN/results/test/misocp_ieee123.pkl'
# with open(save_path, 'rb') as f:
#     data = pickle.load(f)

# print('done')
