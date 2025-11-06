from typing import Dict, Tuple
import numpy as np
from pyscipopt import Model, quicksum
from pandapower.pypower.idx_bus import PD, QD, BUS_TYPE
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X
from pandapower.pypower.idx_brch_tdpf import RATE_I_KA

def _bus_maps(net):
    bus_names = net.bus["name"].values.tolist()
    bus2id = {name: int(idx) for idx, name in zip(net.bus.index, bus_names)}
    id2bus = {v: k for k, v in bus2id.items()}
    return bus2id, id2bus

def _cap_maps(net, id2bus):
    """" shunt capacitor """
    # bus2id, id2bus = _bus_maps(net)
    bid2cap = dict(zip(net.shunt['bus'].values, net.shunt['name'].values))
    bus2cap = {id2bus[bus_id]: cid for bus_id, cid in bid2cap.items()}
    cap2bus = {cid: bus for bus, cid in bus2cap.items()}

    return bus2cap, cap2bus

def _vr_maps(net, id2bus):
    """" voltage regulator """
    # bus2id, id2bus = _bus_maps(net)
    bus2vr = {}
    for idx, trafo in net.trafo.iterrows():
        if 'Regulator' in trafo['name']:
            hv, lv = trafo[['hv_bus','lv_bus']].values
            bus2vr[(hv, lv)] = idx

    return bus2vr

# build discrete ratios for each regulator index
def regulator_tap_positions(trafo_row):
    tmin = int(trafo_row["tap_min"])
    tmax = int(trafo_row["tap_max"])
    tneu = int(trafo_row["tap_neutral"])
    step_pct = float(trafo_row["tap_step_percent"]) / 100.0  # fractional per step
    side = str(trafo_row["tap_side"]).lower()  # 'lv' or 'hv'
    taps = list(range(tmin, tmax + 1))
    ratios = {}
    for k in taps:
        tap_step = (k - tneu) * step_pct
        if side == "lv":
            rho = 1.0 + tap_step
        else:  # 'hv'
            rho = 1.0 / (1.0 + tap_step)
        ratios[k] = rho
    return taps, ratios  # dict: tap -> ratio

def solve_one_day(
    net,
    scenario: Dict,
    day_data: Dict,
    T: int
) -> Dict:
    """
    Generic MISOCOPF for T timesteps with arbitrary sets
    (generators/storage/renewables defined in scenario).
    """
    ppc = net._ppc
    n_b, n_l = ppc["bus"].shape[0], ppc["branch"].shape[0]
    r = ppc["branch"][:, BR_R].real
    x = ppc["branch"][:, BR_X].real
    z2 = r*r + x*x
    f_bus = ppc["branch"][:, F_BUS].real.astype(int)
    t_bus = ppc["branch"][:, T_BUS].real.astype(int)
    line_max_I_kA = net.line.max_i_ka.values
    trafo_max_I_kA = net.trafo.assign(
        I_rated_lv_kA = lambda d: d.sn_mva / (np.sqrt(3) * d.vn_lv_kv),
        I_rated_hv_kA = lambda d: d.sn_mva / (np.sqrt(3) * d.vn_hv_kv),
    ).assign(
        max_I_kA = lambda d: d[["I_rated_lv_kA", "I_rated_hv_kA"]].max(axis=1)
    )["max_I_kA"].values
    l_ub = init_l_ub = np.concatenate([line_max_I_kA, trafo_max_I_kA])
    delta_l_ub = 0.1 * l_ub

    PD0 = ppc["bus"][:, PD]
    QD0 = ppc["bus"][:, QD]
    bus2id, id2bus = _bus_maps(net)

    bus2vr = _vr_maps(net, id2bus)
    bus2cap, cap2bus = _cap_maps(net, id2bus)

    vmin = float(scenario["limits"]["vmin"])
    vmax = float(scenario["limits"]["vmax"])
    pf_min = float(scenario["limits"]["pf_min"])
    s_slack_max = float(scenario["limits"]["s_slack_max"])

    gens = scenario.get("generators", {})
    stor = scenario.get("storage", {})
    rens = scenario.get("renewables", {})
    caps = scenario.get("capacitors", {})
    regs = scenario.get("regulators", {})
    slack_bus = bus2id[scenario["slack"]["bus"]]

    eta_ch = float(scenario.get("storage_defaults", {}).get("eta_ch", 0.98))
    eta_dch = float(scenario.get("storage_defaults", {}).get("eta_dch", 0.98))

    price = np.asarray(day_data["price"], dtype=float)          # (T,)
    solar = np.asarray(day_data.get("solar", np.zeros(T)), float)
    wind  = np.asarray(day_data.get("wind",  np.zeros(T)), float)
    load_scale = np.asarray(day_data.get("load", np.ones(T)), float)  # area profile multiplier

    branch_i_to_adjust = 0
    converged = False
    while not converged:
        # Build model
        m = Model("GenericMISOCOPF")

        # Decision vars
        Pij, Qij, I2, V2 = {}, {}, {}, {}
        Pg, Qg = {}, {}
        Ps_ch, Ps_dch, Es, ch_dsh_mode = {}, {}, {}, {}
        Qc = {}
        Tap, Ratio = {}, {}

        # Slack vars per t
        sell_discount = float(scenario.get("slack", {}).get("sell_discount", 1.0))

        P_import, P_export, P_net, Q_slack, y_mode = {}, {}, {}, {}, {}
        for t in range(T):
            P_import[t] = m.addVar(lb=0.0,          ub=s_slack_max,  vtype="C", name=f"P_import[{t}]")
            P_export[t] = m.addVar(lb=0.0,          ub=s_slack_max,  vtype="C", name=f"P_export[{t}]")
            P_net[t]    = m.addVar(lb=-s_slack_max, ub= s_slack_max, vtype="C", name=f"P_net[{t}]")
            Q_slack[t]  = m.addVar(lb=-s_slack_max, ub= s_slack_max, vtype="C", name=f"Q_slack[{t}]")
            y_mode[t]   = m.addVar(vtype="B", name=f"y_import_mode[{t}]")  # 1=import, 0=export

            # link net = import - export
            m.addCons(P_net[t] == P_import[t] - P_export[t])

            # if y=1 (import), P_export <= 0; if y=0 (export), P_import <= 0
            m.addCons(P_import[t] <= s_slack_max * y_mode[t])
            m.addCons(P_export[t] <= s_slack_max * (1 - y_mode[t]))

            # Apparent power cap on net exchange
            m.addCons(P_net[t]*P_net[t] + Q_slack[t]*Q_slack[t] <= s_slack_max*s_slack_max)
            
        # Generators
        for gid, g in gens.items():
            i = bus2id[g["bus"]]
            for t in range(T):
                Pg[t, gid] = m.addVar(lb=g["p_min"], ub=g["p_max"], vtype="C", name=f"P_{gid}[{t}]")
                Qg[t, gid] = m.addVar(lb=g["q_min"], ub=g["q_max"], vtype="C", name=f"Q_{gid}[{t}]")

        # Storage
        for sid, s in stor.items():
            i = bus2id[s["bus"]]
            for t in range(T):
                Ps_ch[t, sid]  = m.addVar(lb=0.0,         ub=s["p_max"], vtype="C", name=f"Ps_ch_{sid}[{t}]")
                Ps_dch[t, sid] = m.addVar(lb=0.0,         ub=s["p_max"], vtype="C", name=f"Ps_dch_{sid}[{t}]")
                Es[t, sid]     = m.addVar(lb=s["e_min"],  ub=s["e_max"], vtype="C", name=f"E_{sid}[{t}]")
                ch_dsh_mode[t, sid] = m.addVar(vtype="B", name=f"ch_dsh_mode_{sid}[{t}]")  # 1=charge, 0=discharge

                # if ch_dsh_mode=1 (charge), P_export <= 0; if y=0 (export), P_import <= 0
                m.addCons(Ps_ch[t, sid] <= s["p_max"] * ch_dsh_mode[t, sid])
                m.addCons(Ps_dch[t, sid] <= s["p_max"] * (1 - ch_dsh_mode[t, sid]))

        # Storage dynamics
        for sid, s in stor.items():
            e0 = float(s.get("e_init", s["e_min"]))
            for t in range(T):
                if t == 0:
                    m.addCons(Es[t, sid] == e0 + eta_ch*Ps_ch[t, sid] - Ps_dch[t, sid]/eta_dch)
                else:
                    m.addCons(Es[t, sid] == Es[t-1, sid] + eta_ch*Ps_ch[t, sid] - Ps_dch[t, sid]/eta_dch)

        # Define VR variables
        for t in range(T):
            for (fbus, tbus), vr_idx in bus2vr.items():
                row = net.trafo.loc[vr_idx]
                taps, ratios = regulator_tap_positions(row)
                vid = row["name"]
                if regs is not None and vid in regs: # controllable
                    tmin, tmax = taps[0], taps[1]
                    Tap[t, (fbus, tbus)] = m.addVar(lb=tmin, ub=tmax, vtype="I", name=vid+"_{}".format(t))
                    tneu = int(row["tap_neutral"])
                    tneu = int(row["tap_step_percent"])
                    tap_step = (Tap[t, (fbus, tbus)] - tneu) * tneu / 100
                    ratio = 1 + tap_step if row.tap_side == 'lv' else 1 / (1 + tap_step)
                    Ratio[t, (fbus, tbus)] = ratio
                else:
                    Tap[t, (fbus, tbus)] = row["tap_pos"]
                    Ratio[t, (fbus, tbus)] = ratios[row["tap_pos"]]

        # Shunt capacitors
        for bus, cid in bus2cap.items():
            idx = net.shunt[net.shunt["name"] == cid].index[0]
            q_max = net.shunt.loc[idx].q_mvar
            if caps is not None and cid in caps: # controllable
                for t in range(T):
                    lb, ub = 0, net.shunt.loc[idx].max_step
                    name = net.shunt.loc[idx]['name']+"_{}".format(t)
                    Qc[t, cid] = m.addVar(lb=lb, ub=ub, vtype="I", name=name) * q_max
            else:
                for t in range(T):
                    Qc[t, cid] = net.shunt.loc[idx].step * q_max

        # Branch flows and squared voltages/currents
        for t in range(T):
            for k in range(n_l):
                Pij[t, k] = m.addVar(lb=-1e3, ub=1e3, vtype="C", name=f"Pij[{t},{k}]")
                Qij[t, k] = m.addVar(lb=-1e3, ub=1e3, vtype="C", name=f"Qij[{t},{k}]")
                I2[t, k]  = m.addVar(lb=0.0,  ub=l_ub[k]**2, vtype="C", name=f"I2[{t},{k}]")
            for i in range(n_b):
                if ppc["bus"][i, BUS_TYPE] == 3:    # slack
                    V2[t, i] = 1.0
                else:
                    V2[t, i] = m.addVar(lb=vmin*vmin, ub=vmax*vmax, vtype="C", name=f"V2[{t},{i}]")

        # PF constraints for controllable gens (including slack implicitly via S limit)
        phi = float(np.tan(np.arccos(pf_min)))
        for gid, g in gens.items():
            for t in range(T):
                m.addCons( Qg[t, gid] <=  Pg[t, gid] *  phi )
                m.addCons(-Qg[t, gid] <=  Pg[t, gid] *  phi )

        # Slack S limit
        for t in range(T):
            m.addCons(P_net[t]*P_net[t] + Q_slack[t]*Q_slack[t] <= s_slack_max*s_slack_max)

        # Renewable injections lookup per (t, bus) from profiles
        ren_at_bus = {
            (t, bus2id[ren["bus"]]): 0.0 for t in range(T) for ren in rens.values()
        }

        for rid, ren in rens.items():
            src = ren["source"]
            if src == "solar":
                series = solar
            elif src == "wind":
                series = wind
            else:
                # custom profile name is unsupported here; use zeros by default or plug your adapter
                series = np.zeros(T, dtype=float)

            bus_id = bus2id[ren["bus"]]
            scale = float(ren["scale"])
            for t in range(T):
                ren_at_bus[(t, bus_id)] = ren_at_bus.get((t, bus_id), 0.0) + scale * float(series[t])

        # Nodal balances
        for t in range(T):
            for i in range(n_b):
                # inflow (k -> i)
                Pin = quicksum(Pij[t, k] - r[k]*I2[t, k] for k in np.where(t_bus == i)[0])
                Qin = quicksum(Qij[t, k] - x[k]*I2[t, k] for k in np.where(t_bus == i)[0])
                # outflow (i -> j)
                Pout = quicksum(Pij[t, k] for k in np.where(f_bus == i)[0])
                Qout = quicksum(Qij[t, k] for k in np.where(f_bus == i)[0])

                # controlled gen
                Pg_i = quicksum(Pg[t, gid] for gid, g in gens.items() if bus2id[g["bus"]] == i)
                Qg_i = quicksum(Qg[t, gid] for gid, g in gens.items() if bus2id[g["bus"]] == i)

                # storage net
                Ps_i = quicksum(Ps_dch[t, sid] - Ps_ch[t, sid] for sid, s in stor.items() if bus2id[s["bus"]] == i)

                # capacitor gen
                Qc_i = quicksum(Qc[t, cid] for bus, cid in bus2cap.items() if bus2id[bus] == i)

                # slack at its bus
                Psl_i = P_net[t]   if i == slack_bus else 0.0
                Qsl_i = Q_slack[t] if i == slack_bus else 0.0

                # renewables (params)
                Pren_i = ren_at_bus.get((t, i), 0.0)

                # scaled load (area profile multiplies all PD/QD uniformly)
                Pd_i = PD0[i] * load_scale[t]
                Qd_i = QD0[i] * load_scale[t]

                # Balance
                m.addCons(Pin + Pg_i + Ps_i + Psl_i + Pren_i - Pd_i == Pout)
                m.addCons(Qin + Qg_i + Qsl_i + Qc_i          - Qd_i == Qout)

        # Branch constraints (Baran–Wu)
        for t in range(T):
            for k in range(n_l):
                i, j = int(f_bus[k]), int(t_bus[k])
                ratio = Ratio.get((t, (i, j)), 1.0)
                m.addCons(V2[t, j]*(ratio**2) == V2[t, i] - 2*(r[k]*Pij[t, k] + x[k]*Qij[t, k]) + z2[k]*I2[t, k])
                m.addCons(I2[t, k]*V2[t, i] >= Pij[t, k]*Pij[t, k] + Qij[t, k]*Qij[t, k])

        # Cost via epigraph variables (linear objective)
        z = {}  # cost epigraph variables
        obj_terms = []

        for gid, g in gens.items():
            a, b, c = float(g["cost"]["a"]), float(g["cost"]["b"]), float(g["cost"]["c"])
            for t in range(T):
                z[t, gid] = m.addVar(lb=-1e10, ub=1e10, vtype="C", name=f"z_{gid}[{t}]")
                # Convex epigraph constraint: z >= a*Pg^2 + b*Pg + c
                # (Equality would also work, but >= is the safe convex form.)
                m.addCons(z[t, gid] >= a*(Pg[t, gid]*Pg[t, gid]) + b*Pg[t, gid] + c)
                obj_terms.append(z[t, gid])

        # Grid energy cost (linear)
        for t in range(T):
            obj_terms.append(price[t] * P_import[t])                    # pay for imports
            obj_terms.append(- sell_discount * price[t] * P_export[t])  # revenue for exports

        m.setObjective(quicksum(obj_terms))

        m.setRealParam("limits/time", 600.0)
        m.hideOutput()
        m.optimize()
        try:
            sol = m.getBestSol()
            obj = float(m.getObjVal())
            converged = True
        except:
            l_ub = init_l_ub.copy()
            l_ub[branch_i_to_adjust] += delta_l_ub[branch_i_to_adjust]
            branch_i_to_adjust += 1
            if branch_i_to_adjust == len(l_ub):
                init_l_ub += delta_l_ub
            branch_i_to_adjust %= len(l_ub)

    # cost_dg = np.sum(sol[z[t, gid]] for gid in gens for t in range(T))
    # cost_grid = price[t] * sol[P_import[t]] - sell_discount * price[t] * sol[P_export[t]]
    # obj = cost_dg + cost_grid

    # for t in range(T):
    #     np.array(sol[z[t, gid]] for gid in gens)
    #     p_slack = sol[P_net[t]]
    #     p_g = np.sum([np.array([sol[Pg[t, gid]]]) for gid in gens])
    #     p_s = np.sum([np.array(sol[Ps_ch[t, sid]] - sol[Ps_dch[t, sid]]) for sid in stor])
    #     p_net_load = np.sum([PD0[i] * load_scale[t]  - ren_at_bus.get((t, i), 0.0) for i in range(n_b)])

    #     print(p_slack + p_g - p_s - p_net_load)

    # Extract a minimal, general result package
    result = {
        "obj": obj,
        "P_slack": np.array([sol[P_net[t]] for t in range(T)]),
        "Q_slack": np.array([sol[Q_slack[t]] for t in range(T)]),
        "Pg": {gid: np.array([sol[Pg[t, gid]] for t in range(T)]) for gid in gens},
        "Qg": {gid: np.array([sol[Qg[t, gid]] for t in range(T)]) for gid in gens},
        "Es": {sid: np.array([sol[Es[t, sid]] for t in range(T)]) for sid in stor},
        "Ps": {sid: np.array([sol[Ps_ch[t, sid]] - sol[Ps_dch[t, sid]] for t in range(T)]) for sid in stor},
        "Vm": {id2bus[i]: np.sqrt([sol[V2[t, i]] if ppc["bus"][i, BUS_TYPE] != 3 else V2[t, i]
                             for t in range(T)]) for i in range(n_b)},
        "price": price,
    }
    return result
