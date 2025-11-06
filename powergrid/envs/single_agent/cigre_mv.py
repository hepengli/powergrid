import os, pickle
import numpy as np
import pandapower as pp

from os.path import dirname, abspath
from collections import OrderedDict

from powergrid.base_env import GridBaseEnv
from powergrid.devices import *
from powergrid.networks.modified_cigre_mv import create_cigre_network_mv

def read_data(train, load_area, renew_area, price_area):
    dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
    data_dir = os.path.join(dir, 'data', 'data2023-2024.pkl')
    with open(data_dir, 'rb') as file:
        dataset = pickle.load(file)

    return {
        'load' : dataset[train]['load'][load_area],
        'solar': dataset[train]['solar'][renew_area],
        'wind' : dataset[train]['wind'][renew_area],
        'price': dataset[train]['price'][price_area]
    }

class CIGREMVEnv(GridBaseEnv):
    def _build_net(self):
        self.area = ""
        net = create_cigre_network_mv()
        net.area = self.area
        # Register devices (names must be unique per area)
        RFC_1 = DG('Residential fuel cell 1', bus='Bus 5', min_p_mw=0, max_p_mw=0.033, sn_mva=0.033, cost_curve_coefs=[30, 92.6, 3.21])
        RFC_2 = DG('Residential fuel cell 2', bus='Bus 10', min_p_mw=0, max_p_mw=0.014, sn_mva=0.014, cost_curve_coefs=[30, 85.4, 4.73])
        FC    = DG('Fuel cell', bus='Bus 9', min_p_mw=0, max_p_mw=0.212, sn_mva=0.212, cost_curve_coefs=[60, 70.7, 3.52])
        CHP   = DG('CHP diesel', bus='Bus 9', min_p_mw=0, max_p_mw=0.310, sn_mva=0.310, cost_curve_coefs=[50, 65.8, 5.51])
        PV_3 = RES('PV 3', source='solar', bus='Bus 3', sn_mva=0.02)
        PV_4 = RES('PV 4', source='solar', bus='Bus 4', sn_mva=0.02)
        PV_5 = RES('PV 5', source='solar', bus='Bus 5', sn_mva=0.03)
        PV_6 = RES('PV 6', source='solar', bus='Bus 6', sn_mva=0.03)
        PV_8 = RES('PV 8', source='solar', bus='Bus 8', sn_mva=0.03)
        PV_9 = RES('PV 9', source='solar', bus='Bus 9', sn_mva=0.03)
        PV_10 = RES('PV 10', source='solar', bus='Bus 10', sn_mva=0.04)
        PV_11 = RES('PV 11', source='solar', bus='Bus 11', sn_mva=0.01)
        WKA_7 = RES('WKA 7', source='wind', bus='Bus 7', sn_mva=1.5)
        BAT_1 = ESS('Battery 1', bus='Bus 5', min_p_mw=-0.6, max_p_mw=0.6, capacity=4, min_e_mwh=0.4, max_e_mwh=3.6)
        BAT_2 = ESS('Battery 2', bus='Bus 10', min_p_mw=-0.2, max_p_mw=0.2, capacity=1, min_e_mwh=0.1, max_e_mwh=0.9)
        GRID = Grid('GRID', bus='Bus 0', sn_mva=5000, sell_discount=0.9)
        trafos = []
        for index, row in net.trafo.iterrows():
            name, sn_mva = row["name"], row["sn_mva"]
            trafos.append((name, Transformer(name=name, sn_mva=sn_mva)))
        # Let the base take care of attaching these to the net
        self.devices = OrderedDict([
            (BAT_1.name, BAT_1),
            (BAT_2.name, BAT_2),
            (RFC_1.name, RFC_1),
            (RFC_2.name, RFC_2),
            (FC.name, FC),
            (CHP.name, CHP),
            (PV_3.name, PV_3),
            (PV_4.name, PV_4),
            (PV_5.name, PV_5),
            (PV_6.name, PV_6),
            (PV_8.name, PV_8),
            (PV_9.name, PV_9),
            (PV_10.name, PV_10),
            (PV_11.name, PV_11),
            (WKA_7.name, WKA_7),
            (GRID.name, GRID),
            *trafos,
        ])
        self.net = net

        # Provide dataset series (T time steps)
        self.dataset = read_data(self.train, 'BANC', 'NP15', '0096WD_7_N001')

    def _reward_and_safety(self):
        """
        Example:
          - If PF converged: reward = -device.cost, safety = device.safety
          - If not converged: flat penalty
          - Optional penalty coefficient can be passed in cfg['penalty']
        """
        if self.net["converged"]:
            bus_ids = pp.get_element_index(self.net, 'bus', self.area, False)
            vm = self.net.res_bus.loc[bus_ids].vm_pu.values
            overvoltage = np.maximum(vm - 1.05, 0).sum()
            undervoltage = np.maximum(0.95 - vm, 0).sum()

            line_ids = pp.get_element_index(self.net, 'line', self.area, False)
            line_loading = self.net.res_line.loc[line_ids].loading_percent.values
            overloading = np.maximum(line_loading - 100, 0).sum() * 0.01

            reward = {n: -float(dev.cost) for n, dev in self.devices.items()}
            safety = {n:  float(dev.safety) for n, dev in self.devices.items()}
            safety.update({
                "overloading": overloading,
                "overvoltage": overvoltage,
                "undervoltage": undervoltage,
            })
        else:
            reward = {"convergence": -100.0}
            safety  = {"convergence": 1.0}

        self.reward = reward
        self.safety = safety

        return reward, safety


if __name__ == '__main__':
    from powergrid.envs.single_agent.cigre_mv import CIGREMVEnv
    env = CIGREMVEnv(env_config={})
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)