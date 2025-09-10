import os, pickle
import numpy as np
import pandapower as pp

from os.path import dirname, abspath
from collections import OrderedDict

from powergrid.envs.base_env import GridBaseEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.core import *

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

class IEEE13Env(GridBaseEnv):
    @override
    def _build_net(self):
        self.area = "MG1"
        net = IEEE13Bus(self.area)
        # Register devices (names must be unique per area)
        ess1 = ESS('ESS1', bus='Bus 645',
            min_p_mw=-0.5,
            max_p_mw=0.5,
            capacity=3.0,
            max_e_mwh=2.7,
            min_e_mwh=0.3,
        )
        dg1 = DG('DG1', bus='Bus 675', 
            min_p_mw=0.0, 
            max_p_mw=0.66, 
            sn_mva=1.0,
            cost_curve_coefs=[100, 51.6, 0.4615],
        )
        dg2 = DG('DG2', bus='Bus 611', 
            min_p_mw=0.0, 
            max_p_mw=0.50, 
            sn_mva=0.625, 
            cost_curve_coefs=[100, 72.4, 0.5011],
        )
        pv1 = RES('PV1', bus='Bus 652', sn_mva=0.5, source='solar')
        wt1 = RES('WT1', bus='Bus 645', sn_mva=0.5, source='wind')
        grid = Grid("Grid", bus='Bus 650', sn_mva=5, sell_discount=0.9)
        trafos = []
        for index, row in net.trafo.iterrows():
            name, sn_mva = row["name"][len(self.area)+1:], row["sn_mva"]
            trafos.append((name, Transformer(name=name, sn_mva=sn_mva)))
        # Let the base take care of attaching these to the net
        self.devices = OrderedDict([
            (ess1.name, ess1),
            (dg1.name, dg1),
            (dg2.name, dg2),
            (pv1.name, pv1),
            (wt1.name, wt1),
            (grid.name, grid),
            *trafos,
        ])
        self.net = net

        # Provide dataset series (T time steps)
        self.dataset = read_data(self.train, 'AVA', 'NP15', '0096WD_7_N001')

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
