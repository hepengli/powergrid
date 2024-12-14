import os, pickle
import pandapower as pp

from os.path import dirname, abspath
from powergrid.core import *
from powergrid.networks.lines import *
from powergrid.envs.multiagent.base import GridEnv, NetworkedGridEnv

from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus

dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
data_dir = os.path.join(dir, 'data', 'data2024.pkl')
with open(data_dir, 'rb') as file:
    dataset = pickle.load(file)


def read_data(d, load_area, renew_area):
    return {
        'load' : d['load'][load_area],
        'solar': d['solar'][renew_area],
        'wind' : d['wind'][renew_area],
        'price': d['price']['LMP']
    }

class MultiAgentMicrogrids(NetworkedGridEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.max_episode_steps = 24
        self.data_size =  self.possible_agents['DSO'].dataset['price'].size
        self.total_days = self.data_size // self.max_episode_steps

    def _build_net(self):
        net = IEEE34Bus('DSO') # main grid
        dso = GridEnv(net, load_scale=0.1, base_power=3)
        dso.add_dataset(read_data(dataset, 'BANC', 'NP15'))
        
        mg1 = GridEnv(IEEE13Bus('MG1'), load_scale=0.1, base_power=3)
        mg1_ess1 = ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5, max_e_mwh=2, min_e_mwh=0.2)
        mg1_dg1 = DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.66, sn_mva=1.0, cost_curve_coefs=[100, 72.4, 0.5011])
        mg1_pv1 = DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar')
        mg1_wt1 = DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind')
        mg1.add_storage(mg1_ess1)
        mg1.add_sgen([mg1_dg1, mg1_pv1, mg1_wt1])
        mg1.add_dataset(read_data(dataset, 'AVA', 'NP15'))
        net = mg1.add_to(net, 'DSO Bus 822')

        mg2 = GridEnv(IEEE13Bus('MG2'), load_scale=0.1, base_power=3)
        mg2_ess1 = ESS('ESS1', bus='Bus 645', min_p_mw=-0.5, max_p_mw=0.5, max_e_mwh=2, min_e_mwh=0.2)
        mg2_dg1 = DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.60, cost_curve_coefs=[100, 51.6, 0.4615])
        mg2_pv1 = DG('PV1', bus='Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar')
        mg2_wt1 = DG('WT1', bus='Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind')
        mg2.add_storage(mg2_ess1)
        mg2.add_sgen([mg2_dg1, mg2_pv1, mg2_wt1])
        mg2.add_dataset(read_data(dataset, 'BANCMID', 'NP15'))
        net = mg2.add_to(net, 'DSO Bus 848')

        mg3 = GridEnv(IEEE13Bus('MG3'), load_scale=0.1, base_power=3)
        mg3_ess1 = ESS('ESS1', 'Bus 645', min_p_mw=-0.5, max_p_mw=0.5, max_e_mwh=2, min_e_mwh=0.2)
        mg3_dg1 = DG('DG1', 'Bus 675', min_p_mw=0, max_p_mw=0.50, cost_curve_coefs=[100, 51.6, 0.4615])
        mg3_pv1 = DG('PV1', 'Bus 652', min_p_mw=0, max_p_mw=0.1, type='solar')
        mg3_wt1 = DG('WT1', 'Bus 645', min_p_mw=0, max_p_mw=0.1, type='wind')
        mg3.add_storage(mg3_ess1)
        mg3.add_sgen([mg3_dg1, mg3_pv1, mg3_wt1])
        mg3.add_dataset(read_data(dataset, 'AZPS', 'NP15'))
        net = mg3.add_to(net, 'DSO Bus 856')
        pp.runpp(net)

        self.net = net
        self.possible_agents = {a.name:a for a in [dso, mg1, mg2, mg3]}
        self.agents = self.possible_agents

    def _reward_and_safety(self):
        if self.net["converged"]:
            # reward and safety
            rewards = {n: -a.cost for n, a in self.agents.items()}
            safety = {n: a.safety for n, a in self.agents.items()}
        else:
            rewards = {n: -200.0 for n in self.agents}
            safety = {n: 20 for n in self.agents}
            # print('Doesn\'t converge!')

        if self.env_config.get('penalty'):
            for name in self.agents:
                rewards[name] -= safety[name]*self.env_config.get('penalty')

        return rewards, safety


if __name__ == '__main__':
    env_config = {
    "train": True,
    "penalty": 10,
    "share_reward": True,
    }
    env = MultiAgentMicrogrids(env_config)
    obs, info = env.reset()