"""
DEPRECATED: This module contains legacy multi-agent environment classes.

This module will be removed in v3.0. Please use the new architecture instead:
- powergrid.envs.multi_agent.MultiAgentPowerGridEnv (PettingZoo-based)
- powergrid.agents.GridAgent (hierarchical coordinator)
- powergrid.agents.DeviceAgent (device-level agents)

See docs/ARCHITECTURE_COMPARISON.md for migration guide.
"""

import warnings
import numpy as np
import pandapower as pp
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium.utils.seeding as seeding
from collections.abc import Iterable
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict
from ray.rllib.models.preprocessors import get_preprocessor

warnings.warn(
    "powergrid.envs.multiagent.base is deprecated and will be removed in v3.0. "
    "Use powergrid.envs.multi_agent.MultiAgentPowerGridEnv instead. "
    "See docs/ARCHITECTURE_COMPARISON.md for details.",
    FutureWarning,
    stacklevel=2
)

''' Grid Environment
'''
class GridEnv:
    """
    DEPRECATED: Use GridAgent instead.

    This class wraps a pandapower network with devices but is NOT a proper RL agent.
    Use powergrid.agents.GridAgent for the new hierarchical agent architecture.
    """
    def __init__(self, net, **kwargs):
        warnings.warn(
            "GridEnv is deprecated. Use powergrid.agents.GridAgent instead.",
            FutureWarning,
            stacklevel=2
        )
        self.net = net
        self.name = net.name
        self.kwargs = kwargs
        self.sgen = {}
        self.storage = {}
        self.base_power = kwargs.get('base_power', 1)
        self.load_scale = kwargs.get('load_scale', 1)
        self.load_rescaling(net, self.load_scale)

    def add_dataset(self, dataset):
        self.dataset = dataset

    def add_to(self, ext_net, bus_name):
        self.net.ext_grid.in_service = False
        net, index = pp.merge_nets(ext_net, self.net, validate=False, 
                                       return_net2_reindex_lookup=True)
        substation = pp.get_element_index(net, 'bus', bus_name)
        ext_grid = index['bus'][self.net.ext_grid.bus.values[0]]
        pp.fuse_buses(net, ext_grid, substation)

        return net

    def add_sgen(self, sgens):
        if not isinstance(sgens, Iterable):
            sgens = [sgens]

        for sgen in sgens:
            bus_id = pp.get_element_index(self.net, 'bus', self.name+' '+sgen.bus)
            pp.create_sgen(self.net, bus_id, p_mw=sgen.state.P, sn_mva=sgen.sn_mva, 
                        index=len(self.sgen), name=self.name+' '+sgen.name, 
                        max_p_mw=sgen.max_p_mw, min_p_mw=sgen.min_p_mw, 
                        max_q_mvar=sgen.max_q_mvar, min_q_mvar=sgen.min_q_mvar)
            self.sgen[sgen.name] = sgen

    def add_storage(self, storages):
        if not isinstance(storages, Iterable):
            storages = [storages]
        
        for ess in storages:
            bus_id = pp.get_element_index(self.net, 'bus', self.name+' '+ess.bus)
            pp.create_storage(self.net, bus_id, ess.state.P, ess.max_e_mwh, 
                            sn_mva=ess.sn_mva, soc_percent=ess.state.soc,
                            min_e_mwh=ess.min_e_mwh, name=self.name+' '+ess.name, 
                            index=len(self.storage), max_p_mw=ess.max_p_mw, 
                            min_p_mw=ess.min_p_mw, max_q_mvar=ess.max_q_mvar, 
                            min_q_mvar=ess.min_q_mvar)
            self.storage[ess.name] = ess

    def load_rescaling(self, net, scale):
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_load_ids, 'scaling'] *= scale

    def step(self, net, action, t):
        self._set_action(action)
        self._update_state(net, t)

    def reset(self, net, t, rng=None):
        for ess in self.storage.values():
            ess.reset(rng)
        self._update_state(net, t)

    def _get_obs(self, net):
        obs = np.array([])
        # P, Q, SoC of energy storage units
        for ess in self.storage.values():
            obs = np.concatenate([obs, ess.state.get()])
        # P, Q, UC status of generators
        for dg in self.sgen.values():
            obs = np.concatenate([obs, dg.state.get()])
        # P, Q at all buses
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        load_pq = net.res_load.iloc[local_load_ids].values
        obs = np.concatenate([obs, load_pq.ravel() / self.base_power])
        
        return obs.astype(np.float32)

    def _set_action(self, action):
        devices = list(self.storage.values()) + list(self.sgen.values())
        for dev in devices:
            # continuous actions
            dev.action.c[:] = action[:dev.action.c.size]
            if self.kwargs.get('discrete_action'):
                cats = self.kwargs.get('discrete_action_cats')
                low, high = dev.action.range
                acts = np.linspace(low, high, cats).transpose()
                dev.action.c[:] = [a[action[i]] for i, a in enumerate(acts)]
            action = action[dev.action.c.size:]
            # discrete action space
            dev.action.d[:] = action[:dev.action.d.size]
            action = action[dev.action.d.size:]

    def _get_action_space(self):
        devices = list(self.storage.values()) + list(self.sgen.values())
        ac_space = dict()
        for dev in devices:
            # continuous action space
            if dev.action.c.size > 0:
                low, high = dev.action.range
                ac_space[dev.name] = Box(low=low, high=high, dtype=np.float32)
                if self.kwargs.get('discrete_action'):
                    cats = self.kwargs.get('discrete_action_cats')
                    if low.size == 1:
                        ac_space[dev.name] = Discrete(cats)
                    else:
                        ac_space[dev.name] = MultiDiscrete([cats]*low.size)
            # discrete action space
            if dev.action.d.size > 0:
                ac_space[dev.name] = Discrete(dev.action.ncats)

        return ac_space

    def _get_observation_space(self, net):
        return Box(
            low=-np.inf, 
            high=np.inf, 
            shape=self._get_obs(net).shape, 
            dtype=np.float32)

    def _combined_action_space(self):
        low, high, discrete_n = [], [], []
        for sp in self._get_action_space().values():
            if isinstance(sp, Box):
                low = np.append(low, sp.low)
                high = np.append(high, sp.high)
            elif isinstance(sp, Discrete):
                discrete_n.append(sp.n)
            elif isinstance(sp, MultiDiscrete):
                discrete_n.extend(list(sp.nvec))

        if len(low) and len(discrete_n):
            raise Dict({"continuous": Box(low=low, high=high, dtype=np.float32),
                        'discrete': MultiDiscrete(discrete_n)})
        elif len(low): # continuous
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n): # discrete
            return MultiDiscrete(discrete_n)
        else: # non actionable agents
            return Discrete(1)

    def _update_state(self, net, t):
        load_scaling = self.dataset['load'][t]
        solar_scaling = self.dataset['solar'][t]
        wind_sclaing = self.dataset['wind'][t]

        local_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_ids, 'scaling'] = load_scaling
        self.load_rescaling(net, self.load_scale)

        for name, ess in self.storage.items():
            ess.update_state()
            local_ids = pp.get_element_index(net, 'storage', self.name+' '+name)
            states = ['p_mw', 'q_mvar', 'soc_percent', 'in_service']
            values = [ess.state.P, ess.state.Q, ess.state.soc, bool(ess.state.on)]
            net.storage.loc[local_ids, states] = values

        for name, dg in self.sgen.items():
            scaling = solar_scaling if dg.type == 'solar' else wind_sclaing
            dg.update_state(scaling)
            local_ids = pp.get_element_index(net, 'sgen', self.name+' '+name)
            states = ['p_mw', 'q_mvar', 'in_service']
            values = [dg.state.P, dg.state.Q, bool(dg.state.on)]
            net.sgen.loc[local_ids, states] = values

    def _update_cost_safety(self, net):
        self.cost, self.safety = 0, 0
        for ess in self.storage.values():
            ess.update_cost_safety()
            self.cost += ess.cost
            self.safety += ess.safety
        
        for dg in self.sgen.values():
            dg.update_cost_safety()
            self.cost += dg.cost
            self.safety += dg.safety

        if net["converged"]:
            local_bus_ids = pp.get_element_index(net, 'bus', self.name, False)
            local_vm = net.res_bus.loc[local_bus_ids].vm_pu.values
            overvoltage = np.maximum(local_vm - 1.05, 0).sum()
            undervoltage = np.maximum(0.95 - local_vm, 0).sum()

            local_line_ids = pp.get_element_index(net, 'line', self.name, False)
            local_line_loading = net.res_line.loc[local_line_ids].loading_percent.values
            overloading = np.maximum(local_line_loading - 100, 0).sum() * 0.01
            
            self.safety += overloading + overvoltage + undervoltage


"""
Networked Power Grid Environment
"""
from abc import abstractmethod

class NetworkedGridEnv(MultiAgentEnv):
    """
    DEPRECATED: Use MultiAgentPowerGridEnv instead.

    This class is based on RLlib's MultiAgentEnv and uses GridEnv pseudo-agents.
    Use powergrid.envs.multi_agent.MultiAgentPowerGridEnv (PettingZoo-based) instead.
    """
    def __init__(self, env_config):
        warnings.warn(
            "NetworkedGridEnv is deprecated. Use MultiAgentPowerGridEnv instead.",
            FutureWarning,
            stacklevel=2
        )
        self.env_config = env_config
        self.train = env_config.get('train', True)
        self.type = env_config.get('type', 'AC')
        self._build_net()
        self._init_space()

    @property
    def actionable_agents(self):
        return {n : a for n, a in self.possible_agents.items() 
                if len(a._get_action_space()) > 0}

    @abstractmethod
    def _build_net(self):
        pass

    @abstractmethod
    def _reward_and_safety(self):
        pass

    def step(self, action_n):
        # set action for each agent
        for name, action in action_n.items():
            if name in self.actionable_agents:
                if self.env_config.get('share_policy'):
                    action = action[name]
                self.actionable_agents[name].step(self.net, action, self._t)
        # run power flow for the whole network
        try:
            pp.runpp(self.net)
        except:
            pass
        # update and get reward
        for agent in self.agents.values():
            agent._update_cost_safety(self.net)
        rewards, safety = self._reward_and_safety()
        if self.env_config.get('share_reward'):
            shared_reward = np.mean(list(rewards.values()))
            rewards = {name: shared_reward for name in self.agents}
        # timestep counter
        self._t = self._t + 1 if self._t < self.data_size else 0
        # done
        done = self._t % self.max_episode_steps == 0
        terminateds = {"__all__": done}
        truncateds = {"__all__": done}
        # info
        infos = safety

        return self._get_obs(), rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset all agents
        if self.train:
            self._day = self.np_random.integers(self.total_days - 1)
            self._t = self._day * self.max_episode_steps
            for agent in self.agents.values():
                agent.reset(self.net, self.max_episode_steps, self.np_random)
        else:
            if hasattr(self, '_t'):
                self._day += 1
            else:
                self._t, self._day = 0, 0

        info = {}
        try:
            pp.runpp(self.net)
        except:
            print("===============================================================================================")

        return self._get_obs(), info

    def _get_obs(self):
        obs_dict = {n: a._get_obs(self.net) for n, a in self.agents.items()}

        if self.env_config.get('share_policy'):
            shared_obs_dict = {}
            for agent in self.agents:
                obs = {name: np.zeros_like(obs) if agent != name else obs
                       for name, obs in obs_dict.items()}
                shared_obs = self._flattened_obs_spaces.transform(obs)
                shared_obs_dict[agent] = shared_obs.astype(np.float32)

            return shared_obs_dict

        return obs_dict

    # def shared_observation_spaces(self):

    #     observation_spaces = {}
    #     for name, agent in self.agents.items():
    #         ob_space = agent._get_observation_space(self.net)
    #         self.observation_spaces[name] = ob_space
    #         if isinstance(space, Box):
    #             obs.append(np.zeros(space.shape))
    #         elif isinstance(space, Discrete):
    #             obs.append(np.zeros((1,)))
    #         elif isinstance(space, MultiDiscrete):
    #             obs.append(np.zeros(space.nvec.shape))
    #         else:
    #             raise NotImplementedError("Dict space is not supported.")

    def _init_space(self):
        # self.neighbor_mask = np.zeros((self.num_agents, self.num_agents)).astype(int)
        # self.distance_mask = np.zeros((self.num_agents, self.num_agents)).astype(int)
        # cur_distance = list(range(self.n_agent))
        # for i in range(self.n_agent):
        #     self.distance_mask[i] = cur_distance
        #     cur_distance = [i+1] + cur_distance[:-1]
        #     if i >= 1:
        #         self.neighbor_mask[i,i-1] = 1
        #     if i <= self.n_agent-2:
        #         self.neighbor_mask[i,i+1] = 1
        
        ac_spaces = {}
        ob_spaces = {}
        for name, agent in self.agents.items():
            ac_spaces[name] = agent._combined_action_space()
            ob_spaces[name] = agent._get_observation_space(self.net)

        if self.env_config.get('share_policy'):
            shared_ac_space = Dict(ac_spaces)
            shared_ob_space = Dict(ob_spaces)
            self._flattened_obs_spaces = get_preprocessor(shared_ob_space)(shared_ob_space)
            for name in self.agents.keys():
                ac_spaces[name] = shared_ac_space
                ob_shape = self._flattened_obs_spaces.observation_space.shape
                ob_spaces[name] = Box(-np.inf, np.inf, shape=ob_shape)

        self.action_spaces = ac_spaces
        self.observation_spaces = ob_spaces
