"""Grid-level coordinator agents for hierarchical control.

GridAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.
"""


from typing import Dict, Iterable, Iterable, List, Any, Optional
import numpy as np
import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete

from .base import Agent, Observation, Message, AgentID
from ..core.policies import Policy
from .device_agent import DeviceAgent
from ..devices.generator import RES
from ..devices.storage import ESS
from ..core.protocols import VerticalProtocol, NoProtocol, Protocol


class GridAgent(Agent):

    def __init__(
        self,
        agent_id: AgentID,
        devices: List[DeviceAgent],
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        centralized: bool = True,
    ):
        """Initialize grid coordinator.

        Args:
            agent_id: Unique identifier
            devices: List of device agents to coordinate
            protocol: Protocol for coordinating devices (agent-owned)
            policy: High-level policy (optional)
            centralized: If True, outputs single action for all devices
        """
        # Temporarily set devices for space building
        self.devices = {agent.agent_id: agent for agent in devices}

        super().__init__(
            agent_id=agent_id,
            level=2,  # Grid level
        )
        self.protocol = protocol
        self.policy = policy
        self.centralized = centralized

    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Collect observations from device agents.

        Args:
            global_state: Environment state

        Returns:
            Aggregated observation from all devices
        """
        obs = Observation(
            timestamp=self._timestep,
            messages=self.mailbox.copy()
        )

        # Collect device observations
        device_obs = {}
        for agent_id, agent in self.devices.items():
            device_obs[agent_id] = agent.observe(global_state)

        # Aggregate local state
        obs.local = self.build_local_observation(device_obs, *args, **kwargs)

        # Aggregate global info (take from first device)            
        # TODO: update global info aggregation if needed
        obs.global_info = global_state

        return obs
    
    def build_local_observation(self, device_obs: Dict[AgentID, Observation]) -> Any:
        return {"device_obs": device_obs}

    def act(self, observation: Observation, given_action: Any=None) -> Any:
        """Compute coordination action and distribute to devices.

        Args:
            observation: Aggregated observation
            give_action: Pre-computed action (if any)

        Returns:
            Coordinator action (or None if decentralized)
        """
        # Get coordinator action from policy if available
        if given_action:
            action = given_action
        elif self.centralized:
            assert self.policy is not None, "GridAgent requires a policy to compute actions."
            # this is actual action computation
            action = self.policy.forward(observation)
        else:
            # TODO: this is coordinator action computation
            # Non-centralized GridAgent using coordination_policy to coordinate devices
            # to compute their actions individually
            # Afterwards, GridAgent can also send messages to devices if needed
            raise NotImplementedError("")
        
        self.coordinate_device(observation, action)
        return action

    def coordinate_device(self, observation: Observation, action: Any) -> None:
        self.protocol.coordinate_action(self.devices, observation, action)
        self.protocol.coordinate_message(self.devices, observation, action)

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset coordinator and all devices.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)

        # Reset devices
        for agent in self.devices.values():
            agent.reset(seed=seed, **kwargs)

        # Reset policy
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_device_actions(
        self,
        observations: Dict[AgentID, Observation],
    ) -> Dict[AgentID, Any]:
        # TODO: use this function to get device actions in decentralized setting
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.devices[agent_id].act(obs)
        return actions

    def __repr__(self) -> str:
        num_subs = len(self.devices)
        protocol_name = self.protocol.__class__.__name__
        return f"GridAgent(id={self.agent_id}, devices={num_subs}, protocol={protocol_name})"


class PowerGridAgent(GridAgent):
    def __init__(
        self,
        net,
        grid_config: Dict[str, Any],
        *,
        # Base class args
        devices: List[DeviceAgent],
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        centralized: bool = False,
    ):
        self.net = net
        self.name = net.name
        self.config = grid_config
        self.sgen: Dict[str, RES] = {}
        self.storage: Dict[str, ESS] = {}
        self.base_power = grid_config.get("base_power", 1)
        self.load_scale = grid_config.get("load_scale", 1)
        self.load_rescaling(net, self.load_scale)

        super().__init__(
            agent_id=self.name,
            devices=devices,
            protocol=protocol,
            policy=policy,
            centralized=centralized,
        )

    def add_dataset(self, dataset):
        self.dataset = dataset

    def fuse_buses(self, ext_net, bus_name):
        self.net.ext_grid.in_service = False
        net, index = pp.merge_nets(
            ext_net, 
            self.net, 
            validate=False, 
            return_net2_reindex_lookup=True 
        )
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
            self.devices[sgen.name] = sgen

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
            self.devices[ess.name] = ess

    def load_rescaling(self, net, scale):
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_load_ids, 'scaling'] *= scale


    def build_local_observation(self, device_obs: Dict[AgentID, Observation], net) -> Any:
        local = super().build_local_observation(device_obs)
        local['state'] = self._get_obs(net, device_obs)
        return local
    
    def _get_obs(self, net, device_obs = None):
        if device_obs is None:
            device_obs = {
                agent_id: agent.observe()
                for agent_id, agent in self.devices.items()
            }
        obs = np.array([])
        for _, ob in device_obs.items():
            # P, Q, SoC of energy storage units
            # P, Q, UC status of generators
            obs = np.concatenate((obs, ob.local['state']))
        # P, Q at all buses
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        load_pq = net.res_load.iloc[local_load_ids].values
        obs = np.concatenate([obs, load_pq.ravel() / self.base_power])
        return obs.astype(np.float32)
    
    def get_device_action_spaces(self) -> Dict[str, gym.Space]:
        return {
            device.agent_id: device.action_space
            for device in self.devices.values()
        }
    
    def get_grid_action_space(self):
        low, high, discrete_n = [], [], []
        for sp in self.get_device_action_spaces().values():
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

    def get_grid_observation_space(self, net):
        return Box(
            low=-np.inf, 
            high=np.inf, 
            shape=self._get_obs(net).shape, 
            dtype=np.float32
        )
    
    def update_state(self, net, t):
        load_scaling = self.dataset['load'][t]
        solar_scaling = self.dataset['solar'][t]
        wind_scaling = self.dataset['wind'][t]

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
            scaling = solar_scaling if dg.type == 'solar' else wind_scaling
            dg.update_state(scaling)
            local_ids = pp.get_element_index(net, 'sgen', self.name+' '+name)
            states = ['p_mw', 'q_mvar', 'in_service']
            values = [dg.state.P, dg.state.Q, bool(dg.state.on)]
            net.sgen.loc[local_ids, states] = values

    def update_cost_safety(self, net):
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