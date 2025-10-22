"""Grid-level coordinator agents for hierarchical control.

GridAgent manages a set of device agents, implementing coordination
protocols like price signals, setpoints, or consensus algorithms.
"""

from typing import Any, Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from powergrid.agents.base import Agent, AgentID, Observation
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.policies import Policy
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.devices.generator import RES
from powergrid.devices.storage import ESS


class GridAgent(Agent):
    """Grid-level coordinator for managing device agents.

    GridAgent coordinates multiple device agents using specified protocols
    and optionally a centralized policy for joint decision-making.

    Attributes:
        devices: Dictionary mapping device agent IDs to DeviceAgent instances
        protocol: Coordination protocol for managing subordinate devices
        policy: Optional centralized policy for joint action computation
        centralized: If True, uses centralized policy; if False, devices act independently
    """

    def __init__(
        self,
        agent_id: AgentID,
        devices: List[DeviceAgent] = [],
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

    # Core agent lifecycle methods
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

    def build_local_observation(self, device_obs: Dict[AgentID, Observation], *args, **kwargs) -> Any:
        """Build local observation from device observations.

        Args:
            device_obs: Dictionary mapping device IDs to their observations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated local observation dictionary
        """
        return {"device_obs": device_obs}

    def act(self, observation: Observation, given_action: Any = None) -> Any:
        """Compute coordination action and distribute to devices.

        Args:
            observation: Aggregated observation
            given_action: Pre-computed action (if any)

        Returns:
            Coordinator action (or None if decentralized)

        Raises:
            NotImplementedError: If using decentralized mode (not yet implemented)
        """
        # Get coordinator action from policy if available
        if given_action:
            action = given_action
        elif self.centralized:
            assert self.policy is not None, "GridAgent requires a policy to compute actions."
            # This is actual action computation
            action = self.policy.forward(observation)
        else:
            # TODO: this is coordinator action computation
            # Non-centralized GridAgent using coordination_policy to coordinate devices
            # to compute their actions individually
            # Afterwards, GridAgent can also send messages to devices if needed
            raise NotImplementedError("Decentralized coordination not yet implemented")

        self.coordinate_device(observation, action)
        return action

    # Coordination methods
    def coordinate_device(self, observation: Observation, action: Any) -> None:
        """Coordinate device actions using the protocol.

        Args:
            observation: Current observation
            action: Computed action from coordinator
        """
        self.protocol.coordinate_action(self.devices, observation, action)
        self.protocol.coordinate_message(self.devices, observation, action)

    def get_device_actions(
        self,
        observations: Dict[AgentID, Observation],
    ) -> Dict[AgentID, Any]:
        """Get actions from all devices in decentralized mode.

        Args:
            observations: Dictionary mapping device IDs to observations

        Returns:
            Dictionary mapping device IDs to their computed actions

        Note:
            This function is intended for decentralized coordination where
            each device computes its own action independently.
        """
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.devices[agent_id].act(obs)
        return actions

    # Utility methods
    def __repr__(self) -> str:
        num_subs = len(self.devices)
        protocol_name = self.protocol.__class__.__name__
        return f"GridAgent(id={self.agent_id}, devices={num_subs}, protocol={protocol_name})"


class PowerGridAgent(GridAgent):
    """Grid agent for power system coordination with PandaPower integration.

    PowerGridAgent extends GridAgent with power system-specific functionality,
    including PandaPower network integration, device management, and state updates.

    Attributes:
        net: PandaPower network object
        name: Grid name (from network)
        config: Grid configuration dictionary
        sgen: Dictionary of renewable energy sources (RES)
        storage: Dictionary of energy storage systems (ESS)
        base_power: Base power for normalization (MW)
        load_scale: Scaling factor for loads
    """

    def __init__(
        self,
        net,
        grid_config: Dict[str, Any],
        *,
        # Base class args
        devices: List[DeviceAgent] = [],
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        centralized: bool = False,
    ):
        """Initialize power grid agent.

        Args:
            net: PandaPower network object
            grid_config: Grid configuration dictionary
            devices: List of device agents to coordinate
            protocol: Coordination protocol
            policy: Optional centralized policy
            centralized: If True, uses centralized control
        """
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

    # Network setup methods
    def add_dataset(self, dataset):
        """Add time-series dataset for loads and renewables.

        Args:
            dataset: Dictionary containing 'load', 'solar', 'wind' time series
        """
        self.dataset = dataset

    def add_sgen(self, sgens):
        """Add renewable generators (solar/wind) to the network.

        Args:
            sgens: Single RES instance or iterable of RES instances
        """
        if not isinstance(sgens, Iterable):
            sgens = [sgens]

        for sgen in sgens:
            bus_id = pp.get_element_index(self.net, 'bus', self.name + ' ' + sgen.bus)
            pp.create_sgen(self.net, bus_id, p_mw=sgen.state.P, sn_mva=sgen.sn_mva,
                          index=len(self.sgen), name=self.name + ' ' + sgen.name,
                          max_p_mw=sgen.max_p_mw, min_p_mw=sgen.min_p_mw,
                          max_q_mvar=sgen.max_q_mvar, min_q_mvar=sgen.min_q_mvar)
            self.sgen[sgen.name] = sgen
            self.devices[sgen.name] = sgen

    def add_storage(self, storages):
        """Add energy storage systems to the network.

        Args:
            storages: Single ESS instance or iterable of ESS instances
        """
        if not isinstance(storages, Iterable):
            storages = [storages]

        for ess in storages:
            bus_id = pp.get_element_index(self.net, 'bus', self.name + ' ' + ess.bus)
            pp.create_storage(self.net, bus_id, ess.state.P, ess.max_e_mwh,
                            sn_mva=ess.sn_mva, soc_percent=ess.state.soc,
                            min_e_mwh=ess.min_e_mwh, name=self.name + ' ' + ess.name,
                            index=len(self.storage), max_p_mw=ess.max_p_mw,
                            min_p_mw=ess.min_p_mw, max_q_mvar=ess.max_q_mvar,
                            min_q_mvar=ess.min_q_mvar)
            self.storage[ess.name] = ess
            self.devices[ess.name] = ess

    def fuse_buses(self, ext_net, bus_name):
        """Merge this grid with an external network by fusing buses.

        Args:
            ext_net: External PandaPower network
            bus_name: Name of bus to fuse with external grid

        Returns:
            Merged PandaPower network
        """
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

    def load_rescaling(self, net, scale):
        """Apply scaling factor to local loads.

        Args:
            net: PandaPower network
            scale: Scaling multiplier
        """
        local_load_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_load_ids, 'scaling'] *= scale

    # Observation methods
    def build_local_observation(self, device_obs: Dict[AgentID, Observation], net) -> Any:
        """Build local observation including device states and network results.

        Args:
            device_obs: Device observations dictionary
            net: PandaPower network with power flow results

        Returns:
            Local observation dictionary with device and network state
        """
        local = super().build_local_observation(device_obs)
        local['state'] = self._get_obs(net, device_obs)
        return local

    def _get_obs(self, net, device_obs=None):
        """Extract numerical observation vector from network state.

        Args:
            net: PandaPower network
            device_obs: Optional device observations (computed if not provided)

        Returns:
            Flattened observation array (float32)
        """
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

    # Space construction methods
    def get_device_action_spaces(self) -> Dict[str, gym.Space]:
        """Get action spaces for all devices.

        Returns:
            Dictionary mapping device IDs to their action spaces
        """
        return {
            device.agent_id: device.action_space
            for device in self.devices.values()
        }

    def get_grid_action_space(self):
        """Construct combined action space for all devices.

        Returns:
            Gymnasium space representing joint action space of all devices
        """
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
            # Mixed continuous and discrete (not yet supported, should use Dict space)
            raise NotImplementedError("Mixed action spaces not yet supported")
        elif len(low):  # Continuous only
            return Box(low=low, high=high, dtype=np.float32)
        elif len(discrete_n):  # Discrete only
            return MultiDiscrete(discrete_n)
        else:  # No actionable agents
            return Discrete(1)

    def get_grid_observation_space(self, net):
        """Get observation space for this grid.

        Args:
            net: PandaPower network

        Returns:
            Gymnasium Box space for grid observations
        """
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self._get_obs(net).shape,
            dtype=np.float32
        )

    # State update methods
    def update_state(self, net, t):
        """Update grid state from dataset and device actions.

        Args:
            net: PandaPower network to update
            t: Timestep index in dataset
        """
        load_scaling = self.dataset['load'][t]
        solar_scaling = self.dataset['solar'][t]
        wind_scaling = self.dataset['wind'][t]

        local_ids = pp.get_element_index(net, 'load', self.name, False)
        net.load.loc[local_ids, 'scaling'] = load_scaling
        self.load_rescaling(net, self.load_scale)

        for name, ess in self.storage.items():
            ess.update_state()
            local_ids = pp.get_element_index(net, 'storage', self.name + ' ' + name)
            states = ['p_mw', 'q_mvar', 'soc_percent', 'in_service']
            values = [ess.state.P, ess.state.Q, ess.state.soc, bool(ess.state.on)]
            net.storage.loc[local_ids, states] = values

        for name, dg in self.sgen.items():
            scaling = solar_scaling if dg.type == 'solar' else wind_scaling
            dg.update_state(scaling)
            local_ids = pp.get_element_index(net, 'sgen', self.name + ' ' + name)
            states = ['p_mw', 'q_mvar', 'in_service']
            values = [dg.state.P, dg.state.Q, bool(dg.state.on)]
            net.sgen.loc[local_ids, states] = values

    def update_cost_safety(self, net):
        """Update cost and safety metrics for the grid.

        Args:
            net: PandaPower network with power flow results
        """
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
