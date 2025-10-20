"""
PettingZoo-compatible multi-agent environment for power grid control.

This environment supports:
- Hierarchical control with GridAgents (microgrid controllers)
- Vertical protocols (GridAgent → DeviceAgent coordination)
- Horizontal protocols (GridAgent ↔ GridAgent peer coordination)
- Flexible configuration via dict-based config
"""

from pettingzoo import ParallelEnv
import pandapower as pp
import numpy as np
from typing import Dict, Any, List, Optional
import gymnasium as gym
from gymnasium.spaces import Box

from powergrid.agents import GridAgent, DeviceAgent
from powergrid.agents.protocols import (
    VerticalProtocol, HorizontalProtocol,
    PriceSignalProtocol, SetpointProtocol, NoProtocol,
    PeerToPeerTradingProtocol, ConsensusProtocol, NoHorizontalProtocol
)
from powergrid.agents.base import Message


class MultiAgentPowerGridEnv(ParallelEnv):
    """
    PettingZoo-compatible environment for multi-agent microgrid control.

    Key Features:
    - GridAgents as primary RL-controllable agents (microgrid controllers)
    - DeviceAgents managed internally by GridAgents
    - Vertical protocols (agent-owned) for parent → child coordination
    - Horizontal protocols (environment-owned) for peer ↔ peer coordination
    - Extensible level-based architecture

    Example:
        config = {
            'microgrids': [
                {
                    'name': 'MG1',
                    'network': IEEE13Bus('MG1'),
                    'devices': [ESS(...), DG(...), RES(...)],
                    'vertical_protocol': 'price_signal',
                    'dataset': {...}
                },
                ...
            ],
            'horizontal_protocol': 'p2p_trading',
            'episode_length': 24,
            'train': True
        }

        env = MultiAgentPowerGridEnv(config)
        obs, info = env.reset()

        for _ in range(24):
            actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
            obs, rewards, dones, truncated, infos = env.step(actions)
    """

    metadata = {"name": "multi_agent_powergrid"}

    def __init__(self, config: Dict):
        """
        Initialize multi-agent environment.

        Args:
            config: Configuration dictionary with structure:
                {
                    'network': Base network (e.g., IEEE34Bus('DSO')),
                    'microgrids': [
                        {
                            'name': str,  # Agent ID
                            'network': pandapower network,
                            'devices': [Device, ...],
                            'vertical_protocol': 'price_signal' | 'setpoint' | 'none',
                            'dataset': {'load': array, 'solar': array, ...}
                        },
                        ...
                    ],
                    'horizontal_protocol': 'p2p_trading' | 'consensus' | 'none',
                    'topology': {'adjacency': {aid: [neighbor_aids, ...]}},  # Optional
                    'episode_length': int,
                    'base_power': float,
                    'train': bool,
                    'penalty': float,
                    'share_reward': bool
                }
        """
        super().__init__()

        self.config = config
        self.episode_length = config.get('episode_length', 24)
        self.base_power = config.get('base_power', 1.0)
        self.train = config.get('train', True)
        self.penalty = config.get('penalty', 10)
        self.share_reward = config.get('share_reward', True)
        self.timestep = 0

        # Build network and agents
        self.net = self._build_network(config)
        self.agents = self._build_grid_agents(config['microgrids'])

        # Build horizontal protocol (environment-owned)
        self.horizontal_protocol = self._build_horizontal_protocol(
            config.get('horizontal_protocol', 'none')
        )

        # PettingZoo API requirements
        self.possible_agents = list(self.agents.keys())
        self._agent_ids = self.possible_agents.copy()

        # Action/observation spaces per agent
        self.action_spaces = {
            aid: agent.action_space
            for aid, agent in self.agents.items()
        }

        # Fix observation spaces to match actual observations
        # Get a sample observation to determine actual size
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        sample_obs = self._get_observations()
        self.observation_spaces = {
            aid: Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            for aid, obs in sample_obs.items()
        }

    def _build_network(self, config: Dict) -> pp.pandapowerNet:
        """
        Build merged pandapower network from all microgrids.

        Args:
            config: Full config dict

        Returns:
            Merged pandapower network
        """
        # Get base network or create empty
        if 'network' in config and config['network'] is not None:
            main_net = config['network']
        else:
            # Create empty network
            main_net = pp.create_empty_network()

        # Merge each microgrid into main network
        for mg_cfg in config['microgrids']:
            mg_net = mg_cfg['network']

            # Add devices to microgrid network (create PP elements)
            self._add_devices_to_network(mg_net, mg_cfg['name'], mg_cfg['devices'])

            # Merge into main network
            if len(main_net.bus) > 0:
                # TODO: Implement merge logic (connect at PCC bus)
                main_net, _ = pp.merge_nets(main_net, mg_net, validate=False,
                                             return_net2_reindex_lookup=True)
            else:
                main_net = mg_net

        return main_net

    def _add_devices_to_network(
        self,
        net: pp.pandapowerNet,
        mg_name: str,
        devices: List
    ) -> None:
        """
        Add pandapower elements for each device.

        Args:
            net: Pandapower network
            mg_name: Microgrid name (prefix for element names)
            devices: List of Device objects
        """
        for device in devices:
            element_name = f"{mg_name}_{device.name}"
            bus_name = f"{mg_name} {device.bus}"
            bus_id = pp.get_element_index(net, 'bus', bus_name)

            if device.__class__.__name__ == 'ESS':
                pp.create_storage(
                    net, bus_id, device.state.P, device.max_e_mwh,
                    sn_mva=device.sn_mva, soc_percent=device.state.soc * 100,
                    min_e_mwh=device.min_e_mwh, name=element_name,
                    max_p_mw=device.max_p_mw, min_p_mw=device.min_p_mw,
                    max_q_mvar=getattr(device, 'max_q_mvar', 0),
                    min_q_mvar=getattr(device, 'min_q_mvar', 0)
                )

            elif device.__class__.__name__ in ['DG', 'RES']:
                pp.create_sgen(
                    net, bus_id, p_mw=device.state.P,
                    sn_mva=getattr(device, 'sn_mva', 1.0),
                    name=element_name,
                    max_p_mw=device.max_p_mw, min_p_mw=device.min_p_mw,
                    max_q_mvar=getattr(device, 'max_q_mvar', 0),
                    min_q_mvar=getattr(device, 'min_q_mvar', 0)
                )

    def _build_grid_agents(self, microgrids_config: List[Dict]) -> Dict[str, GridAgent]:
        """
        Build GridAgent objects from microgrid configs.

        Args:
            microgrids_config: List of microgrid config dicts

        Returns:
            Dict mapping agent_id → GridAgent
        """
        agents = {}

        for mg_cfg in microgrids_config:
            name = mg_cfg['name']

            # Create DeviceAgents for all devices
            device_agents = []
            for device in mg_cfg['devices']:
                # Set device dataset
                if 'dataset' in mg_cfg:
                    device.dataset = mg_cfg['dataset']

                dev_agent = DeviceAgent(
                    agent_id=f"{name}_{device.name}",
                    device=device
                )
                device_agents.append(dev_agent)

            # Create vertical protocol
            v_protocol_name = mg_cfg.get('vertical_protocol', 'none')
            if v_protocol_name == 'price_signal':
                v_protocol = PriceSignalProtocol()
            elif v_protocol_name == 'setpoint':
                v_protocol = SetpointProtocol()
            else:
                v_protocol = NoProtocol()

            # Create GridAgent (microgrid controller)
            grid_agent = GridAgent(
                agent_id=name,
                subordinates=device_agents,
                vertical_protocol=v_protocol,
                centralized=True  # GridAgent outputs joint action for all devices
            )

            agents[name] = grid_agent

        return agents

    def _build_horizontal_protocol(self, protocol_name: str) -> HorizontalProtocol:
        """
        Build horizontal protocol owned by environment.

        Args:
            protocol_name: Protocol type ('p2p_trading', 'consensus', 'none')

        Returns:
            HorizontalProtocol instance
        """
        if protocol_name == 'p2p_trading':
            return PeerToPeerTradingProtocol()
        elif protocol_name == 'consensus':
            return ConsensusProtocol()
        else:
            return NoHorizontalProtocol()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment and all agents.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observations: Dict mapping agent_id → observation array
            infos: Dict mapping agent_id → info dict
        """
        # Don't call super().reset() - ParallelEnv base class raises NotImplementedError
        # We provide our own complete reset implementation

        # Initialize random number generator
        if seed is not None:
            import numpy as np
            self.np_random = np.random.RandomState(seed)
        elif not hasattr(self, 'np_random'):
            import numpy as np
            self.np_random = np.random.RandomState()

        self.timestep = 0

        # Reset all GridAgents (which resets subordinate DeviceAgents)
        for agent in self.agents.values():
            agent.reset(seed=seed)

        # Reset dataset timestep
        if self.train:
            # Random starting day for training
            first_agent = next(iter(self.agents.values()))
            first_device = next(iter(first_agent.subordinates.values())).device
            if hasattr(first_device, 'dataset') and 'load' in first_device.dataset:
                total_steps = len(first_device.dataset['load'])
                total_days = total_steps // self.episode_length
                if total_days > 1:
                    start_day = self.np_random.randint(0, total_days - 1)
                    self.timestep = start_day * self.episode_length

        # Solve initial power flow
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        # Get initial observations
        obs = self._get_observations()
        infos = {aid: {} for aid in self.possible_agents}

        return obs, infos

    def step(self, actions: Dict[str, Any]):
        """
        Execute one environment step with multi-protocol coordination.

        Execution order:
        1. Horizontal coordination (environment-level peer communication)
        2. Vertical coordination (agent-level subordinate coordination)
        3. Action execution (set device actions)
        4. Device state updates
        5. Power flow solve
        6. Cost/safety computation
        7. Reward calculation
        8. Observation collection

        Args:
            actions: Dict mapping agent_id → action

        Returns:
            observations: Dict mapping agent_id → observation array
            rewards: Dict mapping agent_id → reward
            dones: Dict mapping agent_id → done (plus '__all__')
            truncated: Dict mapping agent_id → truncated (plus '__all__')
            infos: Dict mapping agent_id → info dict
        """
        # PHASE 1: Horizontal coordination (environment-level)
        if not isinstance(self.horizontal_protocol, NoHorizontalProtocol):
            obs_dict = {
                aid: agent.observe(self._get_global_state())
                for aid, agent in self.agents.items()
            }

            signals = self.horizontal_protocol.coordinate(
                agents=self.agents,
                observations=obs_dict,
                topology=self.config.get('topology')
            )

            # Deliver signals to agents
            for agent_id, signal in signals.items():
                if signal:
                    self.agents[agent_id].receive_message(
                        Message(sender='MARKET', content=signal, timestamp=self.timestep)
                    )

        # PHASE 2: Vertical coordination (agent-level)
        global_state = self._get_global_state()
        for agent in self.agents.values():
            agent.coordinate_subordinates(global_state)

        # PHASE 3: Action execution
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if agent.centralized:
                self._set_subordinate_actions(agent, action)

        # PHASE 4: Update device states
        self._update_device_states()
        self._sync_to_pandapower()

        # PHASE 5: Power flow
        converged = self._solve_power_flow()

        # PHASE 6: Costs/safety
        self._update_cost_safety(converged)

        # PHASE 7: Rewards
        rewards = self._compute_rewards(converged)

        # PHASE 8: Observations
        obs = self._get_observations()

        # PHASE 9: Termination
        self.timestep += 1
        terminated = (self.timestep % self.episode_length == 0)
        dones = {aid: terminated for aid in self.possible_agents}
        dones['__all__'] = terminated
        truncated = {aid: False for aid in self.possible_agents}
        truncated['__all__'] = False

        # PHASE 10: Info
        infos = {
            aid: {
                'converged': converged,
                'cost': getattr(self.agents[aid], 'cost', 0),
                'safety': getattr(self.agents[aid], 'safety', 0),
                'timestep': self.timestep
            }
            for aid in self.possible_agents
        }

        return obs, rewards, dones, truncated, infos

    def _set_subordinate_actions(self, agent: GridAgent, action: Any) -> None:
        """
        Distribute action vector to subordinate devices.

        Args:
            agent: GridAgent with centralized control
            action: Action array to distribute
        """
        action = np.array(action, dtype=np.float32).flatten()
        idx = 0

        for sub_agent in agent.subordinates.values():
            action_dim = sub_agent.action_space.shape[0]
            sub_action = action[idx:idx + action_dim]
            # Set action on device through device.action.c
            sub_agent.device.action.c = sub_action
            idx += action_dim

    def _update_device_states(self) -> None:
        """Update all device states (dynamics + dataset scalers)."""
        for agent in self.agents.values():
            for sub_agent in agent.subordinates.values():
                device = sub_agent.device
                # Apply dynamics if available
                if hasattr(device, 'step'):
                    device.step()
                # Apply dataset scalers if available
                if hasattr(device, 'dataset') and device.dataset is not None and hasattr(device, 'apply_dataset'):
                    device.apply_dataset(self.timestep)

    def _sync_to_pandapower(self) -> None:
        """Push device states to pandapower network."""
        for agent in self.agents.values():
            mg_name = agent.agent_id
            for sub_agent in agent.subordinates.values():
                device = sub_agent.device
                element_name = f"{mg_name}_{device.name}"

                if device.type == 'ESS':
                    idx = pp.get_element_index(self.net, 'storage', element_name)
                    self.net.storage.at[idx, 'p_mw'] = device.state.P
                    self.net.storage.at[idx, 'q_mvar'] = device.state.Q
                    self.net.storage.at[idx, 'soc_percent'] = device.state.soc * 100
                elif device.type in ['DG', 'RES', 'fossil', 'solar', 'wind']:
                    idx = pp.get_element_index(self.net, 'sgen', element_name)
                    self.net.sgen.at[idx, 'p_mw'] = device.state.P
                    self.net.sgen.at[idx, 'q_mvar'] = device.state.Q

    def _solve_power_flow(self) -> bool:
        """Run pandapower power flow."""
        try:
            pp.runpp(self.net)
            return self.net.get('converged', False)
        except:
            self.net['converged'] = False
            return False

    def _update_cost_safety(self, converged: bool) -> None:
        """Update cost/safety for all devices and aggregate to GridAgents."""
        for agent in self.agents.values():
            total_cost = 0
            total_safety = 0

            for sub_agent in agent.subordinates.values():
                device = sub_agent.device
                # Compute device cost and safety if methods exist
                if hasattr(device, 'update_cost_safety'):
                    device.update_cost_safety()

                total_cost += getattr(device, 'cost', 0)
                total_safety += getattr(device, 'safety', 0)

            # Aggregate to GridAgent
            agent.cost = total_cost
            agent.safety = total_safety

    def _compute_rewards(self, converged: bool) -> Dict[str, float]:
        """Compute rewards for each GridAgent."""
        rewards = {}

        for aid, agent in self.agents.items():
            # Base reward: negative cost
            reward = -agent.cost

            # Add safety penalty
            reward -= self.penalty * agent.safety

            # Convergence penalty
            if not converged:
                reward -= self.penalty * 10

            rewards[aid] = reward

        # Share rewards if configured
        if self.share_reward:
            total_reward = sum(rewards.values())
            avg_reward = total_reward / len(rewards)
            rewards = {aid: avg_reward for aid in rewards}

        return rewards

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all GridAgents."""
        global_state = self._get_global_state()
        obs = {}

        for aid, agent in self.agents.items():
            agent_obs = agent.observe(global_state)
            obs[aid] = agent_obs.as_vector()

        return obs

    def _get_global_state(self) -> Dict[str, Any]:
        """Build global state dict for agent observations."""
        if self.net.get('converged', False):
            bus_vm = self.net.res_bus['vm_pu'].values
            bus_va = self.net.res_bus['va_degree'].values
        else:
            bus_vm = np.ones(len(self.net.bus))
            bus_va = np.zeros(len(self.net.bus))

        return {
            'bus_vm': bus_vm,
            'bus_va': bus_va,
            'timestep': self.timestep,
            'converged': self.net.get('converged', False)
        }
