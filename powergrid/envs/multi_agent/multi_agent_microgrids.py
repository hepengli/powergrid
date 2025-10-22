"""
MultiAgentMicrogrids: Concrete environment for 3 networked microgrids.

This is a modernized version of the legacy MultiAgentMicrogrids that uses
PowerGridAgent instead of GridEnv while maintaining identical logic.
"""


import pandapower as pp
from typing import List

from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.data.data_loader import load_dataset
from powergrid.devices.generator import DG, RES
from powergrid.devices.storage import ESS
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.networks.ieee34 import IEEE34Bus




class MultiAgentMicrogrids(NetworkedGridEnv):
    """
    Environment with 3 microgrids (MG1, MG2, MG3) connected to a DSO grid.

    Each microgrid has:
    - 1 ESS (Energy Storage System)
    - 1 DG (Dispatchable Generator)
    - 1 PV (Solar)
    - 1 WT (Wind Turbine)

    This implementation uses PowerGridAgent to manage devices instead of
    the legacy GridEnv, but maintains identical environment logic.
    """

    def __init__(self, env_config):
        """
        Initialize multi-agent microgrids environment.

        Args:
            env_config: Configuration dict with keys:
                - train: bool, training mode
                - penalty: float, safety penalty multiplier
                - share_reward: bool, share rewards across agents
        """
        super().__init__(env_config)
        self._dataset = load_dataset(env_config.get('dataset_path'))
        self._safety = self.env_config.get('penalty')
        self._convergence_failure_reward = self.env_config.get('convergence_failure_reward', -200.0)
        self._convergence_failure_safety = self.env_config.get('convergence_failure_safety', 20.0)

    def _read_data(self, load_area, renew_area):
        """Read data from dataset with train/test split support.

        Args:
            load_area: Load area identifier (e.g., 'AVA', 'BANC', 'BANCMID')
            renew_area: Renewable energy area identifier (e.g., 'NP15')
        """
        split = 'train' if self.train else 'test'
        data = self._dataset[split]

        return {
            'load': data['load'][load_area],
            'solar': data['solar'][renew_area],
            'wind': data['wind'][renew_area],
            'price': data['price']['0096WD_7_N001']
        }

    def _build_dso_net(self):
        """Build DSO main grid (non-actionable)."""
        net=IEEE34Bus('DSO')

        dso_config = self.env_config['dso_config']
        load_area = dso_config.get('load_area', 'BANC')
        renew_area = dso_config.get('renew_area', 'NP15')

        self.dso = PowerGridAgent(
            net=net,
            grid_config=dso_config,
            devices=[],  # No actionable devices in DSO
            centralized=False
        )
        self.dso.add_dataset(self._read_data(load_area, renew_area))
        self.dataset_size = self.dso.dataset['price'].size
        self.total_days = self.data_size // self.max_episode_steps

        return net

    def _build_mg_agent(self, mg_config) -> PowerGridAgent:
        """Build microgrid agent from config."""
        mg_net = IEEE13Bus(mg_config['name'])
        storage = []
        sgen = []
        for device_args in mg_config['devices']:
            device_type = device_args.pop('type', None)
            if device_type == 'ESS':
                storage.append(ESS(**device_args))
            elif device_type == 'DG':
                sgen.append(DG(**device_args))
            elif device_type == 'RES':
                sgen.append(RES(**device_args))
            else:
                raise ValueError(f"Unknown device type: {device_type}")
        mg_agent = PowerGridAgent(
            net=mg_net,
            grid_config=mg_config,
            centralized=mg_config.get('centralized', True)
        )
        load_area = mg_config.get('load_area', 'AVA')
        renew_area = mg_config.get('renew_area', 'NP15')
        mg_agent.add_sgen(sgen)
        mg_agent.add_storage(storage)
        mg_agent.add_dataset(self._read_data(load_area, renew_area))
        return mg_agent


    def _build_net(self):
        """Build network with 3 microgrids connected to DSO grid."""
        # Create DSO main grid (non-actionable)
        net = self._build_dso_net()

        # Create microgrids (actionable)
        mg_agents: List[PowerGridAgent] = []
        for mg_config in self.env_config['mg_configs']:
            mg_agent = self._build_mg_agent(mg_config)
            net = mg_agent.fuse_buses(net, mg_config['connection_bus'])
            mg_agents.append(mg_agent)

        # Run initial power flow
        pp.runpp(net)

        # Set network and agents
        self.net = net
        self.agent_dict.update({a.agent_id: a for a in mg_agents})
        self.possible_agents = list(self.agent_dict.keys())
        self.agents = self.possible_agents

    def _reward_and_safety(self):
        """
        Compute rewards and safety violations.

        Returns:
            rewards: Dict mapping agent_id → reward
            safety: Dict mapping agent_id → safety violation
        """
        if self.net["converged"]:
            # Reward and safety
            rewards = {n: -a.cost for n, a in self.agent_dict.items()}
            safety = {n: a.safety for n, a in self.agent_dict.items()}
        else:
            # Convergence failure penalty
            rewards = {n: self._convergence_failure_reward for n in self.agent_dict}
            safety = {n: self._convergence_failure_safety for n in self.agent_dict}

        # Apply safety penalty
        if self._safety:
            for name in self.agent_dict:
                rewards[name] -= safety[name] * self._safety

        return rewards, safety


if __name__ == '__main__':
    # Example usage
    env_config = {
        "train": True,
        "penalty": 10,
        "share_reward": True,
    }
    env = MultiAgentMicrogrids(env_config)
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Action spaces: {env.action_spaces}")
