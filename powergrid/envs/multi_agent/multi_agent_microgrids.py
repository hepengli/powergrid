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
        # Load dataset before calling super().__init__ since _build_net needs it
        self._dataset = load_dataset(env_config.get('dataset_path'))
        self._safety = env_config.get('penalty')
        self._convergence_failure_reward = env_config.get('convergence_failure_reward', -200.0)
        self._convergence_failure_safety = env_config.get('convergence_failure_safety', 20.0)

        super().__init__(env_config)

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
        self._total_days = self.dataset_size // self.max_episode_steps

        return net

    def _build_mg_agent(self, mg_config) -> PowerGridAgent:
        """Build microgrid agent from config."""
        mg_net = IEEE13Bus(mg_config['name'])
        storage = []
        sgen = []
        for device_args in mg_config['devices']:
            # Use .get() instead of .pop() to avoid modifying the config
            device_type = device_args.get('type', None)
            # Create a copy without 'type' for device initialization
            device_kwargs = {k: v for k, v in device_args.items() if k != 'type'}
            if device_type == 'ESS':
                storage.append(ESS(**device_kwargs))
            elif device_type == 'DG':
                sgen.append(DG(**device_kwargs))
            elif device_type == 'RES':
                sgen.append(RES(**device_kwargs))
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
    """Test building MultiAgentMicrogrids environment with real config."""
    from powergrid.envs.configs.config_loader import load_config

    print("Loading IEEE 34-13 bus configuration...")
    env_config = load_config('ieee34_ieee13')

    print(f"Config loaded:")
    print(f"  - Training mode: {env_config['train']}")
    print(f"  - Dataset: {env_config['dataset_path']}")
    print(f"  - Number of microgrids: {len(env_config['mg_configs'])}")
    print(f"  - Penalty: {env_config['penalty']}")

    print("\nBuilding environment...")
    try:
        env = MultiAgentMicrogrids(env_config)
        print("✓ Environment built successfully!")

        print(f"\nEnvironment details:")
        print(f"  - Total agents: {len(env.agent_dict)}")
        print(f"  - Agent names: {list(env.agent_dict.keys())}")
        print(f"  - Actionable agents: {list(env.actionable_agents.keys())}")
        print(f"  - Dataset size: {env.dataset_size}")
        print(f"  - Total days: {env._total_days}")

        print("\nResetting environment...")
        obs, info = env.reset(seed=42)
        print(f"✓ Reset successful!")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")

        print("\nAction spaces:")
        for agent_name, space in env.action_spaces.items():
            print(f"  - {agent_name}: {space}")

        print("\nTaking a test step...")
        actions = {name: space.sample() for name, space in env.action_spaces.items()}
        obs_next, rewards, terminateds, truncateds, infos = env.step(actions)
        print(f"✓ Step successful!")
        print(f"  - Rewards: {[(k, f'{v:.2f}') for k, v in rewards.items()]}")
        print(f"  - Converged: {env.net['converged']}")

        print("\n✓ All tests passed! Environment is working correctly.")

    except Exception as e:
        print(f"\n✗ Error building environment: {e}")
        import traceback
        traceback.print_exc()
