"""
NetworkedGridEnv: Multi-agent environment for networked microgrids using agent classes.

This is a modernized version of the legacy NetworkedGridEnv that replaces GridEnv
with PowerGridAgent while maintaining identical environment logic and API.
"""

from abc import abstractmethod
from typing import Any, Dict

import gymnasium.utils.seeding as seeding
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import NoProtocol, Protocol


class NetworkedGridEnv(ParallelEnv):
    metadata = {"name": "networked_grid_env"}

    def __init__(self, env_config):
        super().__init__()
        self.agent_dict: Dict[str, PowerGridAgent] = {}
        self.data_size: int = 0
        self._t: int = 0  # current timestep
        # _day will be initialized in reset() for test mode
        self._total_days: int = 0  # total number of days in the dataset

        self.env_config = env_config
        self.max_episode_steps = env_config.get('max_episode_steps', 24)
        self.train = env_config.get('train', True)
        self.type = env_config.get('type', 'AC')
        self.protocol = env_config.get('protocol', NoProtocol())

        # Build network (must set self.net, self.possible_agents, self.agent_dict)
        self._build_net()
        self._init_space()

    @property
    def actionable_agents(self):
        """Get agents that have actionable devices."""
        return {
            n: a for n, a in self.agent_dict.items()
            if len(a.get_device_action_spaces()) > 0
        }

    @abstractmethod
    def _build_net(self):
        """
        Build network and agents.

        Subclasses must implement this method to:
        1. Create merged pandapower network (self.net)
        2. Create PowerGridAgent instances (self.possible_agents dict)
        3. Set active agents (self.agent_dict)
        4. Set episode parameters (self.max_episode_steps, self.data_size, etc.)
        """
        pass

    @abstractmethod
    def _reward_and_safety(self):
        """
        Compute rewards and safety violations.

        Returns:
            rewards: Dict mapping agent_id → reward (float)
            safety: Dict mapping agent_id → safety violation (float)
        """
        pass

    def step(self, action_n: Dict[str, Any]):
        """
        Execute one environment step.

        Args:
            action_n: Dict mapping agent_id → action

        Returns:
            observations: Dict mapping agent_id → observation
            rewards: Dict mapping agent_id → reward
            dones: Dict with '__all__' → done
            truncated: Dict with '__all__' → truncated
            infos: Safety information dict
        """
        # Set action for each agent
        if self.protocol.no_op():
            # note that action can be None here -> decentralized action computation per grid agent
            for name, action in action_n.items():
                if name in self.actionable_agents:
                    # Get observation for this agent
                    obs = self.actionable_agents[name].observe(net=self.net)
                    # Compute and set actions on devices
                    self.actionable_agents[name].act(obs, given_action=action)
        else:
            self.protocol.coordinate_actions(self.actionable_agents, action_n, self.net, self._t)

        # Update device states and sync to pandapower
        for agent in self.agent_dict.values():
            agent.update_state(self.net, self._t)

        # Run power flow for the whole network
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        # Update costs and safety for all agents
        for agent in self.agent_dict.values():
            agent.update_cost_safety(self.net)

        # Get rewards and safety from subclass
        rewards, safety = self._reward_and_safety()

        # Share rewards if configured
        if self.env_config.get('share_reward'):
            shared_reward = np.mean(list(rewards.values()))
            rewards = {name: shared_reward for name in self.agent_dict}

        # Timestep counter
        self._t = self._t + 1 if self._t < self.data_size else 0

        # Done
        done = self._t % self.max_episode_steps == 0
        terminateds = {"__all__": done}
        truncateds = {"__all__": done}

        # Info
        infos = safety

        return self._get_obs(), rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        """
        Reset environment and all agents.

        Args:
            seed: Random seed
            options: Reset options (unused)

        Returns:
            observations: Dict mapping agent_id → observation
            info: Empty info dict
        """
        # Initialize RNG
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = seeding.np_random(None)

        # Reset all agents
        if self.train:
            self._day = self.np_random.integers(self.total_days - 1)
            self._t = self._day * self.max_episode_steps
            for agent in self.agent_dict.values():
                agent.reset(seed=seed)
        else:
            if hasattr(self, '_day'):
                self._day += 1
                self._t = self._day * self.max_episode_steps
            else:
                self._t, self._day = 0, 0
            for agent in self.agent_dict.values():
                agent.reset(seed=seed)

        # Update initial states
        for agent in self.agent_dict.values():
            agent.update_state(self.net, self._t)

        # Initial power flow
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        info = {}

        return self._get_obs(), info

    def _get_obs(self):
        """
        Get observations for all agents.

        Returns:
            Dict mapping agent_id → observation array
        """
        obs_dict = {}
        for n, a in self.agent_dict.items():
            obs = a.observe(net=self.net)
            # Extract state from observation
            obs_dict[n] = obs.local['state']

        return obs_dict

    def _init_space(self):
        """Initialize action and observation spaces for all agents."""
        ac_spaces = {}
        ob_spaces = {}

        for name, agent in self.agent_dict.items():
            ac_spaces[name] = agent.get_grid_action_space()
            ob_spaces[name] = agent.get_grid_observation_space(self.net)

        self.action_spaces = ac_spaces
        self.observation_spaces = ob_spaces
