"""Debug script to check device action spaces."""

from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv

env_config = {'train': True, 'penalty': 10, 'share_reward': True}
env = MultiAgentMicrogridsV2(env_config)

print("Grid Agents and their devices:")
for agent_id in env.possible_agents:
    agent = env.agents[agent_id]
    print(f"\n{agent_id}:")
    print(f"  Total action space: {agent.action_space.shape}")
    print(f"  Subordinates:")
    for sub_id, sub_agent in agent.subordinates.items():
        print(f"    {sub_id}: {sub_agent.action_space.shape}")
        print(f"      Device: {sub_agent.device}")
