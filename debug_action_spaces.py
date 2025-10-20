"""Debug script to check action spaces of agents."""

from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2

env_config = {'train': True, 'penalty': 10, 'share_reward': True}
env = MultiAgentMicrogridsV2(env_config)

print("Possible agents:", env.possible_agents)
print("\nAction spaces:")
for agent_id in env.possible_agents:
    space = env.action_spaces[agent_id]
    print(f"  {agent_id}: {space}")
    print(f"    Shape: {space.shape}")
    print(f"    Low: {space.low[:5] if len(space.low) > 5 else space.low}")
    print(f"    High: {space.high[:5] if len(space.high) > 5 else space.high}")

print("\nObservation spaces:")
for agent_id in env.possible_agents:
    space = env.observation_spaces[agent_id]
    print(f"  {agent_id}: {space}")
    print(f"    Shape: {space.shape}")
