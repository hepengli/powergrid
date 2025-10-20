"""Debug script to check reward calculation."""

from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogridsV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

env_config = {'train': True, 'penalty': 10, 'share_reward': True}
env = MultiAgentMicrogridsV2(env_config)

print("Testing environment rewards...")
print("\nReset environment:")
obs, info = env.reset()
print(f"  Agents: {env.agents}")
print(f"  Observation keys: {obs.keys()}")

# Take a few steps
for step in range(5):
    print(f"\n=== Step {step + 1} ===")

    # Random actions for all agents
    actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
    print(f"Actions: {actions}")

    obs, rewards, dones, truncated, infos = env.step(actions)

    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Truncated: {truncated}")

    # Check if environment converged
    if hasattr(env, 'net'):
        print(f"Network converged: {env.net.get('converged', 'N/A')}")

    # Check agent costs
    for aid, agent in env.agents.items():
        if hasattr(agent, 'cost'):
            print(f"  {aid} cost: {agent.cost}")
