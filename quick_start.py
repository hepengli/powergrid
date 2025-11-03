from powergrid.envs.single_agent.ieee13_mg import IEEE13Env

# Create and wrap: agent acts in [-1,1] for the continuous part
env = IEEE13Env({"episode_length": 24, "train": True})
obs, info = env.reset()

# Take a random step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("reward=", reward, "converged=", info.get("converged"))