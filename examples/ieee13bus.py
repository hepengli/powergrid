import gymnasium as gym
from stable_baselines3 import A2C

from powergrid.envs.single_agent.ieee34_mg import IEEE34Env

env = IEEE34Env(env_config={})

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    unwrapped_env = vec_env.unwrapped.envs[0]
    print("\n------------ Step {}-------------".format(i))
    for name, dev in unwrapped_env.env.devices.items():
        print(name, dev.state.as_vector(), dev.cost, dev.safety)
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()