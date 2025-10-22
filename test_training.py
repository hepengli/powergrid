"""
Quick test script to verify MAPPO training works.
Runs 3 iterations with minimal configuration.
"""

import sys
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids

print("="*70)
print("MAPPO Training Test")
print("="*70)

# Initialize Ray
print("\n1. Initializing Ray...")
ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=0)
print("   ✓ Ray initialized")

# Create and register environment
print("\n2. Creating environment...")
def env_creator(config):
    env = MultiAgentMicrogrids(config)
    return ParallelPettingZooEnv(env)

env_config = {'train': True, 'penalty': 10, 'share_reward': True}
temp_env = env_creator(env_config)
print(f"   ✓ Environment created")
print(f"   Agents: {temp_env.par_env.possible_agents}")

# Register the environment
register_env("powergrid_multiagent", env_creator)

# Configure PPO
print("\n3. Configuring PPO algorithm...")
config = (
    PPOConfig()
    .environment(env="powergrid_multiagent", env_config=env_config)
    .framework("torch")
    .training(
        train_batch_size=512,  # Small for quick test
        sgd_minibatch_size=64,
        num_sgd_iter=3,
        lr=5e-4,
        gamma=0.99,
        model={"fcnet_hiddens": [64, 64]},  # Small network
    )
    .multi_agent(
        policies={'shared_policy': (None, None, None, {})},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: 'shared_policy',
    )
    .rollouts(
        num_rollout_workers=0,  # No parallel workers for test
        num_envs_per_worker=1,
    )
    .resources(num_gpus=0)
    .debugging(seed=42)
)
print("   ✓ PPO configured (MAPPO with shared policy)")

# Build algorithm
print("\n4. Building algorithm...")
algo = config.build()
print("   ✓ Algorithm built")

# Train for 3 iterations
print("\n5. Running training (3 iterations)...")
print("-" * 70)
print(f"{'Iter':>5} | {'Reward':>12} | {'Episodes':>9} | {'Steps':>10}")
print("-" * 70)

results = []
for i in range(3):
    result = algo.train()

    reward_mean = result.get('episode_reward_mean', 0)
    episodes = result.get('episodes_this_iter', 0)
    timesteps = result.get('timesteps_total', 0)

    results.append({
        'iteration': i + 1,
        'reward': reward_mean,
        'episodes': episodes,
        'timesteps': timesteps
    })

    print(f"{i+1:5d} | {reward_mean:12.2f} | {episodes:9d} | {timesteps:10d}")

print("-" * 70)

# Summary
print("\n6. Training Summary:")
print(f"   Initial reward: {results[0]['reward']:.2f}")
print(f"   Final reward:   {results[-1]['reward']:.2f}")
print(f"   Total episodes: {results[-1]['episodes']}")
print(f"   Total timesteps: {results[-1]['timesteps']}")

# Check if reward improved or training ran without errors
if len(results) == 3:
    print("\n   ✓ Training completed successfully!")
else:
    print("\n   ✗ Training failed!")
    sys.exit(1)

# Cleanup
print("\n7. Cleaning up...")
algo.stop()
ray.shutdown()
print("   ✓ Cleanup complete")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED - MAPPO training works!")
print("="*70)
print("\nYou can now run full training with:")
print("  python examples/train_mappo_microgrids.py --iterations 100")
