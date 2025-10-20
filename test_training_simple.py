"""
Simple MAPPO training test without complex dataset dependencies.
Creates a minimal test environment.
"""

import sys
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from powergrid.envs.multi_agent import MultiAgentPowerGridEnv
from powergrid.networks.ieee13 import IEEE13Bus
from powergrid.devices import ESS, DG

print("="*70)
print("MAPPO Training Test (Simplified)")
print("="*70)

# Create simple mock dataset
print("\n1. Creating mock dataset...")
mock_dataset = {
    'load': np.random.rand(8760) * 0.5 + 0.5,  # Random load profile
    'solar': np.random.rand(8760) * 0.3,
    'solar bus 23': np.random.rand(8760) * 0.3,
    'wind': np.random.rand(8760) * 0.2,
    'price': np.random.rand(8760) * 50 + 30  # $30-80/MWh
}
print("   ✓ Mock dataset created")

# Initialize Ray
print("\n2. Initializing Ray...")
ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=0)
print("   ✓ Ray initialized")

# Create environment
print("\n3. Creating simple 2-microgrid environment...")
def env_creator(rllib_config):
    # Note: rllib_config is passed by RLlib but we ignore it for this simple test
    env_config = {
        'network': None,
        'microgrids': [
            {
                'name': 'MG1',
                'network': IEEE13Bus('MG1'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.0, max_e_mwh=1.0, min_e_mwh=0.1, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.5,
                       cost_curve_coefs=[100, 70.0, 0.5]),
                ],
                'vertical_protocol': 'none',
                'dataset': mock_dataset
            },
            {
                'name': 'MG2',
                'network': IEEE13Bus('MG2'),
                'devices': [
                    ESS('ESS1', bus='Bus 645', min_p_mw=-0.3, max_p_mw=0.3,
                        capacity=1.0, max_e_mwh=1.0, min_e_mwh=0.1, init_soc=0.5),
                    DG('DG1', bus='Bus 675', min_p_mw=0, max_p_mw=0.4,
                       cost_curve_coefs=[100, 60.0, 0.45]),
                ],
                'vertical_protocol': 'none',
                'dataset': mock_dataset
            }
        ],
        'horizontal_protocol': 'none',
        'episode_length': 24,
        'train': True,
        'penalty': 10,
    }
    env = MultiAgentPowerGridEnv(env_config)
    return ParallelPettingZooEnv(env)

# Register environment with RLlib
from ray.tune.registry import register_env
register_env("test_env", env_creator)

temp_env = env_creator({})
print(f"   ✓ Environment created")
print(f"   Agents: {temp_env.par_env.possible_agents}")
print(f"   Action spaces: {[(aid, temp_env.action_space[aid].shape) for aid in temp_env.par_env.possible_agents]}")
print(f"   Observation spaces: {[(aid, temp_env.observation_space[aid].shape) for aid in temp_env.par_env.possible_agents]}")

# Get actual observation size
obs, _ = temp_env.par_env.reset()
print(f"   Actual observation shapes: {[(aid, obs[aid].shape) for aid in obs]}")

# Configure PPO
print("\n4. Configuring PPO algorithm...")
config = (
    PPOConfig()
    .environment(env="test_env", env_config={}, disable_env_checking=True)
    .framework("torch")
    .training(
        train_batch_size=384,  # Small for quick test
        sgd_minibatch_size=64,
        num_sgd_iter=3,
        lr=5e-4,
        gamma=0.99,
        model={"fcnet_hiddens": [64, 64]},  # Small network
    )
    .multi_agent(
        policies={
            'shared_policy': (
                None,  # Use default policy class
                temp_env.observation_space['MG1'],  # Use MG1's observation space
                temp_env.action_space['MG1'],  # Use MG1's action space
                {}  # No config overrides
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: 'shared_policy',
    )
    .rollouts(
        num_rollout_workers=0,  # No parallel workers for test
        num_envs_per_worker=1,
        rollout_fragment_length=24,  # Match episode length
    )
    .resources(num_gpus=0)
    .debugging(seed=42)
)
print("   ✓ PPO configured (MAPPO with shared policy)")

# Build algorithm
print("\n5. Building algorithm...")
algo = config.build()
print("   ✓ Algorithm built")

# Train for 3 iterations
print("\n6. Running training (3 iterations)...")
print("-" * 70)
print(f"{'Iter':>5} | {'Reward':>12} | {'Episodes':>9} | {'Steps':>10}")
print("-" * 70)

results = []
try:
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
    print("\n7. Training Summary:")
    print(f"   Initial reward: {results[0]['reward']:.2f}")
    print(f"   Final reward:   {results[-1]['reward']:.2f}")
    print(f"   Total episodes: {sum([r['episodes'] for r in results])}")
    print(f"   Total timesteps: {results[-1]['timesteps']}")

    # Check if training ran without errors
    if len(results) == 3:
        print("\n   ✓ Training completed successfully!")
        success = True
    else:
        print("\n   ✗ Training failed!")
        success = False

except Exception as e:
    print(f"\n   ✗ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    success = False

# Cleanup
print("\n8. Cleaning up...")
algo.stop()
ray.shutdown()
print("   ✓ Cleanup complete")

print("\n" + "="*70)
if success:
    print("✓ ALL TESTS PASSED - MAPPO training works!")
    print("="*70)
    print("\nEnvironment and training pipeline verified successfully!")
    print("\nFull training script is available at:")
    print("  examples/train_mappo_microgrids.py")
    sys.exit(0)
else:
    print("✗ TRAINING FAILED")
    print("="*70)
    sys.exit(1)
