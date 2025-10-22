"""
MAPPO Training Script for Multi-Agent Microgrids

This script trains a Multi-Agent PPO (MAPPO) policy on the 3-microgrid environment
using Ray RLlib. Supports both shared and independent policies.

Usage:
    # Train with shared policy (MAPPO)
    python examples/train_mappo_microgrids.py --iterations 100

    # Train with independent policies (IPPO)
    python examples/train_mappo_microgrids.py --iterations 100 --independent-policies

    # Resume from checkpoint
    python examples/train_mappo_microgrids.py --resume /path/to/checkpoint

    # With W&B logging
    python examples/train_mappo_microgrids.py --wandb --wandb-project powergrid-marl

Requirements:
    pip install "ray[rllib]==2.9.0"
    pip install wandb  # Optional, for logging
"""

import argparse
import os
import json
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Import environment
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MAPPO on multi-agent microgrids')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations (default: 100)')
    parser.add_argument('--train-batch-size', type=int, default=4000,
                        help='Training batch size (default: 4000)')
    parser.add_argument('--sgd-minibatch-size', type=int, default=128,
                        help='SGD minibatch size (default: 128)')
    parser.add_argument('--num-sgd-iter', type=int, default=10,
                        help='Number of SGD iterations (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--lambda', type=float, default=0.95, dest='lambda_',
                        help='GAE lambda (default: 0.95)')

    # Environment parameters
    parser.add_argument('--penalty', type=float, default=10,
                        help='Safety penalty coefficient (default: 10)')
    parser.add_argument('--share-reward', action='store_true',
                        help='Use shared rewards across all agents')
    parser.add_argument('--no-share-reward', dest='share_reward', action='store_false')
    parser.set_defaults(share_reward=True)

    # Policy parameters
    parser.add_argument('--independent-policies', action='store_true',
                        help='Use independent policies for each agent (IPPO)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension (default: 256)')

    # Parallelization
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of rollout workers (default: 4)')
    parser.add_argument('--num-envs-per-worker', type=int, default=1,
                        help='Number of environments per worker (default: 1)')

    # Checkpointing
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='Checkpoint frequency in iterations (default: 10)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='powergrid-marl',
                        help='W&B project name (default: powergrid-marl)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity (username or team)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for logging')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')

    return parser.parse_args()


def env_creator(env_config):
    """Create environment with RLlib compatibility."""
    env = MultiAgentMicrogrids(env_config)
    # Wrap with PettingZoo wrapper for RLlib
    return ParallelPettingZooEnv(env)


def get_policy_configs(env, args):
    """Get policy configuration (shared or independent)."""
    if args.independent_policies:
        # IPPO: Each agent has its own policy
        policies = {
            agent_id: (None, env.observation_space(agent_id), env.action_space(agent_id), {})
            for agent_id in env.possible_agents
        }
        policy_mapping_fn = lambda agent_id, *args_: agent_id
    else:
        # MAPPO: All agents share one policy
        policies = {'shared_policy': (None, None, None, {})}
        policy_mapping_fn = lambda agent_id, *args_: 'shared_policy'

    return policies, policy_mapping_fn


def main():
    """Main training function."""
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("multi_agent_microgrids", env_creator)

    # Create environment to get spaces
    env_config = {
        'train': True,
        'penalty': args.penalty,
        'share_reward': args.share_reward,
    }
    temp_env = env_creator(env_config)

    # Get policy configuration
    policies, policy_mapping_fn = get_policy_configs(temp_env, args)

    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_type = "ippo" if args.independent_policies else "mappo"
        args.experiment_name = f"{policy_type}_mg3_{timestamp}"

    print("=" * 70)
    print("Multi-Agent Microgrid Training with RLlib")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Policy type: {'IPPO (Independent)' if args.independent_policies else 'MAPPO (Shared)'}")
    print(f"Iterations: {args.iterations}")
    print(f"Shared reward: {args.share_reward}")
    print(f"Safety penalty: {args.penalty}")
    print(f"Learning rate: {args.lr}")
    print(f"Workers: {args.num_workers}")
    print("=" * 70)

    # Configure PPO algorithm
    config = (
        PPOConfig()
        .environment(
            env="multi_agent_microgrids",
            env_config=env_config,
        )
        .framework("torch")
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            clip_param=0.3,
            model={
                "fcnet_hiddens": [args.hidden_dim, args.hidden_dim],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
        )
        .resources(
            num_gpus=0 if args.no_cuda else 1,
        )
        .debugging(
            seed=args.seed,
        )
    )

    # Setup W&B logging if requested
    callbacks = []
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.experiment_name,
                config=vars(args),
            )
            print(f"W&B logging enabled: {args.wandb_project}")
        except ImportError:
            print("WARNING: wandb not installed. Install with: pip install wandb")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Build algorithm
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        algo = config.build()
        algo.restore(args.resume)
    else:
        algo = config.build()

    # Training loop
    print("\nStarting training...")
    print("-" * 70)
    print(f"{'Iter':>5} | {'Reward':>10} | {'Cost':>10} | {'Episodes':>8} | "
          f"{'Steps':>10} | {'Time':>8}")
    print("-" * 70)

    best_reward = float('-inf')

    for i in range(args.iterations):
        result = algo.train()

        # Extract metrics
        reward_mean = result.get('episode_reward_mean', 0)
        episodes = result.get('episodes_this_iter', 0)
        timesteps = result.get('timesteps_total', 0)
        time_total = result.get('time_total_s', 0)

        # Custom metrics (if available)
        custom_metrics = result.get('custom_metrics', {})
        cost_mean = custom_metrics.get('cost_mean', 0)

        # Print progress
        print(f"{i+1:5d} | {reward_mean:10.2f} | {cost_mean:10.2f} | "
              f"{episodes:8d} | {timesteps:10d} | {time_total:8.1f}s")

        # Log to W&B
        if args.wandb:
            try:
                wandb.log({
                    'iteration': i + 1,
                    'reward_mean': reward_mean,
                    'cost_mean': cost_mean,
                    'episodes': episodes,
                    'timesteps': timesteps,
                    'time_total': time_total,
                })
            except:
                pass

        # Checkpoint
        if (i + 1) % args.checkpoint_freq == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"  → Checkpoint saved: {checkpoint_path}")

            # Save best model
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_path = os.path.join(checkpoint_dir, 'best_checkpoint')
                os.makedirs(best_path, exist_ok=True)
                algo.save(best_path)
                print(f"  → Best model saved: {best_path}")

    print("-" * 70)
    print("Training complete!")
    print(f"Best reward: {best_reward:.2f}")

    # Final checkpoint
    final_path = algo.save(checkpoint_dir)
    print(f"Final checkpoint: {final_path}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    if args.wandb:
        try:
            wandb.finish()
        except:
            pass

    print("=" * 70)


if __name__ == '__main__':
    main()
