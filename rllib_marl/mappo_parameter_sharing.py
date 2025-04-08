from powergrid.envs.multiagent.ieee34_ieee13 import MultiAgentMicrogrids

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.preprocessors import get_preprocessor

parser = add_rllib_example_script_args(
    default_iters=1000,
    default_timesteps=240000,
    default_reward=0.0,
)

args = parser.parse_args()
args.num_agents = 4
args.verbose = 1
# args.no_tune = True
# Here, we use the "Agent Environment Cycle" (AEC) PettingZoo environment type.
# For a "Parallel" environment example, see the rock paper scissors examples
# in this same repository folder.

env_config = {
    "train": True,
    "penalty": 10,
    "share_policy": True,
    "share_reward": True,
    }

register_env("env", lambda c: MultiAgentMicrogrids(c))

# Policies are called just like the agents (exact 1:1 mapping).
policies = {"DSO"}

base_config = PPOConfig()
base_config.train_batch_size=240
base_config.minibatch_size=64
base_config.num_epochs=10
base_config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
base_config.environment("env", env_config=env_config)
base_config.env_runners(num_env_runners=12)
base_config.learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
base_config.multi_agent(
        policies=policies,
        # Exact 1:1 mapping from AgentID to ModuleID.
        policy_mapping_fn=(lambda aid, *args, **kwargs: "DSO"),
    )
base_config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={"DSO": RLModuleSpec()},
        )
    )

run_rllib_example_script_experiment(base_config, args)
