from gym.envs.registration import register
# from .ieee34 import IEEE34BusSystem
# from .ieee123 import IEEE123BusSystem
# from .ieee123_ddpg import IEEE123BusSystemDDPG
# from systems.environment_ieee34_ddpg import Environment_IEEE34_DDPG
# from systems.environment_cigre_mv import Environment_CIGRE_MV
# from systems.environment_cigre_lv import Environment_CIGRE_LV

# Multiagent Microgrid envs
# ----------------------------------------
register(
    id='ieee34-v1',
    entry_point='powergrid.multi_agent_envs.ieee34:IEEE34BusSystem',
    max_episode_steps=24,
)
register(
    id='ieee34-v0',
    entry_point='powergrid.single_agent_envs.ieee34:IEEE34BusSystem',
    max_episode_steps=24,
)
register(
    id='ieee123-v0',
    entry_point='powergrid.single_agent_envs.ieee123:IEEE123BusSystem',
    max_episode_steps=24,
)
register(
    id='ieee123ddpg-v0',
    entry_point='powergrid.single_agent_envs.ieee123_ddpg:IEEE123BusSystemDDPG',
    max_episode_steps=24,
)
register(
    id='networked_mgs-v1',
    entry_point='powergrid.multi_agent_envs.networked_mgs:NetworkedMGSystems',
    max_episode_steps=24,
)