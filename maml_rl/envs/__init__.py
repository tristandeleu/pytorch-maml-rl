from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVelEnv-v0',
    entry_point='maml_rl.envs.mujoco.ant_vel:AntVelEnv',
    max_episode_steps=200
)

register(
    'AntDirEnv-v0',
    entry_point='maml_rl.envs.mujoco.ant_dir:AntDirEnv',
    max_episode_steps=200
)

register(
    'HalfCheetahVelEnv-v0',
    entry_point='maml_rl.envs.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
    max_episode_steps=200
)

register(
    'HalfCheetahDirEnv-v0',
    entry_point='maml_rl.envs.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
    max_episode_steps=200
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
