import numpy as np

from ant_vel import AntVelEnv
from ant_dir import AntDirEnv
from half_cheetah_vel import HalfCheetahVelEnv
from half_cheetah_dir import HalfCheetahDirEnv


ant_vel = AntVelEnv()
ant_dir = AntDirEnv()
half_cheetah_vel = HalfCheetahVelEnv()
half_cheetah_dir = HalfCheetahDirEnv()

envs = [ant_vel, ant_dir, half_cheetah_vel, half_cheetah_dir]

for env in envs:
    print (env)
    print (env.reset())
    tasks = env.sample_tasks(5)
    print (tasks)
    env.reset_task(tasks[0])
    print (env._task)
    print (env._get_obs())
    low = env.action_space.low
    high = env.action_space.high
    num_actions = env.action_space.shape[0]
    print (env._task)
    for i in range(5):
        action = np.random.uniform(low, high, size=num_actions)
        env.step(action)
    print (env.step(action))
