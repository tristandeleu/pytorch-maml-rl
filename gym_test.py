import gym
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
# Test gym

env = gym.make('RVONavigation-v0')

# print(env.unwrapped._ped_states )
env.unwrapped.print_rvo2_states()

tasks = env.unwrapped.sample_tasks(2)
env.unwrapped.reset_task(tasks[1])
env.unwrapped.print_rvo2_states()



for t in range(10):
	print("\n t = ", t)
	state, reward, done, task = env.unwrapped.step([0., 0.])
	env.unwrapped.print_rvo2_states()
	env.unwrapped.print_robot_state()
	assert env.unwrapped.assert_sim_and_states()
	print(env.unwrapped.assert_sim_and_states())
	print(reward)



