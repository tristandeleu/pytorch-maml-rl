import pickle
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


# goal: an array of 2; train/valid: (2, 100); 
def plot_traj(goal, train, valid):

	print(goal)
	print(" ")
	# print(train[:,1].shape)
	# print(train[:,2].shape)
	# print(" ")

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(train[:,0], train[:,1], '-.', color='b', linewidth=2, label='train')
	ax.plot(valid[:,0], valid[:,1], '--', color='g', linewidth=2, label='valid')

	ax.plot(train[-1,0], train[-1,1], 'bo', markersize=15, markeredgewidth=0, label='train end')
	ax.plot(valid[-1,0], valid[-1,1], 'go', markersize=15, markeredgewidth=0, label='valid end')

	# ax.plot(goal[0], goal[1], '*', color='r')
	ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgewidth=0, label='goal')

	ax.grid(True)
	ax.legend()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_xlim([-1, 1]) 
	ax.set_ylim([-1, 1]) 
	# plt.show()
	return fig
	
def get_traj(folder_path, traj_ind, num_grad):

	trajs_file_name = 'test_episodes_grad'+str(num_grad)+'.pkl'
	trajs = pickle.load(open(folder_path + trajs_file_name, "rb" ))
	traj = np.squeeze(trajs[0][:,traj_ind, :])

	return traj
	
def main():
	parser = argparse.ArgumentParser(description='MAML 2DNavigation plot making')
	parser.add_argument('--task_ind', type=int, default=10, help='which task to be plotted')
	parser.add_argument('--traj_ind', type=int, default=10, help='which trajectory to be plotted')
	# parser.add_argument('--x_scaling_factor', type=float, default=0.36883, help='true x = current_x * x_scaling_factor')
	# parser.add_argument('--y_scaling_factor', type=float, default=0.459005, help='true y = current_y * y_scaling_factor')
	parser.add_argument('--plot-type', type=str, default='train', help='train or test')
	args = parser.parse_args()

	if args.plot_type == 'train':
		# tasks_log_file_name = 'tasks_0.pkl'
		# train_traj_log_file_name = 'train_episodes_observ_0.pkl'
		# valid_traj_log_file_name = 'valid_episodes_observ_0.pkl'
		tasks_log_file_name = 'latest_tasks.pkl'
		train_traj_log_file_name = 'latest_train_episodes_observ.pkl'
		valid_traj_log_file_name = 'latest_valid_episodes_observ.pkl'


		tasks_log_file_path = './logs/2DNavigation-traj-dir/'+tasks_log_file_name
		train_traj_log_file_path = './logs/2DNavigation-traj-dir/'+train_traj_log_file_name
		valid_traj_log_file_path = './logs/2DNavigation-traj-dir/'+valid_traj_log_file_name

		task_list = pickle.load(open(tasks_log_file_path, "rb" ) )
		train_traj_list = pickle.load(open(train_traj_log_file_path, "rb" ) )
		valid_traj_list = pickle.load(open(valid_traj_log_file_path, "rb" ) )

		print('Number of tasks is {}'.format(len(task_list)))
		try:
			one_task = task_list[args.task_ind] # an array of 2: e.g. array([0.0209588 , 0.15981938])
		except:
			sys.exit('Execution stopped: the task list does not contain Task '+str(args.task_ind)+'. Please choose another task.')


		print('Number of trajectories is {}'.format(train_traj_list[args.task_ind].shape[1]))
		try:
			one_goal = task_list[args.traj_ind] 
		except:
			sys.exit('Execution stopped: the trajectory list does not contain Traj '+str(args.traj_ind)+'. Please choose another task.')

		print(" ")
		if tasks_log_file_name[0] == 'l':
			print("latest epoch")
		else:
			print("first epoch")

		one_train = np.squeeze(train_traj_list[args.task_ind][:,args.traj_ind, :])
		one_valid = np.squeeze(valid_traj_list[args.task_ind][:,args.traj_ind, :])
		fig = plot_traj(one_task['goal'], one_train, one_valid)
		fig.savefig('./logs/2DNavigation-traj-dir/' + 'test.png', bbox_inches = 'tight')

	else:
		folder_path = './test_nav/'
		task_file_name = 'task.pkl'
		task = pickle.load(open(folder_path + task_file_name, "rb" ))

		traj0 = get_traj(folder_path, args.traj_ind, 0)

		traj1 = get_traj(folder_path, args.traj_ind, 1)

		fig = plot_traj(task['goal'], traj0, traj1)
		fig.savefig(folder_path+'test.png', bbox_inches='tight')




# len(traj_list) = # of tasks
# traj_list[0].shape = (length of traj, # traj, 2D coordinates) = (100, 20, 2)

if __name__ == '__main__':
	main()


