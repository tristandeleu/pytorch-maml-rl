
import pickle
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# goal: an array of 2; train/valid: (2, 100); 
def plot_traj(goal, train, valid):

	print("\ngoal: {}\n".format(goal))

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(train[:,0], train[:,1], '-.', color='b', linewidth=2, label='train')
	ax.plot(valid[:,0], valid[:,1], '--', color='g', linewidth=2, label='valid')

	ax.plot(train[-1,0], train[-1,1], 'bo', markersize=15, markeredgewidth=0, label='train end')
	ax.plot(valid[-1,0], valid[-1,1], 'go', markersize=15, markeredgewidth=0, label='valid end')

	ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgewidth=0, label='goal')


	ax.grid(True)
	ax.legend()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	# plt.show()
	return fig

	
def anim_traj(goal, valid, ped_num):

	def update_line(num, data, lines, dots, ped_circs):
		
		line_num = len(lines)


		robot_state = data[0][:2, :] # [0] means valid
		ped_state = data[0][2:, :] # (8, 100)
		ped_state = np.split(ped_state, ped_num) # len(ped_state) = 4; each (2,100)
		for i in range(ped_num):
			ped_circs[i].center = (ped_state[i][0,num], ped_state[i][1,num])


		lines[0].set_data(robot_state[..., :num]) 
		dots[0].set_data(robot_state[..., num - 1:num]) # (2,100)

		return lines+dots+ped_circs


	# valid (10, 100)
	fig1 = plt.figure()

	ax = plt.axes(xlim=(min(valid[0,:])-0.2, max(valid[0,:])+0.2), ylim=(min(valid[1,:])-0.2, max(valid[1,:])+0.2))

	ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgewidth=0, label='goal')

	ped_circs = [plt.Circle((valid[2 + i*2,0], valid[2 + i*2+1, 0]), 0.05, fc='y') for i in range(ped_num)]
	for i in range(ped_num):
		ax.add_patch(ped_circs[i])
	
	valid_line, = plt.plot([], [], '-', color='xkcd:lime')
	valid_dot, = plt.plot([], [], 'o', color='g', label='valid')

	# lines = [train_line, valid_line]
	# dots = [train_dot, valid_dot]
	lines = [valid_line]
	dots = [valid_dot]
	data = [valid]

	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)
	plt.legend()
	# plt.title('test')
	print("\nvalid: ", valid.shape)

	line_ani = animation.FuncAnimation(fig1, update_line, frames=valid.shape[1], fargs=(data, lines, dots, ped_circs),
                                   interval=500, blit=True)

	# uncomment below to save as gif
	# line_ani.save('line.gif', dpi=80, writer='imagemagick')
	plt.show()


def main():
	parser = argparse.ArgumentParser(description='MAML 2DNavigation plot making')
	parser.add_argument('--task_ind', type=int, default=0, help='which task to be plotted')
	parser.add_argument('--traj_ind', type=int, default=1, help='which trajectory to be plotted')
	args = parser.parse_args()



	# ---------------   REMEMBER to make changes if needed ------------------------
	ped_num = 4
	self_state = 6
	ped_state = 4
	# ---------------   Finish making changes ------------------------




	# --------------- loading trajectory files ---------------------
	# Epoch_num = 420
	# tasks_log_file_name = 'tasks_{}.pkl'.format(Epoch_num)
	# train_traj_log_file_name = 'train_episodes_observ_{}.pkl'.format(Epoch_num)
	# valid_traj_log_file_name = 'valid_episodes_observ_{}.pkl'.format(Epoch_num)
	tasks_log_file_name = 'task.pkl'
	train_traj_log_file_name = 'test_episodes_grad0.pkl'
	valid_traj_log_file_name = 'test_episodes_grad3.pkl'





	if(tasks_log_file_name[0]!='l'):
		# print(" ")
		print("\nNOT latest epsoide")
	else:
		print("\nlatest epsoide")

	tasks_log_file_path = './test_nav/'+tasks_log_file_name
	train_traj_log_file_path = './test_nav/'+train_traj_log_file_name
	valid_traj_log_file_path = './test_nav/'+valid_traj_log_file_name

	task_list = pickle.load(open(tasks_log_file_path, "rb" ) )
	train_traj_list = pickle.load(open(train_traj_log_file_path, "rb" ) )
	valid_traj_list = pickle.load(open(valid_traj_log_file_path, "rb" ) )

	print('Number of tasks is {}'.format(len(task_list)))
	try:
		one_task = task_list
		# one_task = task_list[args.task_ind] # an array of 2: e.g. array([0.0209588 , 0.15981938])
	except:
		sys.exit('Execution stopped: the task list does not contain Task '+str(args.task_ind)+'. Please choose another task.')


	print('Number of trajectories is {}'.format(train_traj_list[args.task_ind].shape[1]))
	try:
		one_goal = train_traj_list[args.traj_ind] 
	except:
		sys.exit('Execution stopped: the trajectory list does not contain Traj '+str(args.traj_ind)+'. Please choose another task.')
	one_train = np.squeeze(train_traj_list[args.task_ind][:,args.traj_ind, :])
	one_valid = np.squeeze(valid_traj_list[args.task_ind][:,args.traj_ind, :])


    
	traj_len = one_train.shape[0]



	# ------ PLOT train and valid --------------
	fig = plot_traj(one_task['goal'], one_train[:traj_len,:], one_valid[:traj_len,:])    # (99, 4)
	fig.savefig('./test.png', bbox_inches='tight')
	# ------- Finish Ploting -------------------



	# ------ ANIMATE valid --------------
	# # one_valid = one_train # uncomment this one to plot train
	# ped_posi_list = [one_valid[:, :2]]
	# for i in range(ped_num):
	# 	ped_posi_list.append(one_valid[:traj_len-1, self_state+i*ped_state : (self_state+2)+i*ped_state])


	# one_valid = np.hstack(ped_posi_list)


 # #    # Uncomment the function below to do animation
	# anim_traj(one_task['goal'], np.transpose(one_valid), ped_num)



if __name__ == '__main__':
	main()


