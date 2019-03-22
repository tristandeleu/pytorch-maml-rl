import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import rvo2


class NavRVO2Env_all(gym.Env):
    """
    What's new for the new environment:
    Added 4 pedestrians initialized to be at 4 corners ([-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5,0.5]) 
    of a rectangle centering at the origin. 1 pedestrians at each corner. They walk almostly 
    diagonally towards the other side (specific direction is upon randomness). After they exit the rectangle, 
    they will be initialized at the corners again. 

    robot state: 
    'px', 'py', 'vx', 'vy', 'gx', 'gy'
     0     1      2     3     4     5    

    pedestrian state: 
    'px1', 'py1', 'vx1', 'vy1'
      6      7      8       9  
    """
    
    def __init__(self, task={}):
        super(NavRVO2Env_all, self).__init__()

        self._num_ped = 4
        self._num_agent = self._num_ped + 1 # ped_num + robot_num

        self._robot_state_dim = 6
        self._ped_state_dim = 4

        self._state_dim = self._robot_state_dim + self._num_ped * self._ped_state_dim # robot_state_dim + ped_num * ped_state_dim = 6 + 4 * 4 = 22

        # ------------ Remember to revise self_state update in step()!!!!!!! -------------

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self._task = task
        self._goal = task.get('goal', np.array([0., 0.], dtype=np.float32))
        self._default_robot_state = np.array([0., 0., 0., 0., self._goal[0], self._goal[1]], dtype=np.float32)
        self._state = self._default_robot_state.copy()
        self.seed()
        self._done = False

        self._ped_radius = 0.15 # safety distance to start reacting to others
        self._ped_speed = task.get('ped_speed', np.zeros(self._num_ped, dtype=np.float32))
        self._ped_direc = task.get('ped_direc', np.zeros(self._num_ped, dtype=np.float32))
        self._entering_corner = np.float32(0.8)
        self._default_ped_poses = self._entering_corner * np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
        self._ped_poses = self._default_ped_poses.copy()

        self._ped_list = [] # pointer list that each element points to an agent in simulator   
        self._simulator = self.init_simulator()

        for i in range(self._num_ped): # Extrating values from simulator and init self._state
            ai = self._ped_list[i]
            ai_vel = self._simulator.getAgentVelocity(ai)
            ai_pos = self._simulator.getAgentPosition(ai)
            self._state = np.append(self._state, np.append([ai_pos[0], ai_pos[1]], [ai_vel[0], ai_vel[1]]))


    def init_simulator(self):
        # Initializing RVO2 simulator && add agents to self._ped_list
        self._ped_list = []
        timeStep = 1.
        neighborDist = self._ped_radius # safe-radius to observe states
        maxNeighbors = 4
        timeHorizon = 2.0
        timeHorizonObst = timeHorizon
        radius = 0.05 # size of the agent
        maxSpeed = 0.2 
        
        sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)
        for i in range(self._num_ped):
            ai = sim.addAgent((self._default_ped_poses[i,0], self._default_ped_poses[i,1]))
            self._ped_list.append(ai)
            vx = self._ped_speed[i] * np.cos(self._ped_direc[i])
            vy = self._ped_speed[i] * np.sin(self._ped_direc[i])
            sim.setAgentPrefVelocity(ai, (vx, vy))

        # print('navRVO2: Initialized environment with %f RVO2-agents.', self._num_ped)
        return sim


    def check_and_clip_ped_poses(self):
        # update simlator when an agent gets out of boundary
        ai_list = []
        for i in range(self._num_ped):
            if any(abs(self._ped_poses[i,:]) > self._entering_corner + 0.001):
                self._ped_poses[[i, i], [0, 1]] = self._default_ped_poses[i,:]
                self._ped_direc[i] = np.arctan2(-self._ped_poses[i,1], -self._ped_poses[i,0]) + np.random.uniform(-np.pi/4, np.pi/4, size=(1,1)) 
                ai_list.append(i)
        
        if ai_list:
            self.update_simulator(ai_list)

        return ai_list

    def print_rvo2_states(self):
        print("Printing agent-states from rvo-2")
        for i in range(self._num_ped):
            ai = self._ped_list[i]
            print("Agent", ai,": pos=", self._simulator.getAgentPosition(ai), ", vel=", self._simulator.getAgentVelocity(ai))

    def print_ped_poses(self):
        print("Printing agent-states from self._ped_poses")
        for i in range(self._num_ped):
            print("Agent", i,": pos=", self._ped_poses[i])

    def print_robot_state(self):
        print("Robot: pos=", self._state[0:2])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        # tasks includes various goal_pos and ped_direcs
        # goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        # goals = self.np_random.uniform(0.3, 0.5, size=(num_tasks, 2))

        goal_range = [-1., -0.8, 0.8, 1.]
        rand_temp = self.np_random.uniform(goal_range[0]-goal_range[1], goal_range[3]-goal_range[2], size=(num_tasks,))
        rand_temp = rand_temp + goal_range[2] * np.sign(rand_temp) # there's a chance that 0 is sampled, but that's okay
        free_axis = np.random.randint(2, size=num_tasks)

        goals =  np.zeros((num_tasks, 2), dtype=np.float32)
        goals[range(num_tasks),free_axis] = rand_temp
        goals[range(num_tasks),1-free_axis] = self.np_random.uniform(-1., 1., size=(num_tasks,))


        ped_speeds = self.np_random.uniform(0.03, 0.15, size=(num_tasks, self._num_ped))

        ped_direc = np.arctan2(-self._ped_poses[:,1], -self._ped_poses[:,0])
        ram_direcs = self.np_random.uniform(-np.pi/4, np.pi/4, size=(num_tasks, self._num_ped)) # 8 pedestrians
        ped_direcs = ram_direcs + ped_direc

        tasks = [{'goal': goal, 'ped_speed': ped_speed, 'ped_direc': ped_direc} for goal, ped_speed, ped_direc in zip(goals, ped_speeds, ped_direcs)]

        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']
        self._ped_speed = task['ped_speed']
        self._ped_direc = task['ped_direc']
        self.update_simulator(self._ped_list)


    def update_simulator(self, ai_list=[]):
        if ai_list: #only update agents in ai_list
            for ai in ai_list:
                self._simulator.setAgentPosition(ai, (self._ped_poses[ai,0], self._ped_poses[ai,1]))
                vx = self._ped_speed[ai] * np.cos(self._ped_direc[ai])
                vy = self._ped_speed[ai] * np.sin(self._ped_direc[ai])
                self._simulator.setAgentVelocity(ai, (vx, vy))
                self._simulator.setAgentPrefVelocity(ai, (vx, vy))
        else: # update all agents from _ped_poses
            for ai in self._ped_list:
                self._simulator.setAgentPosition(ai, (self._ped_poses[ai,0], self._ped_poses[ai,1]))
                vx = self._ped_speed[ai] * np.cos(self._ped_direc[ai])
                vy = self._ped_speed[ai] * np.sin(self._ped_direc[ai])
                self._simulator.setAgentVelocity(ai, (vx, vy))
                self._simulator.setAgentPrefVelocity(ai, (vx, vy))
                # print("ped i velocity = ", self._simulator.getAgentVelocity(ai)

    def assert_sim_and_states(self):
        for i in self._ped_list:
            if self._ped_poses[i, 0] != self._simulator.getAgentPosition(i)[0]:
                print("Error: X for agent ", i, ": state = ", self._ped_poses[i, 0], ", sim = ", self._simulator.getAgentPosition(i)[0])
                return False
            if self._ped_poses[i, 1] != self._simulator.getAgentPosition(i)[1]:
                print("Error: Y for agent ", i, ": state = ", self._ped_poses[i, 1], ", sim = ", self._simulator.getAgentPosition(i)[1])
                return False
        return True

    def update_ped_poses(self):
        # Update ped_poses from simulator     
        for i in self._ped_list:
            self._ped_poses[i, 0] = self._simulator.getAgentPosition(i)[0]
            self._ped_poses[i, 1] = self._simulator.getAgentPosition(i)[1]
            

    def reset(self, env=True):
        self._done = False
        self._state = self._default_robot_state.copy()
        self._ped_poses = self._default_ped_poses.copy()

        try:
            ped_direc = task.get('ped_direc', np.zeros(self._num_ped, dtype=np.float32))
        except:
            ped_direc = np.zeros(self._num_ped, dtype=np.float32)

        for i in range(self._num_ped):
            vx = self._ped_speed[i] * np.cos(ped_direc[i])
            vy = self._ped_speed[i] * np.sin(ped_direc[i])
            self._state = np.append(self._state, np.append(self._default_ped_poses[i], [vx, vy]))
        self._simulator = self.init_simulator()
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)

        try: # for debugging. Not sure why it gives assertion error sometimes in the middle of training...
            assert self.action_space.contains(action)
        except AssertionError as error:
            print("AssertionError: action is {}".format(action))

        # Update robot's state
        self._state[0:2] = self._state[0:2] + action
        self._state[2:4] = action
        temp_state = self._state[4:6]
        self._state[4:6] = self._goal
        # print(temp_state, self._state[4:6])
        # assert temp_state == self._state[4:6]

        # Update agents' state
        self._simulator.doStep()
        self.update_ped_poses()
        self.check_and_clip_ped_poses() # ensure all agents are within the bounary: reset to default pos if necessary

        mid_point = self._goal/2.
        real_ped_state = self._ped_poses + mid_point
        
        # update self._state
        for i in range(self._num_ped):
            ai_velocity = self._simulator.getAgentVelocity(self._ped_list[i])
            self._state[self._robot_state_dim+i*self._ped_state_dim: self._robot_state_dim+i*self._ped_state_dim+4] = np.append(real_ped_state[i,:], [ai_velocity[0], ai_velocity[1]])
        
        # Calculate rewards
        weight_goal = 1.0
        dx = self._state[0] - self._goal[0]
        dy = self._state[1] - self._goal[1]
        reward_goal = -weight_goal * np.sqrt(dx ** 2 + dy ** 2)
        
        weight_boundary = 0.1
        reward_boundary = 0.
        dist_out = np.maximum(np.abs(self._state[0]), np.abs(self._state[1]))
        if (dist_out > 1.2):
            reward_boundary = - np.abs(dist_out - 1.2) * weight_boundary

        weight_ped = 0.2
        weight_colli = 1.5
        reward_ped = 0.
        for i in range(self._num_ped):
            dist_ped_i = np.sqrt((real_ped_state[i,0] - self._state[0]) ** 2 + (real_ped_state[i,1] - self._state[1]) ** 2)
            if (dist_ped_i < self._ped_radius): # safe distance to pealize the robot
                reward_ped = reward_ped + (dist_ped_i - self._ped_radius) * weight_ped
            if (dist_ped_i < 0.05): # collision with an agent
                reward_ped = reward_ped + (-1) * weight_colli

        reward_sum = reward_goal + reward_boundary + reward_ped
        all_reward = np.array([reward_sum, reward_goal, reward_ped])
        
        # Check if reached goal
        # if self._done:
        #     done = True
        #     self._done = False
        # elif ((np.abs(dx) < 0.05) and (np.abs(dy) < 0.05)):
        #     self._done = True
        #     done = False
        # else:
        #     done = False
        done = ((np.abs(dx) < 0.05) and (np.abs(dy) < 0.05))


        return self._state, all_reward, done, self._task
