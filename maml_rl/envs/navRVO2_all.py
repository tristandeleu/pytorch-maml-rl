import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import rvo2


class NavRVO2Env_all(gym.Env):
    """
    What's new for the new environment:
    Added 4 pedestrians initialized to be at 4 corners ([-0.8,-0.8], [0.8,-0.8], [0.8,0.8], [-0.8,0.8]) 
    of a rectangle centering at the origin. 2 pedestrians at each corner. They walk almostly 
    diagonally towards the other side (specific direction is upon randomness). After they exit the rectangle, 
    they will be initialized at the corners again. 
    """
    
    def __init__(self, task={}):
        super(NavRVO2Env_all, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(4, dtype=np.float32)
        self.seed()

        self._num_ped = 4
        # self._ped_speed = task.get('ped_speed', np.float32(0))
        self._ped_speed = 0.1
        self._ped_direc = task.get('ped_direc', np.zeros(self._num_ped, dtype=np.float32))
        self._ped_list = []
        # self._n_pedestrian = 8 # or use np.random.randint but needs to adjust _ped_states
        self._entering_corner = np.float32(1.)
        self._default_ped_states = self._entering_corner * np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
        self._ped_states = self._default_ped_states.copy()
        self._ped_histories = []
        
        # Initializing RVO2 simulator
        timeStep = 1.
        neighborDist = 0.2
        maxNeighbors = 5
        timeHorizon = 1.0
        timeHorizonObst = timeHorizon
        radius = 0.05
        maxSpeed = 1.2 * self._ped_speed
        
        self._simulator = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)
        for i in range(self._num_ped):
            ai = self._simulator.addAgent((self._default_ped_states[i,0], self._default_ped_states[i,1]))
            self._ped_list.append(ai) 
            vx = self._ped_speed * np.cos(self._ped_direc[i])
            vy = self._ped_speed * np.sin(self._ped_direc[i])
            self._simulator.setAgentPrefVelocity(ai, (vx, vy))
        # print('navRVO2: Initialized environment with %f RVO2-agents.', self._num_ped)

    def check_and_clip_ped_states(self):
        # update simlator when an agent gets out of boundary
        ai_list = []
        for i in range(self._num_ped):
            if any(abs(self._ped_states[i,:]) >= self._entering_corner + 0.001):
                self._ped_states[[i, i], [0, 1]] = self._default_ped_states[i,:]
                self._ped_direc[i] = np.arctan2(-self._ped_states[i,1], -self._ped_states[i,0]) + np.random.uniform(-np.pi/4, np.pi/4, size=(1,1)) 
                ai_list.append(i)
        
        if ai_list:
            self.update_simulator(ai_list)

        return ai_list

    def print_rvo2_states(self):
        print("Printing agent-states from rvo-2")
        for i in range(self._num_ped):
            ai = self._ped_list[i]
            print("Agent", ai,": pos=", self._simulator.getAgentPosition(ai), ", vel=", self._simulator.getAgentVelocity(ai))

    def print_ped_states(self):
        print("Printing agent-states from self._ped_states")
        for i in range(self._num_ped):
            print("Agent", i,": pos=", self._ped_states[i])

    def print_robot_state(self):
        print("Robot: pos=", self._state[0:2])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        # tasks includes various goal_pos and ped_direcs
        # goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        goals = self.np_random.uniform(0.5, 0.5, size=(num_tasks, 2))

        ped_speeds = self.np_random.uniform(0.1, 0.1, size=num_tasks)

        ped_direc = np.arctan2(-self._ped_states[:,1], -self._ped_states[:,0])
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
                self._simulator.setAgentPosition(ai, (self._ped_states[ai,0], self._ped_states[ai,1]))
                vx = self._ped_speed * np.cos(self._ped_direc[ai])
                vy = self._ped_speed * np.sin(self._ped_direc[ai])
                self._simulator.setAgentVelocity(ai, (vx, vy))
                self._simulator.setAgentPrefVelocity(ai, (vx, vy))
        else: # update all agents from _ped_states
            for ai in self._ped_list:
                self._simulator.setAgentPosition(ai, (self._ped_states[ai,0], self._ped_states[ai,1]))
                vx = self._ped_speed * np.cos(self._ped_direc[ai])
                vy = self._ped_speed * np.sin(self._ped_direc[ai])
                self._simulator.setAgentVelocity(ai, (vx, vy))
                self._simulator.setAgentPrefVelocity(ai, (vx, vy))
                # print("ped i velocity = ", self._simulator.getAgentVelocity(ai)

    def assert_sim_and_states(self):
        for i in self._ped_list:
            if self._ped_states[i, 0] != self._simulator.getAgentPosition(i)[0]:
                print("Error: X for agent ", i, ": state = ", self._ped_states[i, 0], ", sim = ", self._simulator.getAgentPosition(i)[0])
                return False
            if self._ped_states[i, 1] != self._simulator.getAgentPosition(i)[1]:
                print("Error: Y for agent ", i, ": state = ", self._ped_states[i, 1], ", sim = ", self._simulator.getAgentPosition(i)[1])
                return False
        return True

    def update_ped_states(self):
        # Update ped_states from simulator     
        for i in self._ped_list:
            self._ped_states[i, 0] = self._simulator.getAgentPosition(i)[0]
            self._ped_states[i, 1] = self._simulator.getAgentPosition(i)[1]
            

    def reset(self, env=True):
        self._state = np.zeros(4, dtype=np.float32)
        self._ped_histories = []
        self._ped_states = self._default_ped_states
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)

        try: # for debugging. Not sure why it gives assertion error sometimes in the middle of training...
            assert self.action_space.contains(action)
        except AssertionError as error:
            print("AssertionError: action is {}".format(action))

        # Update robot's state
        # print(self._state[0:1], action)
        self._state[0:2] = self._state[0:2] + action
        self._state[2:4] = [self._ped_states[0,0], self._ped_states[0,1]]

        dx = self._state[0] - self._goal[0]
        dy = self._state[1] - self._goal[1]

        # Update agents' state
        self._simulator.doStep()
        self.update_ped_states()
        self.check_and_clip_ped_states() # ensure all agents are within the bounary: reset to default pos if necessary
        
        # Calculate rewards
        reward = -np.sqrt(dx ** 2 + dy ** 2)
        safe_dist = 0.2
        weight = 0.2
        ped_dists = np.sqrt((self._ped_states[:,0] - self._state[0]) ** 2 + (self._ped_states[:,1] - self._state[1]) ** 2)

        for dist in ped_dists[ped_dists < safe_dist]:
            reward -= weight*(safe_dist - dist)

        done = ((np.abs(dx) < 0.01) and (np.abs(dy) < 0.01))
        if done:
            reward += 0.5;
        # self._ped_histories = self._ped_histories.append(self._ped_states[0])

        return self._state, reward, done, self._task
