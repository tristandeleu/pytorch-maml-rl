import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class Navigation2DEnv(gym.Env):
    def __init__(self):
        super(Navigation2DEnv, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = None
        self._goal = None
        self._state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, self._task
