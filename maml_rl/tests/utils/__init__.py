import gym
import numpy as np

from gym.spaces import Box


HEIGHT, WIDTH = 64, 64

def make_unittest_env(length):
    def _make_env():
        return UnittestEnv(length)
    return _make_env

class UnittestEnv(gym.Env):
    def __init__(self, max_length):
        super(UnittestEnv, self).__init__()
        self.max_length = max_length
        self._length = 0
        
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(HEIGHT, WIDTH, 3),
                                     dtype=np.uint8)
        self.action_space = Box(low=0., high=1., shape=(2,), dtype=np.float32)

    def reset_task(self):
        pass

    def reset(self):
        self._length = 0
        return self.observation_space.sample()

    def step(self, action):
        observation = self.observation_space.sample()
        self._length += 1
        reward, done = 0, (self._length >= self.max_length)
        return (observation, reward, done, {})
