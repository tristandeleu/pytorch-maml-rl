import numpy as np
import gym
from gym import spaces

class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizedActionWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
            shape=self.env.action_space.shape)

    def action(self, action):
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        # Map the normalized action to original action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = lb + 0.5 * (action + 1.0) * (ub - lb)
        return action

    def reverse_action(self, action):
        # Map the original action to normalized action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = 2.0 * (action - lb) / (ub - lb) - 1.0
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action
