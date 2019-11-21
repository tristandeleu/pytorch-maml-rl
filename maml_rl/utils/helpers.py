import gym
import torch.nn.functional as F

from functools import reduce
from operator import mul

from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy


def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = reduce(mul, env.observation_space.shape, 1)
    nonlinearity = getattr(F, nonlinearity)

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size,
                                 output_size,
                                 hidden_sizes=hidden_sizes,
                                 nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size,
                                      output_size,
                                      hidden_sizes=hidden_sizes,
                                      nonlinearity=nonlinearity)
    return policy
