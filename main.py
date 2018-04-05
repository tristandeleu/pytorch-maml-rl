
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from meta_learner import MetaLearner
import envs
import gym

env = gym.make('TabularMDP-v0')
meta_learner = MetaLearner(env)
print ("[*] meta learner created")
meta_learner.meta_train()
print ("[*] trained")
