import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(),
                 nonlinearity=F.relu,
                 init_std=1.0,
                 min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size,
                                              output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        mu = F.linear(output,
                      weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Independent(Normal(loc=mu, scale=scale), 1)
