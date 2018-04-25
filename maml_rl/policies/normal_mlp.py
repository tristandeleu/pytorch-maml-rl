import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy

class NormalMLPPolicy(Policy):
    def __init__(self, input_size, output_size,
                 hidden_sizes=(), nonlinearity=F.relu):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.add_module('mu', nn.Linear(layer_sizes[-1], output_size))
        self.add_module('sigma', nn.Linear(layer_sizes[-1], output_size))

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
            bias=params['mu.bias'])
        sigma = F.linear(output, weight=params['sigma.weight'],
            bias=params['sigma.bias'])

        return Normal(loc=mu, scale=F.softplus(sigma))
