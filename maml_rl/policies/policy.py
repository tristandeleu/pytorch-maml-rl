import torch
import torch.nn as nn

from collections import OrderedDict

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    def state_dict(self, params=None, destination=None, prefix='', keep_vars=False):
        if params is None:
            destination = super(Policy, self).state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)
        else:
            if destination is None:
                destination = OrderedDict()
            for name, param in params.items():
                if param is not None:
                    destination[prefix + name] = param if keep_vars else param.data

        return destination
