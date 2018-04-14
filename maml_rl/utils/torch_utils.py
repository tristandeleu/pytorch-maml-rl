import torch
# TODO: Replace by torch.distributions in Pytorch 0.4
from maml_rl.distributions import Categorical, Normal

def weighted_mean(tensor, weights=None):
    if weights is None:
        return torch.mean(tensor)
    sum_weights = torch.sum(weights)
    return torch.sum(tensor * weights) / sum_weights

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution
