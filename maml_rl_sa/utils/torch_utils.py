import torch
from torch.distributions import Categorical, Normal

def weighted_mean(tensor, dim=None, weights=None):
    if weights is None:
        out = torch.mean(tensor)
    if dim is None:
        out = torch.sum(tensor * weights)
        out.div_(torch.sum(weights))
    else:
        mean_dim = torch.sum(tensor * weights, dim=dim)
        mean_dim.div_(torch.sum(weights, dim=dim))
        out = torch.mean(mean_dim)
    return out

def weighted_normalize(tensor, dim=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, dim=dim, weights=weights)
    out = tensor * (1 if weights is None else weights) - mean
    std = torch.sqrt(weighted_mean(out ** 2, dim=dim, weights=weights))
    out.div_(std + epsilon)
    return out

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution
