import torch
from torch.distributions import Categorical, Normal

def weighted_mean(tensor, dim=None, weights=None):
    if weights is None:
        return torch.mean(tensor)
    if dim is None:
        sum_weights = torch.sum(weights)
        return torch.sum(tensor * weights) / sum_weights
    if isinstance(dim, int):
        dim = (dim,)
    numerator = tensor * weights
    denominator = weights
    for dimension in dim:
        numerator = torch.sum(numerator, dimension, keepdim=True)
        denominator = torch.sum(denominator, dimension, keepdim=True)
    return numerator / denominator

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution

def weighted_normalize(tensor, dim=None, weights=None, epsilon=1e-8):
    if weights is None:
        weights = torch.ones_like(tensor)
    mean = weighted_mean(tensor, dim=dim, weights=weights)
    centered = tensor * weights - mean
    std = torch.sqrt(weighted_mean(centered ** 2, dim=dim, weights=weights))
    return centered / (std + epsilon)
