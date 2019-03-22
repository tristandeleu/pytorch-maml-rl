import pytest

import numpy as np
import torch

from maml_rl.utils.torch_utils import weighted_mean, weighted_normalize

def test_weighted_mean_no_dim():
    lengths = [2, 3, 5, 7, 11]
    # Inputs
    inputs_np = np.random.rand(5, 13).astype(np.float32)
    weights_np = np.zeros((5, 13), dtype=np.float32)
    for i, length in enumerate(lengths):
        inputs_np[i, length:] = 0.
        weights_np[i, :length] = 1.
    # Pytorch
    inputs_th = torch.from_numpy(inputs_np)
    weights_th = torch.from_numpy(weights_np)
    mean_th = weighted_mean(inputs_th, dim=None, weights=weights_th)
    # Numpy
    sum_np, num_np = 0., 0.
    for i in range(5):
        for j in range(13):
            sum_np += inputs_np[i, j] * weights_np[i, j]
            num_np += weights_np[i, j]
    mean_np = sum_np / num_np

    assert mean_th.dim() == 0
    assert np.allclose(mean_th.item(), mean_np)

def test_weighted_mean_dim():
    lengths = [2, 3, 5, 7, 11]
    # Inputs
    inputs_np = np.random.rand(5, 13).astype(np.float32)
    weights_np = np.zeros((5, 13), dtype=np.float32)
    for i, length in enumerate(lengths):
        inputs_np[i, length:] = 0.
        weights_np[i, :length] = 1.
    # Pytorch
    inputs_th = torch.from_numpy(inputs_np)
    weights_th = torch.from_numpy(weights_np)
    mean_th = weighted_mean(inputs_th, dim=1, weights=weights_th)
    # Numpy
    sum_np = np.zeros((5,), dtype=np.float32)
    for i in range(5):
        num_np = 0.
        for j in range(13):
            sum_np[i] += inputs_np[i, j] * weights_np[i, j]
            num_np += weights_np[i, j]
        sum_np[i] /= num_np
    mean_np = np.mean(sum_np)

    assert mean_th.dim() == 0
    assert np.allclose(mean_th.item(), mean_np)
