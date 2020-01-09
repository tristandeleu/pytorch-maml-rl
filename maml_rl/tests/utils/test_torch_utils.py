import pytest

import numpy as np
import torch
import torch.nn as nn

from maml_rl.utils.torch_utils import (weighted_mean, weighted_normalize,
                                       vector_to_parameters)


def test_weighted_mean():
    lengths = [2, 3, 7, 5, 11]
    # Inputs
    inputs_np = np.random.rand(13, 5).astype(np.float32)
    for i, length in enumerate(lengths):
        inputs_np[length:, i] = 0.

    # Pytorch
    inputs_th = torch.as_tensor(inputs_np)
    mean_th = weighted_mean(inputs_th, lengths=lengths)

    # Numpy
    mean_np = np.zeros((5,), dtype=np.float32)
    for i, length in enumerate(lengths):
        for j in range(13):
            if j < length:
                mean_np[i] += inputs_np[j, i]
        mean_np[i] /= length

    assert mean_th.dim() == 1
    assert mean_th.shape == (5,)
    np.testing.assert_allclose(mean_th.detach().numpy(), mean_np)


def test_weighted_mean_multi_dimensional():
    lengths = [2, 3, 7, 5, 11]
    # Inputs
    inputs_np = np.random.rand(13, 5, 17, 19).astype(np.float32)
    for i, length in enumerate(lengths):
        inputs_np[length:, i] = 0.

    # Pytorch
    inputs_th = torch.as_tensor(inputs_np)
    mean_th = weighted_mean(inputs_th, lengths=lengths)

    # Numpy
    mean_np = np.zeros((5, 17, 19), dtype=np.float32)
    for i, length in enumerate(lengths):
        for j in range(13):
            if j < length:
                mean_np[i] += inputs_np[j, i]
        mean_np[i] /= length

    assert mean_th.dim() == 3
    assert mean_th.shape == (5, 17, 19)
    np.testing.assert_allclose(mean_th.detach().numpy(), mean_np)


def test_weighted_mean_side_effect():
    lengths = [2, 3, 7, 5, 11]
    # Inputs
    inputs_np = np.random.rand(13, 5).astype(np.float32)

    # Pytorch
    inputs_th = torch.as_tensor(inputs_np)
    mean_th = weighted_mean(inputs_th, lengths=lengths)

    for i, length in enumerate(lengths):
        assert (inputs_th[length:, i] == 0.).all()
        assert (inputs_np[length:, i] == 0.).all()


def test_weighted_normalize():
    lengths = [2, 3, 7, 5, 11]
    # Inputs
    inputs_np = np.random.rand(13, 5).astype(np.float32)

    # Pytorch
    inputs_th = torch.as_tensor(inputs_np)
    normalized_th = weighted_normalize(inputs_th, lengths=lengths)

    for i, length in enumerate(lengths):
        assert (normalized_th[length:, i] == 0.).all()


def test_vector_to_parameters_no_shared_memory():
    model = nn.Sequential(
        nn.Linear(2, 3, bias=True),
        nn.Linear(3, 5, bias=True))
    num_params = (2 * 3) + 3 + (3 * 5) + 5
    vector_np = np.random.rand(num_params).astype(np.float32)
    vector = torch.as_tensor(vector_np)

    vector_to_parameters(vector, model.parameters())
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param_np = param.view(-1).detach().numpy()

        np.testing.assert_array_equal(param_np, vector_np[pointer:pointer + num_param])

        pointer += num_param


def test_vector_to_parameters_shared_memory():
    model = nn.Sequential(
        nn.Linear(2, 3, bias=True),
        nn.Linear(3, 5, bias=True))
    model.share_memory()

    for param in model.parameters():
        assert param.data.is_shared()

    num_params = (2 * 3) + 3 + (3 * 5) + 5
    vector_np = np.random.rand(num_params).astype(np.float32)
    vector = torch.as_tensor(vector_np)

    vector_to_parameters(vector, model.parameters())
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param_np = param.view(-1).detach().numpy()

        np.testing.assert_array_equal(param_np, vector_np[pointer:pointer + num_param])
        assert param.data.is_shared()

        pointer += num_param
