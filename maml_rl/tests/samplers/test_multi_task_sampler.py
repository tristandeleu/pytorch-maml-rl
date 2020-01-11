import pytest

import numpy as np
import gym

import maml_rl.envs
from maml_rl.samplers import MultiTaskSampler
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.utils.helpers import get_policy_for_env, get_input_size


@pytest.mark.parametrize('env_name', ['TabularMDP-v0', '2DNavigation-v0'])
@pytest.mark.parametrize('num_workers', [1, 4])
def test_init(env_name, num_workers):
    batch_size = 10
    # Environment
    env = gym.make(env_name)
    env.close()
    # Policy and Baseline
    policy = get_policy_for_env(env)
    baseline = LinearFeatureBaseline(get_input_size(env))

    sampler = MultiTaskSampler(env_name,
                               {}, # env_kwargs
                               batch_size,
                               policy,
                               baseline,
                               num_workers=num_workers)
    sampler.close()


@pytest.mark.parametrize('env_name', ['TabularMDP-v0', '2DNavigation-v0'])
@pytest.mark.parametrize('batch_size', [1, 7])
@pytest.mark.parametrize('num_tasks', [1, 5])
@pytest.mark.parametrize('num_steps', [1, 3])
@pytest.mark.parametrize('num_workers', [1, 3])
def test_sample(env_name, batch_size, num_tasks, num_steps, num_workers):
    # Environment
    env = gym.make(env_name)
    env.close()
    # Policy and Baseline
    policy = get_policy_for_env(env)
    baseline = LinearFeatureBaseline(get_input_size(env))

    sampler = MultiTaskSampler(env_name,
                               {}, # env_kwargs
                               batch_size,
                               policy,
                               baseline,
                               num_workers=num_workers)
    tasks = sampler.sample_tasks(num_tasks=num_tasks)
    train_episodes, valid_episodes = sampler.sample(tasks,
                                                    num_steps=num_steps)
    sampler.close()

    assert len(train_episodes) == num_steps
    assert len(train_episodes[0]) == num_tasks
    assert isinstance(train_episodes[0][0], BatchEpisodes)

    assert len(valid_episodes) == num_tasks
    assert isinstance(valid_episodes[0], BatchEpisodes)
