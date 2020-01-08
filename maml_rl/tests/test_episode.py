import pytest

import torch

from maml_rl.episode import BatchEpisodes
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.samplers.sampler import make_env
from maml_rl.tests.utils import make_unittest_env


@pytest.mark.parametrize('batch_size', [1, 8])
def test_batch_episodes(batch_size):
    episodes = BatchEpisodes(batch_size, gamma=0.95)
    assert episodes.batch_size == batch_size
    assert episodes.gamma == 0.95


def test_batch_episodes_append():
    lengths = [2, 3, 7, 5, 13, 11, 17]
    envs = SyncVectorEnv([make_unittest_env(length) for length in lengths])
    episodes = BatchEpisodes(batch_size=len(lengths), gamma=0.95)

    observations = envs.reset()
    while not envs.dones.all():
        actions = [envs.single_action_space.sample() for _ in observations]
        new_observations, rewards, _, infos = envs.step(actions)
        episodes.append(observations, actions, rewards, infos['batch_ids'])
        observations = new_observations

    assert len(episodes) == 17
    assert episodes.lengths == lengths
    
    # Observations
    assert episodes.observations.shape == (17, 7, 64, 64, 3)
    assert episodes.observations.dtype == torch.float32
    for i in range(7):
        length = lengths[i]
        assert (episodes.observations[length:, i] == 0).all()

    # Actions
    assert episodes.actions.shape == (17, 7, 2)
    assert episodes.actions.dtype == torch.float32
    for i in range(7):
        length = lengths[i]
        assert (episodes.actions[length:, i] == 0).all()

    # Mask
    assert episodes.mask.shape == (17, 7)
    assert episodes.mask.dtype == torch.float32
    for i in range(7):
        length = lengths[i]
        assert (episodes.mask[:length, i] == 1).all()
        assert (episodes.mask[length:, i] == 0).all()
