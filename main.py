import maml_rl.envs
import gym
import numpy as np
import torch

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tqdm import trange, tqdm

sampler = BatchSampler('Bandit-K50-v0', batch_size=100)
policy = CategoricalMLPPolicy(sampler.envs.observation_space.shape[0],
    sampler.envs.action_space.n, hidden_sizes=(32, 32))
baseline = LinearFeatureBaseline(sampler.envs.observation_space.shape[0])

metalearner = MetaLearner(sampler, policy, baseline)

all_rewards = []
for epoch in trange(200):
    episodes = metalearner.sample(meta_batch_size=20)
    metalearner.step(episodes)
    rewards = torch.mean(torch.cat([torch.sum(valid_episodes.rewards)
        for _, valid_episodes in episodes], dim=0))
    all_rewards.append(rewards.data[0])
    tqdm.write('{0}'.format(rewards.data[0]))

with open('tmp/rewards.npy', 'wb') as f:
    np.save(f, np.asarray(all_rewards))