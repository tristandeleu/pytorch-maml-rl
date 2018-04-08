import maml_rl.envs
import gym
import numpy as np
import torch.optim as optim

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline

from tqdm import trange

env = gym.make('Bandit-K50-v0')
policy = CategoricalMLPPolicy(env.observation_space.shape[0],
    env.action_space.n, hidden_sizes=(32, 32))
baseline = LinearFeatureBaseline(env.observation_space.shape[0])

metalearner = MetaLearner('Bandit-K50-v0', policy, baseline, fast_batch_size=100)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

all_rewards = []
for epoch in trange(1000):
    optimizer.zero_grad()
    loss, rewards = metalearner.loss(meta_batch_size=20)
    loss.backward()
    optimizer.step()
    all_rewards.append(rewards.data[0])

with open('tmp/rewards.npy', 'wb') as f:
    np.save(f, np.asarray(all_rewards))