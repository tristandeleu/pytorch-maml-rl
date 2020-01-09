import numpy as np
import torch
import torch.nn.functional as F

from maml_rl.utils.torch_utils import weighted_normalize

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]

        self._observation_shape = None
        self._action_shape = None

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self._advantages = None
        self._lengths = None
        self._logs = {}

    @property
    def observation_shape(self):
        if self._observation_shape is None:
            self._observation_shape = self.observations.shape[2:]
        return self._observation_shape

    @property
    def action_shape(self):
        if self._action_shape is None:
            self._action_shape = self.actions.shape[2:]
        return self._action_shape

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size) + observation_shape,
                                    dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._observations_list[i],
                         axis=0,
                         out=observations[:length, i])
            self._observations = torch.as_tensor(observations, device=self.device)
            del self._observations_list
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size) + action_shape,
                               dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._actions_list[i], axis=0, out=actions[:length, i])
            self._actions = torch.as_tensor(actions, device=self.device)
            del self._actions_list
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._rewards_list[i], axis=0, out=rewards[:length, i])
            self._rewards = torch.as_tensor(rewards, device=self.device)
            del self._rewards_list
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            self._returns = torch.zeros_like(self.rewards)
            return_ = torch.zeros((self.batch_size,), dtype=torch.float32)
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + self.rewards[i] * self.mask[i]
                self._returns[i] = return_
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            self._mask = torch.zeros((len(self), self.batch_size),
                                     dtype=torch.float32,
                                     device=self.device)
            for i in range(self.batch_size):
                length = self.lengths[i]
                self._mask[:length, i].fill_(1.0)
        return self._mask

    @property
    def advantages(self):
        if self._advantages is None:
            raise ValueError('The advantages have not been computed. Call the '
                             'function `episodes.compute_advantages(baseline)` '
                             'to compute and store the advantages in `episodes`.')
        return self._advantages

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_id in zip(
                observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))

    @property
    def logs(self):
        return self._logs

    def log(self, key, value):
        self._logs[key] = value

    def compute_advantages(self, baseline, gae_lambda=1.0, normalize=True):
        # Compute the values based on the baseline
        values = baseline(self).detach()
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        # Compute the advantages based on the values
        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        self._advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros((self.batch_size,), dtype=torch.float32)
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i]
            self._advantages[i] = gae

        # Normalize the advantages
        if normalize:
            self._advantages = weighted_normalize(self._advantages,
                                                  lengths=self.lengths)
        # Once the advantages are computed, the returns are not necessary
        # anymore (only to compute the parameters of the baseline)
        del self._returns
        del self._mask

        return self.advantages

    @property
    def lengths(self):
        if self._lengths is None:
            self._lengths = [len(rewards) for rewards in self._rewards_list]
        return self._lengths

    def __len__(self):
        return max(self.lengths)
