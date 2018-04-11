import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BatchEpisodes(object):
    def __init__(self, gamma=0.95, is_cuda=False, volatile=False):
        self.gamma = gamma
        self.is_cuda = is_cuda
        self._volatile = volatile
        
        self._observations_list = []
        self._actions_list = []
        self._rewards_list = []
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None

    @property
    def batch_size(self):
        if not self._rewards_list:
            raise ValueError('The batch size of an empty batch '
                             'of episodes is undefined')
        return self._rewards_list[0].shape[0]

    @property
    def observations(self):
        if self._observations is None:
            observations = np.stack(self._observations_list, axis=0)
            self._observations = torch.from_numpy(observations).float()
            if self.is_cuda:
                self._observations = self._observations.cuda()
        return Variable(self._observations, volatile=self._volatile)

    @property
    def actions(self):
        if self._actions is None:
            actions = np.stack(self._actions_list, axis=0)
            self._actions = torch.from_numpy(actions).float()
            if self.is_cuda:
                self._actions = self._actions.cuda()
        return Variable(self._actions, volatile=self._volatile)

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.stack(self._rewards_list, axis=0)
            self._rewards = torch.from_numpy(rewards).float()
            if self.is_cuda:
                self._rewards = self._rewards.cuda()
        return Variable(self._rewards, volatile=self._volatile)

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(len(self) - 1, -1, -1):
                return_ = (self.gamma * return_
                    + self._rewards_list[i] * self._mask_list[i])
                returns[i] = return_
            self._returns = torch.from_numpy(returns).float()
            if self.is_cuda:
                self._returns = self._returns.cuda()
        return Variable(self._returns, volatile=self._volatile)

    @property
    def mask(self):
        if self._mask is None:
            mask = np.stack(self._mask_list[:-1], axis=0)
            self._mask = torch.from_numpy(mask).float()
            if self.is_cuda:
                self._mask = self._mask.cuda()
        return Variable(self._mask, volatile=self._volatile)

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros(self.batch_size).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i].data
            advantages.data[i] = gae

        return advantages

    def volatile(self, arg=True):
        self._volatile = arg

    def append(self, observations, actions, rewards, dones):
        self._observations_list.append(observations.astype(np.float32))
        self._actions_list.append(actions.astype(np.float32))
        self._rewards_list.append(rewards.astype(np.float32))
        # Masks are shifted by one timestep (first observation is always valid)
        if not self._mask_list:
            self._mask_list.append(np.ones_like(dones, dtype=np.float32))
        self._mask_list.append(1.0 - dones.astype(np.float32))

    def __len__(self):
        return len(self._actions_list)
