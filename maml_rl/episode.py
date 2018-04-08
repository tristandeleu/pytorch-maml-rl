import numpy as np
import torch
from torch.autograd import Variable

class BatchEpisodes(object):
    def __init__(self, gamma=0.95, is_cuda=False):
        self.gamma = gamma
        self.is_cuda = is_cuda
        
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
    def observations(self):
        if self._observations is None:
            observations = np.stack(self._observations_list, axis=0)
            observations_tensor = torch.from_numpy(observations).float()
            if self.is_cuda:
                observations_tensor = observations_tensor.cuda()
            self._observations = Variable(observations_tensor)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            actions = np.stack(self._actions_list, axis=0)
            actions_tensor = torch.from_numpy(actions).float()
            if self.is_cuda:
                actions_tensor = actions_tensor.cuda()
            self._actions = Variable(actions_tensor)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.stack(self._rewards_list, axis=0)
            rewards_tensor = torch.from_numpy(rewards).float()
            if self.is_cuda:
                rewards_tensor = rewards_tensor.cuda()
            self._rewards = Variable(rewards_tensor)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            batch_size = self._rewards_list[0].shape[0]
            return_ = np.zeros(batch_size, dtype=np.float32)
            returns = np.zeros((len(self), batch_size), dtype=np.float32)
            for i in range(len(self) - 1, -1, -1):
                return_ = (self.gamma * return_
                    + self._rewards_list[i] * self._mask_list[i])
                returns[i] = return_
            returns_tensor = torch.from_numpy(returns).float()
            if self.is_cuda:
                returns_tensor = returns_tensor.cuda()
            self._returns = Variable(returns_tensor)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.stack(self._mask_list[:-1], axis=0)
            mask_tensor = torch.from_numpy(mask).float()
            if self.is_cuda:
                mask_tensor = mask_tensor.cuda()
            self._mask = Variable(mask_tensor)
        return self._mask

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
