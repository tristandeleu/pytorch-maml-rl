import gym
import torch
import torch.nn.functional as F

class MetaLearner(object):
    def __init__(self, sampler, policy, baseline,
                 gamma=0.95, fast_lr=0.5, is_cuda=False):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.is_cuda = is_cuda
    
    def inner_loss(self, episodes, params=None):
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=1.0)

        pi = self.policy(episodes.observations, params=params)
        # TODO: Check log_prob for continuous actions (eg. NormalMLPPolicy)
        log_probs = pi.log_prob(episodes.actions)

        loss = -torch.mean(log_probs * advantages)

        return loss

    def loss(self, meta_batch_size=20):
        tasks = self.sampler.sample_tasks(num_tasks=meta_batch_size)
        losses = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, is_cuda=self.is_cuda)
            self.baseline.fit(train_episodes)
            train_loss = self.inner_loss(train_episodes)

            params = self.policy.update_params(train_loss,
                step_size=self.fast_lr)

            valid_episodes = self.sampler.sample(self.policy,
                params=params, gamma=self.gamma, is_cuda=self.is_cuda)
            valid_loss = self.inner_loss(valid_episodes, params=params)
            losses.append(valid_loss)

        return torch.mean(torch.cat(losses, dim=0)), torch.sum(valid_episodes.rewards)

    def cuda(self, **kwargs):
        self.policy.cuda(**kwargs)
        self.baseline.cuda(**kwargs)
        self.is_cuda = True
