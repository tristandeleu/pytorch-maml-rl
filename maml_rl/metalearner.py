import torch
import torch.nn.functional as F
from torch.autograd import Variable
# TODO: Replace by torch.distributions in Pytorch 0.4
from maml_rl.distributions import Categorical, Normal
from maml_rl.distributions.kl import kl_divergence

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
        log_probs = pi.log_prob(episodes.actions)
        # Sum the log probabilities in case of continuous actions
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        # TODO: Use episodes.mask for mean
        loss = -torch.mean(log_probs * advantages)

        return loss

    def adapt(self, episodes):
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr)

        return params

    def sample(self, meta_batch_size=20):
        tasks = self.sampler.sample_tasks(num_tasks=meta_batch_size)
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, is_cuda=self.is_cuda)

            params = self.adapt(train_episodes)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, is_cuda=self.is_cuda)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def loss(self, episodes):
        # TODO: To be deprecated
        losses = []
        for train_episodes, valid_episodes in episodes:
            params = self.adapt(train_episodes)
            valid_loss = self.inner_loss(valid_episodes, params=params)
            losses.append(valid_loss)
        total_rewards = torch.cat([torch.sum(valid_episodes.rewards)
            for (_, valid_episodes) in episodes])

        return torch.mean(torch.cat(losses, dim=0)), torch.mean(total_rewards)

    def hessian_vector_product(self, episodes, damping=1e-2):
        def _product(vector):
            _, kl = self.surrogate_loss(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = torch.cat([grad.contiguous().view(-1)
                for grad in grad2s])

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls = [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                if isinstance(pi, Categorical):
                    old_pi = Categorical(logits=pi.logits.detach())
                elif isinstance(pi, Normal):
                    old_pi = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
                else:
                    raise NotImplementedError('Only `Categorical` and `Normal` '
                        'policies are valid policies.')

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=1.0)
            ratio = torch.exp(pi.log_prob(valid_episodes.actions)
                - old_pi.log_prob(valid_episodes.actions))

            # TODO: Use valid_episodes.mask for mean
            loss = -torch.mean(ratio * advantages)
            losses.append(loss)

            # TODO: Use valid_episodes.mask for mean
            kl = torch.mean(kl_divergence(pi, old_pi))
            kls.append(kl)

        return (torch.mean(torch.cat(losses, dim=0)),
                torch.mean(torch.cat(kls, dim=0)))

    def cuda(self, **kwargs):
        self.policy.cuda(**kwargs)
        self.baseline.cuda(**kwargs)
        self.is_cuda = True
