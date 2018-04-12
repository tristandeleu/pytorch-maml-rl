import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from copy import deepcopy
# TODO: Replace by torch.distributions in Pytorch 0.4
from maml_rl.distributions import Categorical, Normal
from maml_rl.distributions.kl import kl_divergence

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = Variable(b.data)
    r = Variable(b.data)
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = Variable(f_Ax(p).data)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.data[0] < residual_tol:
            break

    return Variable(x.data)

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

    def hessian_vector_product(self, episodes, damping=1e-2):
        def _product(vector):
            _, kl, _ = self.surrogate_loss(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)
            pis.append(detach_distribution(pi))

            if old_pi is None:
                old_pi = detach_distribution(pi)

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=1.0)
            ratio = torch.exp(pi.log_prob(valid_episodes.actions)
                - old_pi.log_prob(valid_episodes.actions))

            # TODO: Use valid_episodes.mask for mean
            loss = torch.mean(ratio * advantages)
            losses.append(loss)

            # TODO: Use valid_episodes.mask for mean
            kl = torch.mean(kl_divergence(pi, old_pi))
            kls.append(kl)

        return (torch.mean(torch.cat(losses, dim=0)),
                torch.mean(torch.cat(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10):
        self.policy.zero_grad()
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Set volatile=True for the validation episodes
        for _, valid_episodes in episodes:
            valid_episodes.volatile()

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params + step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.data[0] >= 0.0) and (kl.data[0] < max_kl * 1.5):
                break
            step_size *= 0.5
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def cuda(self, **kwargs):
        self.policy.cuda(**kwargs)
        self.baseline.cuda(**kwargs)
        self.is_cuda = True
