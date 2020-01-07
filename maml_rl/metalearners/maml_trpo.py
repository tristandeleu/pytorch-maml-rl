import torch
import asyncio

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.samplers import MultiTaskSampler
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 num_steps=1,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.num_steps = num_steps
        self.first_order = first_order

    def adapt(self, episodes, first_order=None):
        if first_order is None:
            first_order = self.first_order
        params = None
        for _ in range(self.num_steps):
            inner_loss = reinforce_loss(self.policy, episodes, params=params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self,
                             train_futures,
                             valid_futures,
                             old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        params = self.adapt(await train_futures,
                            first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            loss = -weighted_mean(ratio * valid_episodes.advantages,
                                  dim=0,
                                  weights=valid_episodes.mask)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(dim=2)
            kl = weighted_mean(kl_divergence(pi, old_pi),
                               dim=0,
                               weights=mask)

        return loss, kl, old_pi

    def step(self,
             train_episodes,
             valid_episodes,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_episodes)
        logs = {}

        # Compute the surrogate loss
        coroutine = asyncio.gather(*[self.surrogate_loss(train,
                                                         valid,
                                                         old_pi=None)
            for (train, valid) in zip(train_episodes, valid_episodes)])
        old_losses, old_kls, old_pis = zip(
            *self._event_loop.run_until_complete(coroutine))

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            coroutine = asyncio.gather(*[self.surrogate_loss(train,
                                                             valid,
                                                             old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(train_episodes, valid_episodes, old_pis)])

            losses, kls, _ = zip(*self._event_loop.run_until_complete(coroutine))
            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        return logs
