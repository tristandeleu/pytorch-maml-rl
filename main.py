import maml_rl.envs
import gym
import numpy as np
import torch

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

def main(args):
    continuous_actions = (args.env_name in ['AntVelEnv-v0', 'AntDirEnv-v0',
        'HalfCheetahVelEnv-v0', 'HalfCheetahDirEnv-v0', '2DNavigation-v0'])

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            np.prod(sampler.envs.observation_space.shape),
            np.prod(sampler.envs.action_space.shape)
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            np.prod(sampler.envs.observation_space.shape),
            sampler.envs.action_space.n
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        np.prod(sampler.envs.observation_space.shape))

    metalearner = MetaLearner(sampler, policy, baseline,
        gamma=args.gamma, fast_lr=args.fast_lr)
    if args.cuda:
        metalearner.cuda()

    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks)
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='MAML')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-2,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=10,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--cuda', action='store_const', const=True,
        help='use CUDA (if available)')

    args = parser.parse_args()
    args.cuda &= torch.cuda.is_available()

    main(args)
