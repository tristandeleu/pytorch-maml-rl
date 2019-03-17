import maml_rl.envs
import gym
import numpy as np
import torch
import json
import pickle
import time
import sys
import timeit
from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def time_elapsed(elapsed_seconds):
    seconds = int(elapsed_seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds)]
    return  ', '.join('{} {}'.format(value, name) for name, value in periods if value)

def main(args):
    print(args)
    continuous_actions = (args.env_name in ['2DNavigation-v0', 'RVONavigation-v0',  'RVONavigationAll-v0'])
    assert continuous_actions == True

    test_folder = './{0}'.format(args.output_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print("Creating new test folder", test_folder)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    policy = NormalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        int(np.prod(sampler.envs.action_space.shape)),
        hidden_sizes=(args.hidden_size,) * args.num_layers)

    # Loading policy
    saved_policy_file = os.path.join('./saved_policy/{0}'.format('maml-RVONavigation-dir'), 'policy-30.pt')
    if os.path.isfile(saved_policy_file):
        print('Loading saved policy')
        policy_info = torch.load(saved_policy_file, map_location=lambda storage, loc: storage)
        policy.load_state_dict(policy_info)
    else:
        sys.exit("The requested policy does not exist for loading")

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    # Start validation
    print("Starting to test...")
    start_time = time.time()
    # goals = [[-0.3, 0.3]]
    # tasks = [{'goal': goal} for goal in goals]
    tasks = sampler.sample_tasks(num_tasks=1)
    task = tasks[0]

    # test_episodes = metalearner.sample_test(task, first_order=args.first_order)

    test_episodes = metalearner.test(task, n_grad = args.grad_steps, first_order=args.first_order)
    with open(os.path.join(test_folder, 'task.pkl'), 'wb') as f: 
        pickle.dump(task, f)

    for n_grad, ep in test_episodes:
        print(n_grad)
        with open(os.path.join(test_folder, 'test_episodes_grad'+str(n_grad)+'.pkl'), 'wb') as f: 
            pickle.dump([ep.observations.cpu().numpy(), ep], f)
        
    print('Finished test. Time elapsed = {}'.format(time_elapsed(time.time() - start_time)))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='RVONavigationAll-v0',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=15,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='test_nav',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=8,
        help='number of workers for trajectories sampling')
    parser.add_argument('--grad-steps', type=int, default=3,
        help='number of gradient updates steps')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
