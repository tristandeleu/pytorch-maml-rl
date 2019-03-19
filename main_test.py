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
    rewards_total = torch.mean(torch.stack([aggregation(torch.sum(rewards[...,0], dim=0))
        for rewards in episodes_rewards], dim=0))
    rewards_dist = torch.mean(torch.stack([aggregation(torch.sum(rewards[...,1], dim=0))
        for rewards in episodes_rewards], dim=0))
    rewards_col = torch.mean(torch.stack([aggregation(torch.sum(rewards[...,2], dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards_total.item(), rewards_dist.item(), rewards_col.item()

def time_elapsed(elapsed_seconds):
    seconds = int(elapsed_seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds)]
    return  ', '.join('{} {}'.format(value, name) for name, value in periods if value)

def main(args):
    env_name = 'RVONavigationAll-v0' #['2DNavigation-v0', 'RVONavigation-v0',  'RVONavigationAll-v0']
    test_folder = './{0}'.format('test_nav')
    fast_batch_size = 40 # number of trajectories
    saved_policy_file = os.path.join('./TrainingResults/result3/saves/{0}'.format('maml-2DNavigation-dir'), 'policy-180.pt')

    sampler = BatchSampler(env_name, batch_size=fast_batch_size, num_workers=3)
    policy = NormalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        int(np.prod(sampler.envs.action_space.shape)),
        hidden_sizes = (100,) * 2)

    # Loading policy
    if os.path.isfile(saved_policy_file):
        policy_info = torch.load(saved_policy_file, map_location=lambda storage, loc: storage)
        policy.load_state_dict(policy_info)
        print('Loaded saved policy')
    else:
        sys.exit("The requested policy does not exist for loading")

    
    # Creating test folder
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Generate tasks
    # goal = [[-0.8, 0.9]]
    # task = [{'goal': goal}][0]
    tasks = sampler.sample_tasks(num_tasks=1)
    task = tasks[0]

    # Start validation
    print("Starting to test...Total step = ", args.grad_steps)
    start_time = time.time()
    # baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    baseline = LinearFeatureBaseline(int(np.prod((2,))))
    metalearner = MetaLearner(sampler, policy, baseline, gamma=0.9,fast_lr=0.01, tau=0.99, device='cpu')
    
    # test_episodes = metalearner.sample(tasks)
    # for train, valid in test_episodes:
    #     total_reward, dist_reward, col_reward = total_rewards(train.rewards)
    #     print(total_reward)
    #     total_reward, dist_reward, col_reward = total_rewards(valid.rewards)
    #     print(total_reward)
    
    test_episodes = metalearner.test(task, n_grad = args.grad_steps)
    print('-------------------')
    for n_grad, ep in test_episodes:
        total_reward, dist_reward, col_reward = total_rewards(ep.rewards)
        print(total_reward)
    #     with open(os.path.join(test_folder, 'test_episodes_grad'+str(n_grad)+'.pkl'), 'wb') as f: 
    #         pickle.dump([ep.observations.cpu().numpy(), ep], f)
     
    # with open(os.path.join(test_folder, 'task.pkl'), 'wb') as f: 
    #     pickle.dump(task, f)   
    print('Finished test. Time elapsed = {}'.format(time_elapsed(time.time() - start_time)))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--grad-steps', type=int, default=5,
        help='number of gradient updates steps')

    args = parser.parse_args()

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
