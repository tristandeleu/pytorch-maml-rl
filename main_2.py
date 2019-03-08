import maml_rl.envs
import gym
from gym.utils import seeding
import numpy as np
import torch
import json
import pickle

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.policies.policy import Policy, weight_init

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):

    log_traj_folder = './logs/{0}'.format(args.output_traj_folder)
    if not os.path.exists(log_traj_folder):
        os.makedirs(log_traj_folder)

    env_name = '2DNavigation-v0'
    output_folder = 'maml'
    save_folder = './saves/{0}'.format(output_folder)
    PATH = os.path.join(save_folder, 'policy-100.pt'.format(1))
    params = torch.load(PATH)
    sampler = BatchSampler(env_name, batch_size=1)

    # observation = [[50., -50.], [-30., 0.]]
    # observation_tensor = torch.tensor(observation)
    # print(observation_tensor)
    # actions_tensor = Policy(observation_tensor).sample()

    policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),hidden_sizes=(args.hidden_size,) * args.num_layers)
    policy.load_state_dict(params)

    # actions_tensor = policy.forward_mu(observation_tensor, params)
    # actions_tensor = policy.forward(observation_tensor, params).sample()
    # action = actions_tensor.cpu().numpy()
    # print(action)
    
    # np_random, seed = seeding.np_random(None)
    # goals = np_random.uniform(-0.5, 0.5, size=(3, 2))
    goals = [[-0.3, 0.3]]
    tasks = [{'goal': goal} for goal in goals]

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    metalearner = MetaLearner(sampler, policy, baseline)
    # valid_episodes = sampler.sample(policy, params=params)

    
    # tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    episodes = metalearner.sample(tasks, first_order=args.first_order)
    metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
        cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
        ls_backtrack_ratio=args.ls_backtrack_ratio)
    episodes = metalearner.sample(tasks, first_order=args.first_order)
    print(ep.observations.numpy() for _, ep in episodes)

    batch = 0
    log_traj_folder = './logs/{0}'.format(args.output_traj_folder)
    with open(os.path.join(log_traj_folder, 'train_episodes_observ_'+str(batch)+'.pkl'), 'wb') as f: 
        pickle.dump([ep.observations.numpy() for ep, _ in episodes], f)
    with open(os.path.join(log_traj_folder, 'valid_episodes_observ_'+str(batch)+'.pkl'), 'wb') as f: 
        pickle.dump([ep.observations.numpy() for _, ep in episodes], f)
    # save tasks
    # sample task list of 2: [{'goal': array([0.0209588 , 0.15981938])}, {'goal': array([0.45034602, 0.17282322])}]
    with open(os.path.join(log_traj_folder, 'tasks_'+str(batch)+'.pkl'), 'wb') as f: 
        pickle.dump(tasks, f)
        
    # with open(os.path.join(log_traj_folder, 'latest_tasks.pkl'), 'wb') as f: 
    #     pickle.dump(tasks, f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='2DNavigation-v0',
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
    parser.add_argument('--fast-batch-size', type=int, default=30,
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
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml-2DNavigation-dir',
        help='name of the output folder')
    parser.add_argument('--output-traj-folder', type=str, default='2DNavigation-traj-dir',
        help='name of the output trajectory folder')
    parser.add_argument('--save_every', type=int, default=5,     
                        help='save frequency')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # print("--num-workers: mp.cpu_count() - 1 = {}".format(mp.cpu_count() - 1))
    # on my laptop: mp.cpu_count() - 1 = 3

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)