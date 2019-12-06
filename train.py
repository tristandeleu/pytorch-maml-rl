import maml_rl.envs
import gym
import torch
import json
from tqdm import trange

from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    env = gym.make(args.env_name)
    env.close()

    # Policy
    hidden_sizes = (args.hidden_size,) * args.num_layers
    policy = get_policy_for_env(env,
                                hidden_sizes=hidden_sizes,
                                nonlinearity=args.nonlinearity)
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))
    # Sampler
    sampler = MultiTaskSampler(args.env_name,
                               batch_size=args.fast_batch_size,
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(sampler,
                           policy,
                           fast_lr=args.fast_lr,
                           num_steps=args.num_steps,
                           first_order=args.first_order,
                           device=args.device)

    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = metalearner.sample_async(tasks,
                                                                  gamma=args.gamma,
                                                                  tau=args.tau)
        metalearner.step(train_episodes,
                         valid_episodes,
                         max_kl=args.max_kl,
                         cg_iters=args.cg_iters,
                         cg_damping=args.cg_damping,
                         ls_max_steps=args.ls_max_steps,
                         ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--config', type=str, required=False, default=None,
        help='path to the configuration file (optional)')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--env-name', type=str,
        help='name of the environment')
    general.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma (default: 0.95)')
    general.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE (default: 1.0)')
    general.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    policy = parser.add_argument_group('Policy network')
    policy.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer (default: 100)')
    policy.add_argument('--nonlinearity', type=str,
        choices=['relu', 'tanh'], default='relu',
        help='nonlinearity function (default: relu)')
    policy.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers (default: 2)')

    # Task-specific
    task_specific = parser.add_argument_group('Task-specific')
    task_specific.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task (default: 20)')
    task_specific.add_argument('--num-steps', type=int, default=1,
        help='number of gradient steps for adaptation (default: 1)')
    task_specific.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the gradient update of MAML (default: 0.5)')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--num-batches', type=int, default=200,
        help='number of batches (default: 200)')
    optimization.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')
    optimization.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO (default: 1e-2)')
    optimization.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient (default: 10)')
    optimization.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient (default: 1e-5)')
    optimization.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search (default: 15)')
    optimization.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='annealing ratio of the step size for line search (default: 0.8)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder (default: maml)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    misc.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

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
