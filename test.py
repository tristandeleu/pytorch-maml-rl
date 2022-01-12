import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from torch.utils.tensorboard import SummaryWriter

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    config['num-steps'] = args.num_steps

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'], device=args.device)
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env), device=args.device).to(args.device)

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

    tblog_folder = os.path.join(args.output, 'tb' + str(args.seed))
    writer = SummaryWriter(tblog_folder)

    for steps in trange(config['num-steps']):
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=steps,
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)
        rewards = []
        for id in range(len(tasks)):
            reward = valid_episodes[id].rewards.sum()
            writer.add_scalar(f'Eval_Reward/Task_{id}', reward, steps)
            rewards.append(reward.cpu().numpy())
        writer.add_scalar('Eval_Reward/Mean', np.mean(rewards), steps)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
                        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
                        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
                            help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
                            help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
                      help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                           '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full support for cuda '
                           'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--num-steps', type=int, default=10, required=True, help='number of gradient steps to take')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                              and args.use_cuda) else 'cpu')

    main(args)
