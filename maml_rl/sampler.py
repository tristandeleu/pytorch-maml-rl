import gym
import torch
from torch.autograd import Variable

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

class BatchSampler(object):
    def __init__(self, env_name, batch_size):
        self.env_name = env_name
        self.batch_size = batch_size
        
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(batch_size)])

    def sample(self, policy, params=None, gamma=0.95, is_cuda=False):
        episodes = BatchEpisodes(gamma=gamma, is_cuda=is_cuda)
        observations = self.envs.reset()
        dones = [False]
        while not all(dones):
            observations_var = Variable(torch.from_numpy(observations), volatile=True)
            actions_var = policy(observations_var, params=params).sample()
            actions = actions_var.data.cpu().numpy()
            new_observations, rewards, dones, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, dones)
            observations = new_observations
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.batch_size)]
        reset = self.envs.reset_task(tasks)
        return all(reset)
