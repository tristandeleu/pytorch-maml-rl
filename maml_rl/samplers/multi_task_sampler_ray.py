import torch
import torch.multiprocessing as mp
import asyncio
import threading
import time
import numpy as np
from datetime import datetime, timezone
from copy import deepcopy

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss

import ray

def _create_consumer(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
    while True:
        data = queue.get()
        if data is None:
            break
        index, step, episodes = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episodes)


class MultiTaskSamplerRay(Sampler):
    """Vectorized sampler to sample trajectories from multiple environements.

    Parameters
    ----------
    env_name : str
        Name of the environment. This environment should be an environment
        registered through `gym`. See `maml.envs`.

    env_kwargs : dict
        Additional keywork arguments to be added when creating the environment.

    batch_size : int
        Number of trajectories to sample from each task (ie. `fast_batch_size`).

    policy : `maml_rl.policies.Policy` instance
        The policy network for sampling. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    baseline : `maml_rl.baseline.LinearFeatureBaseline` instance
        The baseline. This baseline is an instance of `nn.Module`, with an
        additional `fit` method to fit the parameters of the model.

    env : `gym.Env` instance (optional)
        An instance of the environment given by `env_name`. This is used to
        sample tasks from. If not provided, an instance is created from `env_name`.

    seed : int (optional)
        Random seed for the different environments. Note that each task and each
        environement inside every process use different random seed derived from
        this value if provided.

    num_workers : int
        Number of processes to launch. Note that the number of processes does
        not have to be equal to the number of tasks in a batch (ie. `meta_batch_size`),
        and can scale with the amount of CPUs available instead.
    """
    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=1):
        super(MultiTaskSamplerRay, self).__init__(env_name,
                                               env_kwargs,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)

        self.num_workers = num_workers

        ray.init()

        self.sampler = SamplerWorker(0,
                            env_name,
                            env_kwargs,
                            batch_size,
                            self.env.observation_space,
                            self.env.action_space,
                            self.policy,
                            deepcopy(baseline),
                            num_workers,
                            self.seed)

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample(self,
        tasks,
        num_steps=1,
        fast_lr=0.5,
        gamma=0.95,
        gae_lambda=1.0,
        device='cpu'):
      
        episodes = self.sampler.sample(tasks,
            num_steps,
            fast_lr,
            gamma,
            gae_lambda,
            device)

        return episodes

    @property
    def train_consumer_thread(self):
        if self._train_consumer_thread is None:
            raise ValueError()
        return self._train_consumer_thread

    @property
    def valid_consumer_thread(self):
        if self._valid_consumer_thread is None:
            raise ValueError()
        return self._valid_consumer_thread

    def _start_consumer_threads(self, tasks, num_steps=1):
        # Start train episodes consumer thread
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks]
                                  for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.train_episodes_queue, train_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='train-consumer')
        self._train_consumer_thread.daemon = True
        self._train_consumer_thread.start()

        # Start valid episodes consumer thread
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.valid_episodes_queue, valid_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='valid-consumer')
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()

        return (train_episodes_futures, valid_episodes_futures)

    def _join_consumer_threads(self):
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None)
            self.train_consumer_thread.join()

        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()

        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return

        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self._join_consumer_threads()

        self.closed = True

class SamplerWorker(object):
    def __init__(self,
                 index,
                 env_name,
                 env_kwargs,
                 batch_size,
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 num_workers,
                 seed):
        # super(SamplerWorker, self).__init__()

        # env_fns = [make_env(env_name, env_kwargs=env_kwargs)
        #            for _ in range(batch_size)]
        # self.envs = SyncVectorEnv(env_fns,
        #                           observation_space=observation_space,
        #                           action_space=action_space)
        # self.envs.seed(None if (seed is None) else seed + index * batch_size)
        self.index = index
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.workers = [RolloutWorker.remote(
            index=index,
            batch_size=batch_size,
            env_name=env_name,
            env_kwargs=env_kwargs,
            baseline=deepcopy(baseline)
        )
        for index in range(num_workers)]

    def sample(self,
            tasks,
            num_steps=1,
            fast_lr=0.5,
            gamma=0.95,
            gae_lambda=1.0,
            device='cpu'):


            # TODO: num_step=2 이상에서 되게 하기
            params=None
            params_list=[None]*len(tasks)
            # train_episodes = []
            for step in range(num_steps):
                train_episodes = self.create_episodes(tasks,
                                                    params=params_list,
                                                    gamma=gamma,
                                                    gae_lambda=gae_lambda,
                                                    device=device)
                for i, train_episode in enumerate(train_episodes):
                    train_episode.log('_enqueueAt', datetime.now(timezone.utc))

                    loss = reinforce_loss(self.policy, train_episode, params=params_list[i])
    
                    params_list[i] = self.policy.update_params(loss,
                                                    params=params_list[i],
                                                    step_size=fast_lr,
                                                    first_order=True)
            valid_episodes = self.create_episodes(tasks,
                                                params=params_list,
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                device=device)
            # for valid_episode in valid_episodes:         
            #     valid_episode.log('_enqueueAt', datetime.now(timezone.utc))
            # TODO: num_step=2 이상에서 되게 하기
            return ([deepcopy(train_episodes)], deepcopy(valid_episodes))

    def create_episodes(self,
                tasks,
                params,
                gamma=0.95,
                gae_lambda=1.0,
                device="cpu"):
        # episodes = BatchEpisodes(batch_size=self.batch_size,
        #                          gamma=gamma,
        #                          device=device)
        # episodes.log('_createdAt', datetime.now(timezone.utc))
        # episodes.log('process_name', self.name)
        episodes_ops = []

        # observations_list = []#[[] for _ in range(self.batch_size)]
        # actions_list = []#[[] for _ in range(self.batch_size)]
        # rewards_list = []#[[] for _ in range(self.batch_size)]
        # batch_list = []#[[] for _ in range(self.batch_size)]

        t0 = time.time()
        for index, task in enumerate(tasks):
            episodes_ops.append(
                self.workers[index].create.remote(task=task,
                                                policy=self.policy,
                                                # observations_list=observations_list,
                                                # actions_list=actions_list,
                                                # rewards_list=rewards_list,
                                                # batch_list=batch_list,
                                                # episodes=episodes,
                                                params=params[index],
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                device=device)
            )
        episodes = ray.get(episodes_ops)

        return episodes
        # print('episodes', episodes)
        # observations_list = []
        # actions_list = []
        # rewards_list = []
        # batch_list = []
        # for item in items:
        #     observations_list += item[0]
        #     actions_list += item[1]
        #     rewards_list += item[2]
        #     batch_list += item[3]
        
        # episodes.append(observations_list, actions_list, rewards_list, batch_list)
        # episodes.log('duration', time.time() - t0)
        # self.baseline.fit(episodes)
        # episodes.compute_advantages(self.baseline,
        #                             gae_lambda=gae_lambda,
        #                             normalize=True)
        # return episodes

@ray.remote
class RolloutWorker(object):
    def __init__(self,
                index,
                # policy,
                batch_size,
                env_name,
                env_kwargs,
                baseline,
                ) -> None:
        self.index = index
        # self.policy = policy
        self.baseline = baseline
        self.batch_size = batch_size
        env_kwargs['index'] = index
        self.env = make_env(env_name, env_kwargs=env_kwargs)()

    def sample_trajectories(self, task, policy, params):
        self.env.reset_task(task)
        for i in range(self.batch_size):
            done = False
            observations = self.env.reset()
            with torch.no_grad():
                while not done:
                    # self.env.render()
                    observations_tensor = torch.from_numpy(observations).type(torch.float32)
                    pi = policy(observations_tensor, params)
                    actions_tensor = pi.sample()
                    actions = actions_tensor.cpu().numpy()

                    new_observations, rewards, done, _ = self.env.step(actions)

                    yield (observations, actions, rewards, i)
                    observations = new_observations

    def create(self,
                task,
                  # episodes,
                policy,
                params=None,
                gamma=0.95,
                gae_lambda=1.0,
                device="cpu"):
        episodes = BatchEpisodes(batch_size=self.batch_size,
                    gamma=gamma,
                    device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))
        # episodes.log('process_name', self.name)

        observations_list=[]
        actions_list=[]
        rewards_list=[]
        batch_ids_list=[]
        t0 = time.time()
        for observations, actions, rewards, batch_ids in self.sample_trajectories(task, policy, params):
            observations_list.append(observations)
            actions_list.append(actions)
            rewards_list.append(rewards)
            batch_ids_list.append(batch_ids)
        episodes.append(observations_list, actions_list, rewards_list, batch_ids_list)
        episodes.log('duration', time.time() - t0)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return episodes