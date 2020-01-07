import torch
import torch.multiprocessing as mp
import asyncio
import threading
from copy import deepcopy

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss


def _create_consumer(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
    while True:
        data = queue.get()
        if data is None:
            break
        index, episodes = data
        if not futures[index].cancelled():
            loop.call_soon_threadsafe(futures[index].set_result, episodes)


class MultiTaskSampler(Sampler):
    def __init__(self,
                 env_name,
                 batch_size,
                 policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=1):
        super(MultiTaskSampler, self).__init__(env_name,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)
        
        self.policy.share_memory()
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()

        self.workers = [SamplerWorker(index,
                                      env_name,
                                      batch_size,
                                      self.env.observation_space,
                                      self.env.action_space,
                                      self.policy,
                                      deepcopy(baseline),
                                      self.seed,
                                      self.task_queue,
                                      self.train_episodes_queue,
                                      self.valid_episodes_queue,
                                      policy_lock)
            for index in range(num_workers)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample_async(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Calling `sample_async` while waiting '
                               'for a pending call to `sample_async` '
                               'to complete. Please call `sample_wait` '
                               'before calling `sample_async` again.')

        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))

        # Start train episodes consumer thread
        train_episodes_futures = [self._event_loop.create_future() for _ in tasks]
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

        self._waiting_sample = True
        return (train_episodes_futures, valid_episodes_futures)

    def sample_wait(self, episodes_futures):
        if not self._waiting_sample:
            raise RuntimeError('Calling `sample_wait` without any '
                               'prior call to `sample_async`.')

        async def _wait(train_futures, valid_futures):
            # Gather the train and valid episodes
            train_episodes = await asyncio.gather(*train_futures)
            valid_episodes = await asyncio.gather(*valid_futures)
            self._join_consumer_threads()
            return (train_episodes, valid_episodes)
        samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
        self._waiting_sample = False
        return samples

    def sample(self, tasks, **kwargs):
        futures = self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)

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


class SamplerWorker(mp.Process):
    def __init__(self,
                 index,
                 env_name,
                 batch_size,
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 seed,
                 task_queue,
                 train_queue,
                 valid_queue,
                 policy_lock):
        super(SamplerWorker, self).__init__()

        env_fns = [make_env(env_name) for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=observation_space,
                                  action_space=action_space)
        self.envs.seed(None if (seed is None) else seed + index * batch_size)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        # Sample the training trajectories with the initial policy
        train_episodes = BatchEpisodes(batch_size=self.batch_size,
                                       gamma=gamma,
                                       device=device)
        for item in self.sample_trajectories():
            train_episodes.append(*item)
        self.baseline.fit(train_episodes)
        train_episodes.compute_advantages(self.baseline,
                                          gae_lambda=gae_lambda,
                                          normalize=True)
        self.train_queue.put((index, train_episodes))

        # Adapt the policy to the task, based on the REINFORCE loss computed on
        # the training trajectories. The gradient update in the fast adaptation
        # uses `first_order=True` no matter if the second order version of MAML
        # is used since this is only used for sampling trajectories, and not
        # for optimization.
        with self.policy_lock:
            params = None
            for _ in range(num_steps):
                loss = reinforce_loss(self.policy, train_episodes, params=params)
                params = self.policy.update_params(loss,
                                                   params=params,
                                                   step_size=fast_lr,
                                                   first_order=True)

        # Sample the validation trajectories with the adapted policy
        valid_episodes = BatchEpisodes(batch_size=self.batch_size,
                                       gamma=gamma,
                                       device=device)
        for item in self.sample_trajectories(params=params):
            valid_episodes.append(*item)
        valid_episodes.compute_advantages(self.baseline,
                                          gae_lambda=gae_lambda,
                                          normalize=True)
        self.valid_queue.put((index, valid_episodes))

    def sample_trajectories(self, params=None):
        observations = self.envs.reset()
        dones = self.envs._dones
        with torch.no_grad():
            while not dones.all():
                observations_tensor = torch.from_numpy(observations)
                pi = self.policy(observations_tensor, params=params)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()

                new_observations, rewards, new_dones, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids)
                observations, dones = new_observations, new_dones

    def run(self):
        while True:
            data = self.task_queue.get()

            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break

            index, task, kwargs = data
            self.envs.reset_task(task)
            self.sample(index, **kwargs)
            self.task_queue.task_done()
