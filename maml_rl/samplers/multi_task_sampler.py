import torch
import torch.multiprocessing as mp

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss


class MultiTaskSampler(Sampler):
    def __init__(self, env_name, batch_size, policy, baseline, num_workers=1):
        super(MultiTaskSampler, self).__init__(env_name, batch_size, policy, baseline)
        
        self.policy.share_memory()
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()

        self.workers = [SamplerWorker(env_name, self.env.observation_space,
            self.env.action_space, batch_size, self.policy, self.baseline,
            self.task_queue, self.train_episodes_queue, self.valid_episodes_queue)
            for _ in range(num_workers)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample(self, tasks, **kwargs):
        for i, task in enumerate(tasks):
            self.task_queue.put((i, task, kwargs))

    def close(self):
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self.train_episodes_queue.put(None)
        self.valid_episodes_queue.put(None)

        self.closed = True


class SamplerWorker(mp.Process):
    def __init__(self, env_name, observation_space, action_space, batch_size, policy, baseline, task_queue, train_queue, valid_queue):
        super(SamplerWorker, self).__init__()

        env_fns = [make_env(env_name) for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=observation_space,
                                  action_space=action_space)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, tau=1.0, device='cpu'):
        # Sample the training trajectories with the initial policy
        train_episodes = BatchEpisodes(batch_size=self.batch_size,
                                       gamma=gamma,
                                       device=device)
        for item in self.sample_trajectories():
            train_episodes.append(*item)
        self.baseline.fit(train_episodes)
        train_episodes.compute_advantages(self.baseline, tau=tau, normalize=True)
        self.train_queue.put((index, train_episodes))

        # Adapt the policy to the task, based on the REINFORCE loss computed on
        # the training trajectories. The gradient update in the fast adaptation
        # uses `first_order=True` no matter if the second order version of MAML
        # is used since this is only used for sampling trajectories, and not
        # for optimization.
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
        valid_episodes.compute_advantages(self.baseline, tau=tau, normalize=True)
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
