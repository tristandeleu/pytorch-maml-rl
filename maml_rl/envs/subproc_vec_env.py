import numpy as np
import multiprocessing as mp
import gym

class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.done = False

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = (self.empty_step()
                    if self.done else self.env.step(data))
                self.done |= done
                self.remote.send((observation, reward, done, info))
            elif command == 'reset':
                observation = self.env.reset()
                self.done = False
                self.remote.send(observation)
            elif command == 'reset_task':
                self.env.unwrapped.reset_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                 self.env.action_space))
            else:
                raise NotImplementedError()

class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        self.workers = [EnvWorker(remote, env_fn) for (remote, env_fn)
            in zip(self.work_remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
