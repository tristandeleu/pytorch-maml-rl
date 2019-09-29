import numpy as np
import gym


class SyncVectorEnv(gym.Env):
    def __init__(self, env_fns, observation_space=None, action_space=None):
        super(SyncVectorEnv, self).__init__()
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        self.observation_space = observation_space
        self.action_space = action_space

        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset_task(self, task):
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def reset(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)

        return np.stack(observations, axis=0)

    def step(self, actions):
        observations, infos = [], []
        batch_ids, j = [], 0
        rewards = np.zeros((len(actions),), dtype=np.float_)
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                continue

            observation, rewards[j], self._dones[i], info = env.step(actions[j])
            batch_ids.append(i)

            if not self._dones[i]:
                observations.append(observation)
                infos.append(info)
            j += 1
        assert len(actions) == j

        if observations:
            observations = np.stack(observations, axis=0)

        return (observations, rewards, np.copy(self._dones),
                {'batch_ids': batch_ids, 'infos': infos})

    def close(self):
        for env in self.envs:
            env.close()
