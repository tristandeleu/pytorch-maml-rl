import numpy as np

from gym.vector import SyncVectorEnv as SyncVectorEnv_
from gym.vector.utils import concatenate, create_empty_array


class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self,
                 env_fns,
                 observation_space=None,
                 action_space=None,
                 **kwargs):
        super(SyncVectorEnv, self).__init__(env_fns,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            **kwargs)
        for env in self.envs:
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('The environment provided is not a '
                                 'meta-learning environment. It does not have '
                                 'the method `reset_task` implemented.')

    @property
    def dones(self):
        return self._dones

    def reset_task(self, task):
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def step_wait(self):
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = len(self._actions)
        rewards = np.zeros((num_actions,), dtype=np.float_)
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                continue

            action = self._actions[j]
            observation, rewards[j], self._dones[i], info = env.step(action)
            batch_ids.append(i)

            if not self._dones[i]:
                observations_list.append(observation)
                infos.append(info)
            j += 1
        assert num_actions == j

        if observations_list:
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            concatenate(observations_list,
                        observations,
                        self.single_observation_space)
        else:
            observations = None

        return (observations, rewards, np.copy(self._dones),
                {'batch_ids': batch_ids, 'infos': infos})
