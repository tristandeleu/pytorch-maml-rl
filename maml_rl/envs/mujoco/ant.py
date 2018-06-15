import numpy as np
from gym.envs.mujoco import AntEnv

class AntVelEnv(AntEnv):
    def __init__(self, **task):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        self._action_scaling = None
        super(AntVelEnv, self).__init__()

    @property
    def action_scaling(self):
        if not hasattr(self, 'action_space'):
            return 1.0
        if self._action_scaling is None:
            lb, ub = self.action_space.low, self.action_space.high
            self._action_scaling = 0.5 * (ub - lb)
        return self._action_scaling

    def step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * np.abs(forward_vel - self._goal_vel) + 1.0
        survive_reward = 0.05

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / self.action_scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost, reward_survive=survive_reward,
            task=self._task)
        return (observation, reward, done, infos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.sim.data.get_body_xmat("torso").flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def sample_tasks(self, num_tasks):
        velocities = np.random.uniform(0.0, 3.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

class AntDirEnv(AntEnv):
    def __init__(self, **task):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        self._action_scaling = None
        super(AntDirEnv, self).__init__()

    @property
    def action_scaling(self):
        if not hasattr(self, 'action_space'):
            return 1.0
        if self._action_scaling is None:
            lb, ub = self.action_space.low, self.action_space.high
            self._action_scaling = 0.5 * (ub - lb)
        return self._action_scaling

    def step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        survive_reward = 0.05

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / self.action_scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost, reward_survive=survive_reward,
            task=self._task)
        return (observation, reward, done, infos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.sim.data.get_body_xmat("torso").flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def sample_tasks(self, num_tasks):
        directions = 2 * np.random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task['direction']
