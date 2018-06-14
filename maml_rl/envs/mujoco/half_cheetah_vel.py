import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

class HalfCheetahVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **task):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.get_body_com("torso").flat[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso").flat[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data
        elif mode == 'human':
            self._get_viewer().render()
