import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

class HalfCheetahVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal=0):
        self._task = 0
        self._goal_vel = goal
        #temporary action_space
        self.action_space = spaces.Box(low=0.5, high=1, shape=(1,), dtype=np.float32)

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.*abs(forward_vel - self._goal_vel)

        lb = self.action_space.low
        ub = self.action_space.high
        action = np.clip(action, lb, ub)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        reward = forward_reward - ctrl_cost
        done = False
        observation = self._get_obs()
        return (observation, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            task=self._task))

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso"),
        ]).astype(np.float32).reshape(-1)

    def sample_tasks(self, num_tasks):
        velocities = np.random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

    def reset(self):

        qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel  + np.random.normal(size=self.init_qvel.shape) * 0.1

        #qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        #qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

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
