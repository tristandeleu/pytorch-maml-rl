import numpy as np
import gym
import os

from gym import error
from gym.utils import seeding
from gym.spaces import Box

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install "
        "mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500
BIG = 1e6

class MujocoEnv(gym.Env):
    """MuJoCo [1] environment base class. This base class differs from gym's
    environment, and is taken from rllab [2]. The code is adapted from garage
    [3], a fork of rllab, to work with `mujoco_py`.

    [1] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    [2] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    [3] https://github.com/rlworkgroup/garage
    """
    def __init__(self, model_path, frame_skip=1):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            folder = os.path.dirname(__file__)
            fullpath = os.path.join(folder, "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self._observation_space = None
        self._action_space = None

        self.init_qpos = self.sim.data.qpos
        self.init_qvel = self.sim.data.qvel
        self.init_qacc = self.sim.data.qacc
        self.init_ctrl = self.sim.data.ctrl

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def observation_space(self):
        if self._observation_space is None:
            shape = self.get_current_obs().shape
            self._observation_space = Box(low=-1.0 * BIG, high=BIG,
                shape=shape, dtype=np.float32)
        return self._observation_space

    @property
    def action_space(self):
        if self._action_space is None:
            bounds = self.model.actuator_ctrlrange.copy()
            low, high = bounds[:, 0], bounds[:, 1]
            self._action_space = Box(low=low, high=high, dtype=np.float32)
        return self._action_space

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def reset_model(self):
        self.sim.data.qpos[:] = self.init_qpos + np.random.normal(
            size=self.model.nq) * 0.01
        self.sim.data.qvel[:] = self.init_qvel + np.random.normal(
            size=self.model.nv) * 0.1
        self.sim.data.qacc[:] = self.init_qacc
        self.sim.data.ctrl[:] = self.init_ctrl

    def forward_dynamics(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            self.sim.step()
        self.sim.forward()

    def get_current_obs(self):
        raise NotImplementedError()

    def reset(self, init_state=None):
        self.sim.reset()
        self.reset_model()
        self.sim.forward()
        return self.get_current_obs()

    def get_body_xmat(self, body_name):
        return self.data.get_body_xmat(body_name).reshape((3, 3))

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        return self.data.get_body_xvelp(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
