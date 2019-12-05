import gym

def make_env(env_name, seed=None):
    def _make_env():
        env = gym.make(env_name)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env

class Sampler(object):
    def __init__(self, env_name, batch_size, policy, baseline, env=None):
        self.env_name = env_name
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        if env is None:
            env = gym.make(env_name)
        self.env = env
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
