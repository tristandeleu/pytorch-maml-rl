import gym

def make_env(env_name, seed=None):
    def _make_env():
        env = gym.make(env_name)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env

class Sampler(object):
    def __init__(self, env_name, batch_size, policy, baseline):
        self.env_name = env_name
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.env = gym.make(env_name)
        self.env.close()
        self.closed = False

    def sample(self, *args, **kwargs):
        raise NotImplementedError()
