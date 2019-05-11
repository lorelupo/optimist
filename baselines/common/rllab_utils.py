'''
    Utils for utilize RLLAB
'''

import gym

def rllab_env_from_name(env):
    if env == 'swimmer':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        return SwimmerEnv
    elif env == 'ant':
        from rllab.envs.mujoco.ant_env import AntEnv
        return AntEnv
    elif env == 'half_cheetah':
        from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
        return HalfCheetahEnv
    elif env == 'hopper':
        from rllab.envs.mujoco.hopper_env import HopperEnv
        return HopperEnv
    elif env == 'simple_humanoid':
        from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
        return SimpleHumanoidEnv
    elif env == 'full_humanoid':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        return HumanoidEnv
    elif env == 'walker':
        from rllab.envs.mujoco.walker2d_env import Walker2DEnv
        return Walker2DEnv
    elif env == 'cartpole':
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        return CartpoleEnv
    elif env == 'mountain-car':
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return MountainCarEnv
    elif env == 'inverted_pendulum':
        from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv as InvertedPendulumEnv
        return InvertedPendulumEnv
    elif env == 'acrobot':
        from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv as AcrobotEnv
        return AcrobotEnv
    elif env == 'inverted_double_pendulum':
        from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
        return InvertedPendulumEnv
    else:
        raise Exception('Unrecognized rllab environment.')

def convert_rllab_space(space):

    import rllab
    import gym.spaces

    if isinstance(space, rllab.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high)
    elif isinstance(space, rllab.spaces.Discrete):
        return gym.spaces.Discrete(n=space._n)
    elif isinstance(space, rllab.spaces.Tuple):
        return gym.spaces.Tuple([convert_rllab_space(x) for x in space._components])
    else:
        raise NotImplementedError

class Rllab2GymWrapper(gym.Env):

    def __init__(self, rllab_env, env_name=None):
        import rllab
        from rllab.envs.normalized_env import normalize
        self.rllab_env = normalize(rllab_env)
        self.observation_space = convert_rllab_space(rllab_env.observation_space)
        self.action_space = convert_rllab_space(rllab_env.action_space)
        self.id = env_name
        self.seed()
        self.reset()

    def step(self, action):
        res = self.rllab_env.step(action)
        return tuple(res)

    def reset(self):
        new_state = self.rllab_env.reset()
        return new_state

    def seed(self, seed=0):
        pass
