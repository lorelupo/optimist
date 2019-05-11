#!/usr/bin/env python3
'''
    This script runs rllab or gym environments. To run RLLAB, use the format
    rllab.<env_name> as env name, otherwise gym will be used.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict

# Framework imports
import gym
import tensorflow as tf

# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy
from baselines.pois2 import pois2

def get_env_type(env_id):
    #First load all envs
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        # TODO: solve this with regexes
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)
    # Get env type
    env_type = None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    return env_type

def train(env, policy, seed, njobs=1, **alg_args):

    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\w+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        # Define env maker
        def make_env(seed=0):
            def _thunk():
                env_rllab = Rllab2GymWrapper(env_rllab_class())
                env_rllab.seed(seed)
                return env_rllab
            return _thunk
        parallel_env = SubprocVecEnv([make_env(i + seed) for i in range(njobs)])
        # Used later
        env_type = 'rllab'
    else:
        # Normal gym, get if Atari or not.
        env_type = get_env_type(env)
        assert env_type is not None, "Env not recognized."
        # Define the correct env maker
        if env_type == 'atari':
            # Atari, custom env creation
            def make_env(seed=0):
                def _thunk():
                    _env = make_atari(env)
                    _env.seed(seed)
                    return wrap_deepmind(_env)
                return _thunk
            parallel_env = VecFrameStack(SubprocVecEnv([make_env(i + seed) for i in range(njobs)]), 4)
        else:
            # Not atari, standard env creation
            def make_env(seed=0):
                def _thunk():
                    _env = gym.make(env)
                    _env.seed(seed)
                    return _env
                return _thunk
            parallel_env = SubprocVecEnv([make_env(i + seed) for i in range(njobs)])

    # Create the policy
    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    if policy == 'linear' or policy == 'nn':
        def make_policy(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
                             hidden_W_init=tf.contrib.layers.xavier_initializer(),
                             output_W_init=tf.contrib.layers.xavier_initializer())
    elif policy == 'cnn':
        def make_policy(name, ob_space, ac_space):
            return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=tf.contrib.layers.xavier_initializer(),
                         output_W_init=tf.contrib.layers.xavier_initializer())
    else:
        raise Exception('Unrecognized policy type.')

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    pois2.learn(parallel_env, make_policy, **alg_args)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--center', type=bool, default=False)
    parser.add_argument('--clipping', type=bool, default=False)
    parser.add_argument('--entropy', type=str, default='none')
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)
    train(env=args.env,
          policy=args.policy,
          n_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed,
          njobs=args.njobs,
          max_iters=args.max_iters,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          use_natural_gradient=args.natural,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          max_offline_iters=args.max_offline_iters,
          center_return=args.center,
          clipping=args.clipping,
          entropy=args.entropy)

if __name__ == '__main__':
    main()
