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
import numpy as np

# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# Self imports: algorithm
from baselines.policy.neuron_hyperpolicy import MultiPeMlpPolicy
from baselines.policy.weight_hyperpolicy import PeMlpPolicy
from baselines.pbpois import pbpois, nbpois
from baselines.pbpois.parallel_sampler import ParallelSampler

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

def train(env, max_iters, num_episodes, horizon, iw_norm, bound, delta, gamma, seed, policy, max_offline_iters, aggregate, adaptive_batch, njobs=1):

    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\w+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        # Define env maker
        def make_env():
            env_rllab = env_rllab_class()
            _env = Rllab2GymWrapper(env_rllab)
            return _env
        # Used later
        env_type = 'rllab'
    else:
        # Normal gym, get if Atari or not.
        env_type = get_env_type(env)
        assert env_type is not None, "Env not recognized."
        # Define the correct env maker
        if env_type == 'atari':
            # Atari is not tested here
            raise Exception('Not tested on atari.')
        else:
            # Not atari, standard env creation
            def make_env():
                env_rllab = gym.make(env)
                return env_rllab

    # Create the policy
    if policy == 'linear':
        hid_layers = []
    elif policy == 'nn':
        hid_layers = [100, 50, 25]
    elif policy == 'cnn':
        raise Exception('CNN policy not tested.')

    if aggregate=='none':
        learner = pbpois
        PolicyClass = PeMlpPolicy
    elif aggregate=='neuron':
        learner = nbpois
        PolicyClass = MultiPeMlpPolicy
    else:
        print("Unknown aggregation method, defaulting to none")
        learner = pbpois
        PolicyClass = PeMlpPolicy

    make_policy = lambda name, observation_space, action_space: PolicyClass(name,
                      observation_space,
                      action_space,
                      hid_layers,
                      use_bias=False,
                      seed=seed)

    # sampler = ParallelSampler(make_env, make_policy, gamma, horizon, np.ravel, num_episodes, njobs, seed)
    sampler = None

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)


    learner.learn(
          make_env,
          make_policy,
          sampler,
          gamma=gamma,
          n_episodes=num_episodes,
          horizon=horizon,
          max_iters=max_iters,
          verbose=1,
          feature_fun=None,
          iw_norm=iw_norm,
          bound = bound,
          max_offline_iters=max_offline_iters,
          delta=delta,
          center_return=False,
          line_search_type='parabola',
          adaptive_batch=adaptive_batch)

    sampler.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--aggregate', type=str, default='none')
    parser.add_argument('--adaptive_batch', type=int, default=0)
    parser.add_argument('--delta', type=float, default=0.3)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_offline_iters', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)
    train(env=args.env,
          max_iters=args.max_iters,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_norm=args.iw_norm,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          seed=args.seed,
          policy=args.policy,
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs,
          aggregate=args.aggregate,
          adaptive_batch=args.adaptive_batch)

if __name__ == '__main__':
    main()
