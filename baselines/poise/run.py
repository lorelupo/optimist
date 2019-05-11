#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs rllab or gym environments. To run RLLAB, use the format
    rllab.<env_name> as env name, otherwise gym will be used.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
import numpy as np
# Framework imports
import gym
import tensorflow as tf
# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.bounded_mlp_policy import MlpPolicyBounded
from baselines.policy.cnn_policy import CnnPolicy
from baselines.poise import poise


def get_env_type(env_id):
    # First load all envs
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)
    # Get env type
    env_type = None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    return env_type


def train(env, policy, horizon, seed, bounded_policy,
          trainable_std, gain_init,
          njobs=1, **alg_args):

    # Prepare environment maker
    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\w+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)

        # Define env maker
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
            # Atari, custom env creation
            def make_env():
                _env = make_atari(env)
                return wrap_deepmind(_env)
        else:
            # Not atari, standard env creation
            def make_env():
                env_rllab = gym.make(env)
                return env_rllab

    # Prepare policy maker
    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    if policy == 'linear' or policy == 'nn':
        if bounded_policy:
            print('Bounded policy with trainable_std=', trainable_std)
            def make_policy(name, ob_space, ac_space):
                return MlpPolicyBounded(
                    name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=hid_size, num_hid_layers=num_hid_layers,
                    gaussian_fixed_var=True, trainable_std=trainable_std,
                    use_bias=False, use_critic=False,
                    gain_init=gain_init,
                    max_mean=1,
                    min_mean=-1,
                    max_std=None,
                    min_std=0.1,
                    std_init=0.11)  # added 0.01 to avoid warning
        else:
            print('Unbounded policy')
            def make_policy(name, ob_space, ac_space):
                return MlpPolicy(
                    name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=hid_size, num_hid_layers=num_hid_layers,
                    gaussian_fixed_var=True, trainable_std=trainable_std,
                    use_bias=False, use_critic=False,
                    hidden_W_init=tf.constant_initializer(-0.1))
    else:
        raise Exception('Unrecognized policy type.')

    # Prepare (sequential) sampler to generate ONE trajectory at a time
    sampler = None

    # Initialize
    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()
    set_global_seeds(seed)
    gym.logger.setLevel(logging.WARN)

    # Learn
    poise.learn(make_env, make_policy, horizon=horizon,
                sampler=sampler, **alg_args)

    # Close sampler in the end
    # sampler.close()


def single_run(args, dtheta=None, grid_opt=None, delta=None, seed=None):

    # Import custom envs (must be imported here or wont work with multiple_run)
    import baselines.envs.lqg1d  # registered at import as gym env

    # Manage call from multiple runs
    if delta:
        args.delta = delta
    if seed:
        args.seed = seed
    if dtheta:
        args.dtheta = dtheta
    if grid_opt:
        args.grid_optimization = grid_opt

    # Log file name
    t = time.localtime(time.time())
    tt = int(str(time.time())[-5:])
    time_str = '%s-%s-%s_%s%s%s_%s' % (
        t.tm_hour, t.tm_min, t.tm_sec, t.tm_mday, t.tm_mon, t.tm_year, tt)
    args_str = '%s_delta=%s_seed=%s_dtheta=%s' % (
        args.env.upper(), args.delta, args.seed, args.dtheta)

    if args.file_name == 'progress':
        file_name = args_str + '_' + time_str
    else:
        file_name = args.file_name + '_' + args_str + '_' + time_str

    # Configure logger
    logger.configure(dir=args.logdir,
                     format_strs=['stdout', 'csv', 'tensorboard'],
                     file_name=file_name)

    # Learn
    train(env=args.env,
          policy=args.policy,
          horizon=args.horizon,
          seed=args.seed,
          bounded_policy=args.bounded_policy,
          trainable_std=args.trainable_std,
          gain_init=args.gain_init,  # LQG only
          multiple_init=args.multiple_init,
          njobs=args.njobs,
          bound=args.bound,
          delta=args.delta,
          dtheta=args.dtheta,
          gamma=args.gamma,
          max_offline_iters=args.max_offline_iters,
          max_iters=args.max_iters,
          render_after=args.render_after,
          line_search=args.line_search,
          grid_optimization=args.grid_optimization)


def multiple_runs(args):

    # Import tools for parallelizing runs
    from joblib import Parallel, delayed

    # Define range() for floats
    # dtheta = []
    delta = []
    seed = []
    #  grid_opt = []
    # for i in [n/10 for n in range(1, 8)]:
    for i in [100]:
        for j in [0.05, 0.1, 0.2, 0.99]:
            for k in range(1):
                # dtheta.append(i)
                #  grid_opt.append(i)
                delta.append(j)
                seed.append(k)

    # Parallelize single runs
    n_jobs = len(delta)
    Parallel(n_jobs=n_jobs)(delayed(single_run)(
        args,
        # dtheta=dtheta[i],
        #  grid_opt=grid_opt[i],
        delta=delta[i],
        seed=seed[i]
        ) for i in range(n_jobs))


def main(args):

    import argparse

    # Easily add a boolean argument to the parser with default value
    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    # Command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='LQG1D-v0')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--bound', type=str, default='max-ess')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--gain_init', type=float, default=-0.1)  # LQG only
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--dtheta', type=float, default=1)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--render_after', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--line_search', type=str, default=None)  # 'parabola'
    parser.add_argument('--multiple_init', type=int, default=None)
    parser.add_argument('--grid_optimization', type=int, default=None)
    add_bool_arg(parser, 'bounded_policy', default=True)
    add_bool_arg(parser, 'trainable_std', default=True)
    add_bool_arg(parser, 'experiment', default=False)
    args = parser.parse_args(args)

    if args.experiment:
        multiple_runs(args)
    else:
        single_run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
