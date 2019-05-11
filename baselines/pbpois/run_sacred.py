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

# Sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

# Create experiment
ex = Experiment('POIS')
# Set a File Observer
if os.environ.get('SACRED_RUNS_DIRECTORY') is not None:
    print("Sacred logging at:", os.environ.get('SACRED_RUNS_DIRECTORY'))
    ex.observers.append(FileStorageObserver.create(os.environ.get('SACRED_RUNS_DIRECTORY')))
if os.environ.get('SACRED_SLACK_CONFIG') is not None:
    print("Sacred is using slack.")
    ex.observers.append(SlackObserver.from_config(os.environ.get('SACRED_SLACK_CONFIG')))

@ex.config
def custom_config():
    seed = 0
    env = 'rllab.cartpole'
    num_episodes = 100
    max_iters = 500
    horizon = 500
    iw_method = 'is'
    iw_norm = 'none'
    natural = False
    file_name = 'progress'
    logdir = 'logs'
    bound = 'max-d2'
    delta = 0.99
    aggregate = 'none'
    adaptive_batch = 0
    njobs = -1
    policy = 'nn'
    max_offline_iters = 10
    gamma = 1.0
    center = False
    # ENTROPY can be of 4 schemes:
    #    - 'none'
    #    - 'step:<height>:<duration>': step function which is <height> tall for <duration> iterations
    #    - 'lin:<max>:<min>': linearly decreasing function from <max> to <min> over all iterations, clipped to 0 for negatives
    #    - 'exp:<height>:<scale>': exponentially decreasing curve <height> tall, use <scale> to make it "spread" more
    # Create the filename
    if file_name == 'progress':
        file_name = '%s_iw=%s_bound=%s_delta=%s_gamma=%s_center=%s_entropy=%s_seed=%s_%s' % (env.upper(), iw_method, bound, delta, gamma, center, entropy, seed, time.time())
    else:
        file_name = file_name

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
                      use_bias=True,
                      seed=seed)

    sampler = ParallelSampler(make_env, make_policy, gamma, horizon, np.ravel, num_episodes, njobs, seed)

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
          feature_fun=np.ravel,
          iw_norm=iw_norm,
          bound = bound,
          max_offline_iters=max_offline_iters,
          delta=delta,
          center_return=False,
          line_search_type='parabola',
          adaptive_batch=adaptive_batch)

    sampler.close()

@ex.automain
def main(seed, env, num_episodes, horizon, iw_method, iw_norm, natural, file_name, logdir, bound, delta,
            njobs, policy, max_offline_iters, gamma, center, max_iters, aggregate, adaptive_batch, _run):

    logger.configure(dir=logdir, format_strs=['stdout', 'csv', 'tensorboard', 'sacred'], file_name=file_name, run=_run)
    train(env=env,
          max_iters=max_iters,
          num_episodes=num_episodes,
          horizon=horizon,
          iw_norm=iw_norm,
          bound=bound,
          delta=delta,
          gamma=gamma,
          seed=seed,
          policy=policy,
          max_offline_iters=max_offline_iters,
          njobs=njobs,
          aggregate=aggregate,
          adaptive_batch=adaptive_batch)
