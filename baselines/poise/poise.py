import os
import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import colorize
from baselines.common import zipsame
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg
import matplotlib.pyplot as plt




@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize(
        'done in %.3f seconds' % (time.time() - tstart), color='magenta'))


def traj_segment_generator(pi, env, n_episodes, horizon, stochastic=True):
    """
    Generates a dataset of trajectories
        pi: policy
        env: environment
        n_episodes: batch size
        horizon: max episode length
        stochastic: activates policy stochasticity
    """
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()
    cur_ep_ret = 0
    cur_ep_len = 0

    # Initialize history arrays
    ep_rets = []
    ep_lens = []
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    # Collect trajectories
    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        if i == n_episodes:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred*(1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "mask": mask}
            _, vpred = pi.act(stochastic, ob)

            # Reset episode
            ep_rets = []
            ep_lens = []
            mask = np.ones(horizon * n_episodes, 'float32')
            i = 0
            t = 0

        # Update history arrays
        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac
        # Transition
        ob, rew, new, _ = env.step(ac)
        rews[t] = rew
        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1

        # Next episode
        if new or j == horizon:
            new = True
            env.done = True
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            next_t = (i+1) * horizon
            mask[t+1:next_t] = 0.
            acs[t+1:next_t] = acs[t]
            obs[t+1:next_t] = obs[t]
            t = next_t - 1
            i += 1
            j = 0

        # Next step
        t += 1


def add_disc_rew(seg, gamma):
    """
    Adds discounted rewards and returns to the dataset
        seg: the dataset
        gamma: discount factor
    """
    new = np.append(seg['new'], 1)
    rew = seg['rew']
    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)
    seg['ep_disc_rets'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')
    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

        if new[t + 1]:
            discounter = 0
            ep_disc_ret[i] = ret
            i += 1
            ret = 0.
        else:
            discounter += 1


def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))


def line_search_parabola(den_mise, theta_init, alpha, natural_grad,
                         set_parameter, evaluate_bound, dtheta,
                         iters_so_far, delta_bound_tol=1e-4,
                         max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound(den_mise)
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_grad
        set_parameter(theta)

        bound = evaluate_bound(den_mise)

        if np.isnan(bound):
            print('Got NaN bound value: rolling back!')
            return theta_old, epsilon_old, 0, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1


def plot_bound_profile(
        theta_grid, bound, mise, bonus, miw_1, point_x, point_y, delta, iter):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(theta_grid, bound, label='bound', color='red', linewidth='2')
    ax.plot(theta_grid, mise, label='mise', color='blue', linewidth='0.5')
    ax.plot(theta_grid, bonus, label='bonus', color='green', linewidth='0.5')
    ax.plot(theta_grid, miw_1, label='bonus', color='green', linestyle='--')
    ax.plot(point_x, point_y, 'o', color='orange')
    ax.legend(loc='upper right')
    # Save plot to given dir
    dir = './bound_profile/unbounded_policy/miw_1_delta_{}/'.format(delta)
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def best_of_grid(policy, grid_size, theta_init,
                 old_thetas_list,
                 iters_so_far, mask_iters,
                 set_parameter, set_parameter_old, evaluate_behav,
                 evaluate_bound, evaluate_roba, delta):

    # Compute MISE's denominator
    den_mise = np.zeros(mask_iters.shape).astype(np.float32)
    for i in range(len(old_thetas_list)):
        set_parameter_old(old_thetas_list[i])
        behav = evaluate_behav()
        den_mise = den_mise + np.exp(behav)

    # Compute log of MISE's denominator
    eps = 1e-24  # to avoid inf weights and nan bound
    den_mise = (den_mise + eps) / iters_so_far
    den_mise_log = np.log(den_mise) * mask_iters

    # Calculate the grid of parameters to evaluate
    if hasattr(policy, 'min_mean'):
        theta_grid = np.linspace(policy.min_mean, policy.max_mean, grid_size)
    else:
        theta_grid = np.linspace(-1, 1, grid_size)
    # Evaluate the set of parameters and retain the best one
    bound = []
    mise = []
    bonus = []
    miw_1 = []
    miw_2 = []
    bound_best = 0
    theta_best = theta_init

    for theta in theta_grid:
        set_parameter([theta])
        bound_theta = evaluate_bound(den_mise_log)
        bound.append(bound_theta)
        mise_theta, bonus_theta, miw_1_theta, miw_2_theta = \
            evaluate_roba(den_mise_log)
        mise.append(mise_theta)
        bonus.append(bonus_theta)
        miw_1.append(miw_1_theta)
        miw_2.append(miw_2_theta)
        if bound_theta > bound_best:
            bound_best = bound_theta
            theta_best = [theta]

    # Checkpoint
    set_parameter([theta_grid[0]])
    mise_theta, bonus_theta, miw_1_theta, miw_2_theta = \
        evaluate_roba(den_mise_log)
    print('miw_1, miw_2 e d2 for p(mu=1,std=0.1):',
          theta_grid[0],
          miw_1_theta,
          miw_2_theta,
          miw_2_theta/miw_1_theta)

    # Plot the profile of the bound and its components
    plot_bound_profile(
        theta_grid, bound, mise, bonus, miw_1, theta_best,
        bound_best, delta, iters_so_far)

    # Calculate improvement
    set_parameter(theta_init)
    improvement = bound_best - evaluate_bound(den_mise_log)

    return theta_best, improvement, den_mise_log


def optimize_offline(evaluate_roba, theta_init, dtheta, old_thetas_list,
                     iters_so_far, mask_iters,
                     set_parameter, set_parameter_old, evaluate_behav,
                     evaluate_bound, evaluate_grad,
                     line_search, evaluate_natural_grad=None,
                     grad_tol=1e-4, bound_tol=1e-10, max_offline_ite=10):

    # Compute MISE's denominator
    den_mise = np.zeros(mask_iters.shape).astype(np.float32)
    for i in range(len(old_thetas_list)):
        set_parameter_old(old_thetas_list[i])
        behav = evaluate_behav()
        den_mise = den_mise + np.exp(behav)

    # Compute log of MISE's denominator
    eps = 1e-24  # to avoid inf weights and nan bound
    den_mise = (den_mise + eps) / iters_so_far
    den_mise_log = np.log(den_mise) * mask_iters

    # Optimization loop
    theta = theta_old = theta_init
    improvement = improvement_old = 0.
    set_parameter(theta)
    bound = evaluate_bound(den_mise_log)
    bound_old = bound
    print('Initial bound after last sampling:', bound)

    # Print infos about optimization loop
    fmtstr = '%6i %10.3g %16i %18.3g %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s  %18s %16s %18s %18s %18s'
    print(titlestr % ('iter', 'step size', 'line searches', 'grad norm',
                      'delta theta', 'delta bound ite', 'delta bound tot'))
    num_line_search = 0
    if max_offline_ite > 0:
        for i in range(max_offline_ite):

            grad = evaluate_grad(den_mise_log)
            # Sanity check for the grad
            if np.any(np.isnan(grad)):
                print('Got NaN grad! Stopping!')
                set_parameter(theta_old)
                return theta_old, improvement, den_mise_log, bound_old

            grad_norm = np.sqrt(np.dot(grad, grad))
            # Check that the grad norm is not too small
            if grad_norm < grad_tol:
                print('stopping - grad norm < grad_tol')
                # print('theta', theta)
                # print('theta_old', theta_old)
                # print('theta_init', theta_init)
                return theta_old, improvement_old, den_mise_log, bound_old

            # alpha = dtheta
            alpha = dtheta / grad_norm ** 2
            # Save old values
            theta_old = theta
            improvement_old = improvement
            # Make an optimization step
            if line_search is not None:
                theta, epsilon, delta_bound, num_line_search = \
                    line_search(den_mise_log, theta, alpha, grad,
                                set_parameter, evaluate_bound,
                                iters_so_far, bound_tol)
                set_parameter(theta)
                delta_theta = np.array(theta) - np.array(theta_old)
            else:
                delta_theta = alpha*grad
                theta = theta + delta_theta
                set_parameter(theta)
                # Sanity check for the bound
                bound = evaluate_bound(den_mise_log)
                if np.isnan(bound):
                    print('Got NaN bound! Stopping!')
                    set_parameter(theta_old)
                    return theta_old, improvement_old, den_mise_log, bound_old
                delta_bound = bound - bound_old

            improvement = improvement + delta_bound
            bound_old = bound

            print(fmtstr % (i+1, alpha, num_line_search, grad_norm,
                            delta_theta[0], delta_bound, improvement))

    print('Max number of offline iterations reached.')
    return theta, improvement, den_mise_log, bound


def render(env, pi, horizon):
    """
    Shows a test episode on the screen
        env: environment
        pi: policy
        horizon: episode length
    """
    t = 0
    ob = env.reset()
    env.render()

    done = False
    while not done and t < horizon:
        ac, _ = pi.act(True, ob)
        ob, _, done, _ = env.step(ac)
        time.sleep(0.1)
        env.render()
        t += 1


def learn(make_env, make_policy, *,
          max_iters,
          horizon,
          dtheta,
          delta,
          gamma,
          multiple_init=None,
          sampler=None,
          iw_norm='none',
          bound='max-ess',
          max_offline_iters=10,
          save_weights=False,
          render_after=None,
          callback=None,
          line_search=None,
          grid_optimization=None):
    """
    Learns a policy from scratch
        make_env: environment maker
        make_policy: policy maker
        horizon: max episode length
        delta: probability of failure
        gamma: discount factor
        max_iters: total number of learning iteration
    """

    # Print options
    np.set_printoptions(precision=3)

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space
    max_samples = horizon * max_iters

    # Build the policy
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_params = sum(shapes)

    # Get all oldpi's learnable parameters
    all_var_list_old = oldpi.get_trainable_variables()
    var_list_old = \
        [v for v in all_var_list_old if v.name.split('/')[1].startswith('pol')]

    # My Placeholders
    old_thetas_ = tf.placeholder(shape=[None, n_params],
                                 dtype=tf.float32, name='old_thetas')
    den_mise_log_ = tf.placeholder(dtype=tf.float32, name='den_mise')
    ob_ = ob = U.get_placeholder_cached(name='ob')  # shape=[None, ac_shape]
    ac_ = pi.pdtype.sample_placeholder([max_samples], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='mask')
    mask_iters_ = tf.placeholder(dtype=tf.float32, shape=(max_iters),
                                 name='mask_iters')
    rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='rew')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples),
                               name='disc_rew')
    grad_ = tf.placeholder(dtype=tf.float32,
                               shape=(n_params, 1), name='grad')
    iter_number_ = tf.placeholder(dtype=tf.float32, name='iter_number')

    iter_number_int = tf.cast(iter_number_, dtype=tf.int32)
    losses_with_name = []

    # Policy densities
    target_log_pdf = pi.pd.logp(ac_)
    behavioral_log_pdf = oldpi.pd.logp(ac_)

    # Split operations
    disc_rew_split = tf.stack(tf.split(disc_rew_ * mask_, max_iters))
    rew_split = tf.stack(tf.split(rew_ * mask_, max_iters))
    target_log_pdf_split = tf.stack(
        tf.split(target_log_pdf * mask_, max_iters))
    behavioral_log_pdf_split = tf.stack(
        tf.split(behavioral_log_pdf * mask_, max_iters))
    mask_split = tf.stack(tf.split(mask_, max_iters))

    # Multiple importance weights computation
    # nb: the sum behaviorals' logs is centered for avoiding numerical problems
    target_sum_log = tf.reduce_sum(target_log_pdf_split, axis=1)
    target_sum_log_mean = tf.reduce_sum(target_sum_log)/iter_number_
    behavioral_sum_log = tf.reduce_sum(behavioral_log_pdf_split, axis=1)
    # behavioral_sum_log_mean = tf.reduce_sum(behavioral_sum_log)/iter_number_
    # behavioral_sum_log_centered = \
    #     (behavioral_sum_log - behavioral_sum_log_mean) * mask_iters_
    # log_ratio_split = target_sum_log - behavioral_sum_log_mean - den_mise_log_
    log_ratio_split = target_sum_log - den_mise_log_
    log_ratio_split = log_ratio_split * mask_iters_
    miw = tf.exp(log_ratio_split) * mask_iters_

    den_mise_log_mean = tf.reduce_sum(den_mise_log_)/iter_number_
    den_mise_log_last = den_mise_log_[iter_number_int-1]
    losses_with_name.extend([(target_sum_log_mean, 'target_sum_log_mean'),
                             (den_mise_log_mean, 'DenMISEMeanLog'),
                             (den_mise_log_[0], 'DenMISELogFirst'),
                             (den_mise_log_last, 'DenMISELogLast'),
                             (miw[0], 'IWFirstEpisode'),
                             (miw[iter_number_int-1], 'IWLastEpisode'),
                             (tf.reduce_sum(miw)/iter_number_, 'IWMean'),
                             (tf.reduce_max(miw), 'IWMax'),
                             (tf.reduce_min(miw), 'IWMin')])

    # Return
    ep_return = tf.reduce_sum(disc_rew_split * mask_split, axis=1)
    return_mean = tf.reduce_sum(ep_return)/iter_number_
    return_last = ep_return[iter_number_int - 1]
    return_max = tf.reduce_max(ep_return)
    return_min = tf.reduce_min(ep_return)
    return_abs_max = tf.reduce_max(tf.abs(ep_return))
    return_step_max = tf.reduce_max(tf.abs(rew_split))  # Max step reward

    losses_with_name.extend([(return_mean, 'ReturnMean'),
                             (return_max, 'ReturnMax'),
                             (return_min, 'ReturnMin'),
                             (return_last, 'ReturnLastEpisode'),
                             (return_abs_max, 'ReturnAbsMax'),
                             (return_step_max, 'ReturnStepMax')])

    # MISE
    mise = tf.reduce_sum(miw * ep_return * mask_iters_)/iter_number_
    losses_with_name.append((mise, 'MISE'))
    # test MISE = ISE when sampling always from the same distribution
    # lr_is = target_log_pdf - behavioral_log_pdf
    # lrs_is = tf.stack(tf.split(lr_is * mask_, max_iters))
    # iw = tf.exp(tf.reduce_sum(lrs_is, axis=1))
    # mis = tf.reduce_sum(iw * ep_return)/iter_number_
    # losses_with_name.append((mis, 'ISE'))

    # Bounds
    eps = 1e-18  # for eps<1e-18 miw_2=0 if weights are zero
    miw_ess = (tf.exp(log_ratio_split) + eps) * mask_iters_
    # miw_1 = tf.reduce_sum(miw_ess)
    miw_1 = tf.linalg.norm(miw_ess, ord=1)
    miw_2 = tf.linalg.norm(miw_ess, ord=2)
    if bound == 'J':
        bound = mise
    elif bound == 'max-ess':
        sqrt_ess = miw_1 / miw_2
        # def log10(x):
        #     numerator = tf.log(x)
        #     denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        #     return numerator / denominator
        #
        # delta_t = delta*(1 - 1/log10(iter_number_))
        const = return_abs_max * tf.sqrt(1 / delta - 1)
        exploration_bonus = const / (sqrt_ess)
        bound = mise + exploration_bonus
        losses_with_name.extend([(sqrt_ess, 'SqrtESSClassic'),
                                 (exploration_bonus, 'ExplorationBonus'),
                                 (miw_1, 'IWNorm1'),
                                 (miw_2, 'IWNorm2')])

    losses_with_name.append((bound, 'Bound'))

    # Infos
    assert_ops = tf.group(*tf.get_collection('asserts'))
    print_ops = tf.group(*tf.get_collection('prints'))
    losses, loss_names = map(list, zip(*losses_with_name))

    # TF functions
    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)
    set_parameter_old = U.SetFromFlat(var_list_old)

    compute_behav = U.function(
        [ob_, ac_, mask_, iter_number_, mask_iters_],
        behavioral_sum_log,
        updates=None, givens=None)
    compute_miw = U.function(
        [ob_, ac_, mask_, den_mise_log_],
        miw, updates=None, givens=None)
    compute_bound = U.function(
        [ob_, ac_, rew_, disc_rew_, mask_, iter_number_,
         mask_iters_, den_mise_log_], [bound, assert_ops, print_ops])
    compute_grad = U.function(
        [ob_, ac_, rew_, disc_rew_, mask_, iter_number_, mask_iters_,
         den_mise_log_], [U.flatgrad(bound, var_list), assert_ops, print_ops])
    compute_losses = U.function(
        [ob_, ac_, rew_, disc_rew_, mask_, iter_number_,
         mask_iters_, den_mise_log_], losses)
    compute_roba = U.function(
        [ob_, ac_, rew_, disc_rew_, mask_, iter_number_,
         mask_iters_, den_mise_log_], [mise, exploration_bonus, miw_1, miw_2])

    # Set sampler (default: sequential)
    if sampler is None:
        # Rb:collect only ONE trajectory
        seg_gen = traj_segment_generator(pi, env, 1,
                                         horizon, stochastic=True)
        sampler = type("SequentialSampler", (object,),
                       {"collect": lambda self, _: seg_gen.__next__()})()

    # Set line search
    if line_search is not None:
        s = str(line_search)
        if s == "parabola" or s == "parabolic":
            print('Step size optimization through parabolic line search.')
            line_search = line_search_parabola
        else:
            raise ValueError("{} l.s. not implemented".format(line_search))

    # Tf initialization
    U.initialize()

    # Learning loop
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    mask_iters = np.zeros(max_iters)
    tstart = time.time()
    # Store behaviorals' params and their trajectories
    old_thetas_list = []
    all_seg = {}
    all_seg['ob'] = np.zeros((max_samples, ob_space.shape[0]))
    all_seg['ac'] = np.zeros(shape=ac_.get_shape().as_list())

    for i in ["rew", "disc_rew", "mask"]:
        all_seg[i] = np.zeros(max_samples)

    theta = theta_start = get_parameter()
    print('Optimization of the %s bound' % (bound))
    while True:
        iters_so_far += 1

        # Render one episode
        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        # Custom callback
        if callback:
            callback(locals(), globals())

        # Exit loop in the end
        if iters_so_far >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iters_so_far)

        # Generate one trajectory
        with timed('sampling'):
            seg = sampler.collect(theta)

        # Store the list of arrays representing behaviorals' parameters
        old_thetas_list.append(theta)

        # Retrieve data
        add_disc_rew(seg, gamma)
        lens, u_rets = seg['ep_lens'], seg['ep_rets']
        assert len(lens) == 1
        episodes_so_far += 1
        timesteps_so_far += lens[0]

        # Set arguments for U.functions
        args = ()
        mask_iters[:iters_so_far] = 1

        for key in all_seg.keys():
            start = (iters_so_far-1)*horizon
            all_seg[key][start:start + horizon] = seg[key]

        args = all_seg['ob'], all_seg['ac'],  all_seg['rew'], \
            all_seg['disc_rew'], all_seg['mask'], iters_so_far, mask_iters

        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("URet", u_rets[0])
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Save policy parameters to disk
        if save_weights:
            logger.record_tabular('Weights', str(get_parameter()))
            import pickle
            file = open('checkpoint.pkl', 'wb')
            pickle.dump(theta, file)

        def evaluate_behav():
            args_behav = all_seg['ob'], all_seg['ac'], \
                all_seg['mask'], iters_so_far, mask_iters
            return compute_behav(*args_behav)

        def evaluate_bound(den_mise_log):
            args_bound = args + (den_mise_log,)
            return compute_bound(*args_bound)[0]

        def evaluate_grad(den_mise_log):
            args_grad = args + (den_mise_log,)
            return compute_grad(*args_grad)[0]

        def evaluate_roba(den_mise_log):
            args_roba = args + (den_mise_log,)
            return compute_roba(*args_roba)

        with timed("Optimization"):
            if multiple_init:
                bound = improvement = 0
                check = False
                for i in range(multiple_init):
                    theta_init = [np.arctanh(np.random.uniform(
                        pi.min_mean, pi.max_mean))]
                    theta_i, improvement_i, den_mise_log_i, bound_i = \
                        optimize_offline(evaluate_roba, theta_init, dtheta,
                                         old_thetas_list,
                                         iters_so_far,
                                         mask_iters, set_parameter,
                                         set_parameter_old,
                                         evaluate_behav, evaluate_bound,
                                         evaluate_grad, line_search,
                                         max_offline_ite=max_offline_iters)
                    if bound_i > bound:
                        check = True
                        theta = theta_i
                        improvement = improvement_i
                        den_mise_log = den_mise_log_i
                if not check:
                    den_mise_log = den_mise_log_i
            elif grid_optimization:
                theta, improvement, den_mise_log = \
                    best_of_grid(pi, grid_optimization, theta,
                                 old_thetas_list,
                                 iters_so_far, mask_iters,
                                 set_parameter, set_parameter_old,
                                 evaluate_behav,
                                 evaluate_bound,
                                 evaluate_roba, delta)
            else:
                theta, improvement, den_mise_log, bound = \
                    optimize_offline(evaluate_roba, theta, dtheta,
                                     old_thetas_list,
                                     iters_so_far,
                                     mask_iters, set_parameter,
                                     set_parameter_old,
                                     evaluate_behav, evaluate_bound,
                                     evaluate_grad, line_search,
                                     max_offline_ite=max_offline_iters)
            set_parameter(theta)

        args += (den_mise_log,)
        with timed('summaries after'):
            if env.spec.id == 'LQG1D-v0':
                mu1 = pi.eval_mean([[1]])[0][0]
                mu01 = pi.eval_mean([[0.1]])[0][0]
                sigma = pi.eval_std()[0][0]
                logger.record_tabular("LQGmu1", mu1)
                logger.record_tabular("LQGmu01", mu01)
                logger.record_tabular("LQGsigma", sigma)
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()
