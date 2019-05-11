import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import zipsame, colorize
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize('done in %.3f seconds'%(time.time() - tstart), color='magenta'))

def traj_segment_generator(pi, env, n_episodes, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if i == n_episodes:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "mask" : mask}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            mask = np.ones(horizon * n_episodes, 'float32')
            i = 0
            t = 0

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
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
        t += 1

def add_disc_rew(seg, gamma):
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
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

def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter, evaluate_bound, delta_bound_tol=1e-4, max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound()
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)

        bound = evaluate_bound()

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
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

def line_search_binary(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-4, max_line_search_ite=30):
    low = 0.
    high = None
    bound_init = evaluate_loss()
    delta_bound_old = 0.
    theta_opt = theta_init
    i_opt = 0
    delta_bound_opt = 0.
    epsilon_opt = 0.

    epsilon = 1.

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * natural_gradient * alpha
        set_parameter(theta)

        bound = evaluate_loss()
        delta_bound = bound - bound_init

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')

        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            theta_opt = theta
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        epsilon_old = epsilon

        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.

        if abs(epsilon_old - epsilon) < 1e-12:
            break

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt+1


def optimize_offline(theta_init, set_parameter, line_search, evaluate_loss,
                     evaluate_gradient, evaluate_natural_gradient=None,
                     gradient_tol=1e-4, bound_tol=1e-4, max_offline_ite=100):
    theta = theta_old = theta_init
    improvement = improvement_old = 0.
    set_parameter(theta)


    '''
    bound_init = evaluate_loss()
    import scipy.optimize as opt

    def func(x):
        set_parameter(x)
        return -evaluate_loss()

    def grad(x):
        set_parameter(x)
        return -evaluate_gradient().astype(np.float64)

    theta, bound, d = opt.fmin_l_bfgs_b(func=func,
                                        fprime=grad,
                                x0=theta_init.astype(np.float64),
                                maxiter=100,
                                    )
    print(bound_init, bound)

    print(d)

    set_parameter(theta)
    improvement = bound_init + bound
    return theta, improvement

    '''

    fmtstr = '%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s %10s %18s %18s %18s %18s'
    print(titlestr % ('iter', 'epsilon', 'step size', 'num line search',
                      'gradient norm', 'delta bound ite', 'delta bound tot'))

    for i in range(max_offline_ite):
        bound = evaluate_loss()
        gradient = evaluate_gradient()

        if np.any(np.isnan(gradient)):
            warnings.warn('Got NaN gradient! Stopping!')
            set_parameter(theta_old)
            return theta_old, improvement

        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            set_parameter(theta_old)
            return theta_old, improvement_old

        if evaluate_natural_gradient is not None:
            natural_gradient = evaluate_natural_gradient(gradient)
        else:
            natural_gradient = gradient

        if np.dot(gradient, natural_gradient) < 0:
            warnings.warn('NatGradient dot Gradient < 0! Using vanilla gradient')
            natural_gradient = gradient

        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))

        if gradient_norm < gradient_tol:
            print('stopping - gradient norm < gradient_tol')
            return theta, improvement

        alpha = 1. / gradient_norm ** 2

        theta_old = theta
        improvement_old = improvement
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, evaluate_loss)
        set_parameter(theta)

        improvement += delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, gradient_norm, delta_bound, improvement))

        if delta_bound < bound_tol:
            print('stopping - delta bound < bound_tol')
            return theta, improvement

    return theta, improvement

def render(env, pi, horizon):

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
          n_episodes,
          horizon,
          delta,
          gamma,
          max_iters,
          sampler=None,
          use_natural_gradient=False, #can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='J',
          line_search_type='parabola',
          save_weights=False,
          improvement_tol=0.,
          center_return=False,
          render_after=None,
          max_offline_iters=100,
          callback=None,
          clipping=False,
          entropy='none',
          positive_return=False,
          reward_clustering='none'):

    np.set_printoptions(precision=3)
    max_samples = horizon * n_episodes

    if line_search_type == 'binary':
        line_search = line_search_binary
    elif line_search_type == 'parabola':
        line_search = line_search_parabola
    else:
        raise ValueError()

    # Building the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Building the policy
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]

    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    # Placeholders
    ob_ = ob = U.get_placeholder_cached(name='ob')
    ac_ = pi.pdtype.sample_placeholder([max_samples], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='mask')
    rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='rew')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='disc_rew')
    gradient_ = tf.placeholder(dtype=tf.float32, shape=(n_parameters, 1), name='gradient')
    iter_number_ = tf.placeholder(dtype=tf.int32, name='iter_number')
    losses_with_name = []

    # Policy densities
    target_log_pdf = pi.pd.logp(ac_)
    behavioral_log_pdf = oldpi.pd.logp(ac_)
    log_ratio = target_log_pdf - behavioral_log_pdf

    # Split operations

    disc_rew_split = tf.stack(tf.split(disc_rew_ * mask_, n_episodes))
    rew_split = tf.stack(tf.split(rew_ * mask_, n_episodes))
    log_ratio_split = tf.stack(tf.split(log_ratio * mask_, n_episodes))
    target_log_pdf_split = tf.stack(tf.split(target_log_pdf * mask_, n_episodes))
    behavioral_log_pdf_split = tf.stack(tf.split(behavioral_log_pdf * mask_, n_episodes))
    mask_split = tf.stack(tf.split(mask_, n_episodes))

    # Renyi divergence
    emp_d2_split = tf.stack(tf.split(pi.pd.renyi(oldpi.pd, 2) * mask_, n_episodes))
    emp_d2_cum_split = tf.reduce_sum(emp_d2_split, axis=1)
    empirical_d2 = tf.reduce_mean(tf.exp(emp_d2_cum_split))

    # Return
    ep_return = tf.reduce_sum(mask_split * disc_rew_split, axis=1)

    if clipping:
        rew_split = tf.clip_by_value(rew_split, -1, 1)

    if center_return:
        ep_return = ep_return - tf.reduce_mean(ep_return)
        rew_split = rew_split - (tf.reduce_sum(rew_split) / (tf.reduce_sum(mask_split) + 1e-24))

    discounter = [pow(gamma, i) for i in range(0, horizon)] # Decreasing gamma
    discounter_tf = tf.constant(discounter)
    disc_rew_split = rew_split * discounter_tf

    return_mean = tf.reduce_mean(ep_return)
    return_std = U.reduce_std(ep_return)
    return_max = tf.reduce_max(ep_return)
    return_min = tf.reduce_min(ep_return)
    return_abs_max = tf.reduce_max(tf.abs(ep_return))
    return_step_max = tf.reduce_max(tf.abs(rew_split)) # Max step reward
    return_step_mean = tf.abs(tf.reduce_mean(rew_split))
    positive_step_return_max = tf.maximum(0.0, tf.reduce_max(rew_split))
    negative_step_return_max = tf.maximum(0.0, tf.reduce_max(-rew_split))
    return_step_maxmin = tf.abs(positive_step_return_max - negative_step_return_max)

    losses_with_name.extend([(return_mean, 'InitialReturnMean'),
                             (return_max, 'InitialReturnMax'),
                             (return_min, 'InitialReturnMin'),
                             (return_std, 'InitialReturnStd'),
                             (empirical_d2, 'EmpiricalD2'),
                             (return_step_max, 'ReturnStepMax'),
                             (return_step_maxmin, 'ReturnStepMaxmin')])

    if iw_method == 'pdis':
        # log_ratio_split cumulative sum
        log_ratio_cumsum = tf.cumsum(log_ratio_split, axis=1)
        # Exponentiate
        ratio_cumsum = tf.exp(log_ratio_cumsum)
        # Multiply by the step-wise reward (not episode)
        ratio_reward = ratio_cumsum * disc_rew_split
        # Average on episodes
        ratio_reward_per_episode = tf.reduce_sum(ratio_reward, axis=1)
        w_return_mean = tf.reduce_sum(ratio_reward_per_episode, axis=0) / n_episodes
        # Get d2(w0:t) with mask
        d2_w_0t = tf.exp(tf.cumsum(emp_d2_split, axis=1)) * mask_split # LEAVE THIS OUTSIDE
        # Sum d2(w0:t) over timesteps
        episode_d2_0t = tf.reduce_sum(d2_w_0t, axis=1)
        # Sample variance
        J_sample_variance = (1/(n_episodes-1)) * tf.reduce_sum(tf.square(ratio_reward_per_episode - w_return_mean))
        losses_with_name.append((J_sample_variance, 'J_sample_variance'))
        losses_with_name.extend([(tf.reduce_max(ratio_cumsum), 'MaxIW'),
                                 (tf.reduce_min(ratio_cumsum), 'MinIW'),
                                 (tf.reduce_mean(ratio_cumsum), 'MeanIW'),
                                 (U.reduce_std(ratio_cumsum), 'StdIW')])
        losses_with_name.extend([(tf.reduce_max(d2_w_0t), 'MaxD2w0t'),
                                 (tf.reduce_min(d2_w_0t), 'MinD2w0t'),
                                 (tf.reduce_mean(d2_w_0t), 'MeanD2w0t'),
                                 (U.reduce_std(d2_w_0t), 'StdD2w0t')])

    elif iw_method == 'is':
        iw = tf.exp(tf.reduce_sum(log_ratio_split, axis=1))
        print("SHAPE IW=", iw.get_shape().as_list())
        print("SHAPE log_ratio_split=", log_ratio_split.get_shape().as_list())
        if iw_norm == 'none':
            iwn = iw / n_episodes
            w_return_mean = tf.reduce_sum(iwn * ep_return)
            J_sample_variance = (1/(n_episodes-1)) * tf.reduce_sum(tf.square(iw * ep_return - w_return_mean))
            losses_with_name.append((J_sample_variance, 'J_sample_variance'))
        elif iw_norm == 'sn':
            iwn = iw / tf.reduce_sum(iw)
            w_return_mean = tf.reduce_sum(iwn * ep_return)
        elif iw_norm == 'regression':
            iwn = iw / n_episodes
            mean_iw = tf.reduce_mean(iw)
            beta = tf.reduce_sum((iw - mean_iw) * ep_return * iw) / (tf.reduce_sum((iw - mean_iw) ** 2) + 1e-24)
            w_return_mean = tf.reduce_mean(iw * ep_return - beta * (iw - 1))
        else:
            raise NotImplementedError()
        ess_classic = tf.linalg.norm(iw, 1) ** 2 / tf.linalg.norm(iw, 2) ** 2
        sqrt_ess_classic = tf.linalg.norm(iw, 1) / tf.linalg.norm(iw, 2)
        ess_renyi = n_episodes / empirical_d2
        losses_with_name.extend([(tf.reduce_max(iwn), 'MaxIWNorm'),
                                 (tf.reduce_min(iwn), 'MinIWNorm'),
                                 (tf.reduce_mean(iwn), 'MeanIWNorm'),
                                 (U.reduce_std(iwn), 'StdIWNorm'),
                                 (tf.reduce_max(iw), 'MaxIW'),
                                 (tf.reduce_min(iw), 'MinIW'),
                                 (tf.reduce_mean(iw), 'MeanIW'),
                                 (U.reduce_std(iw), 'StdIW'),
                                 (ess_classic, 'ESSClassic'),
                                 (ess_renyi, 'ESSRenyi')])
    elif iw_method == 'rbis':
        # Check if we need to cluster rewards
        rew_clustering_options = reward_clustering.split(':')
        if reward_clustering == 'none':
            pass # Do nothing
        elif rew_clustering_options[0] == 'global':
            assert len(rew_clustering_options) == 2, "Reward clustering: Provide the correct number of parameters"
            N = int(rew_clustering_options[1])
            tf.add_to_collection('prints', tf.Print(ep_return, [ep_return], 'ep_return', summarize=20))
            global_rew_min = tf.Variable(float('+inf'), trainable=False)
            global_rew_max = tf.Variable(float('-inf'), trainable=False)
            rew_min = tf.reduce_min(ep_return)
            rew_max = tf.reduce_max(ep_return)
            global_rew_min = tf.assign(global_rew_min, tf.minimum(global_rew_min, rew_min))
            global_rew_max = tf.assign(global_rew_max, tf.maximum(global_rew_max, rew_max))
            interval_size = (global_rew_max - global_rew_min) / N
            ep_return = tf.floordiv(ep_return, interval_size) * interval_size
        elif rew_clustering_options[0] == 'batch':
            assert len(rew_clustering_options) == 2, "Reward clustering: Provide the correct number of parameters"
            N = int(rew_clustering_options[1])
            rew_min = tf.reduce_min(ep_return)
            rew_max = tf.reduce_max(ep_return)
            interval_size = (rew_max - rew_min) / N
            ep_return = tf.floordiv(ep_return, interval_size) * interval_size
        elif rew_clustering_options[0] == 'manual':
            assert len(rew_clustering_options) == 4, "Reward clustering: Provide the correct number of parameters"
            N, rew_min, rew_max = map(int, rew_clustering_options[1:])
            interval_size = (rew_max - rew_min) / N
            # Clip to avoid overflow and cluster
            ep_return = tf.clip_by_value(ep_return, rew_min, rew_max)
            ep_return = tf.floordiv(ep_return, interval_size) * interval_size
        else:
            raise Exception('Unrecognized reward clustering scheme.')

        # Get pdfs for episodes
        target_log_pdf_episode = tf.reduce_sum(target_log_pdf_split, axis=1)
        behavioral_log_pdf_episode = tf.reduce_sum(behavioral_log_pdf_split, axis=1)
        # Normalize log_proba (avoid as overflows as possible)
        normalization_factor = tf.reduce_mean(tf.stack([target_log_pdf_episode, behavioral_log_pdf_episode]))
        target_norm_log_pdf_episode = target_log_pdf_episode - normalization_factor
        behavioral_norm_log_pdf_episode = behavioral_log_pdf_episode - normalization_factor
        # Exponentiate
        target_pdf_episode = tf.clip_by_value(tf.cast(tf.exp(target_norm_log_pdf_episode), tf.float64), 1e-300, 1e+300)
        behavioral_pdf_episode = tf.clip_by_value(tf.cast(tf.exp(behavioral_norm_log_pdf_episode), tf.float64), 1e-300, 1e+300)
        tf.add_to_collection('asserts', tf.assert_positive(target_pdf_episode, name='target_pdf_positive'))
        tf.add_to_collection('asserts', tf.assert_positive(behavioral_pdf_episode, name='behavioral_pdf_positive'))
        # Compute the merging matrix (reward-clustering) and the number of clusters
        reward_unique, reward_indexes = tf.unique(ep_return)
        episode_clustering_matrix = tf.cast(tf.one_hot(reward_indexes, n_episodes), tf.float64)
        max_index = tf.reduce_max(reward_indexes) + 1
        trajectories_per_cluster = tf.reduce_sum(episode_clustering_matrix, axis=0)[:max_index]
        tf.add_to_collection('asserts', tf.assert_positive(tf.reduce_sum(episode_clustering_matrix, axis=0)[:max_index], name='clustering_matrix'))
        # Get the clustered pdfs
        clustered_target_pdf = tf.matmul(tf.reshape(target_pdf_episode, (1, -1)), episode_clustering_matrix)[0][:max_index]
        clustered_behavioral_pdf = tf.matmul(tf.reshape(behavioral_pdf_episode, (1, -1)), episode_clustering_matrix)[0][:max_index]
        tf.add_to_collection('asserts', tf.assert_positive(clustered_target_pdf, name='clust_target_pdf_positive'))
        tf.add_to_collection('asserts', tf.assert_positive(clustered_behavioral_pdf, name='clust_behavioral_pdf_positive'))
        # Compute the J
        ratio_clustered = clustered_target_pdf / clustered_behavioral_pdf
        #ratio_reward = tf.cast(ratio_clustered, tf.float32) * reward_unique                                                  # ---- No cluster cardinality
        ratio_reward = tf.cast(ratio_clustered, tf.float32) * reward_unique * tf.cast(trajectories_per_cluster, tf.float32)   # ---- Cluster cardinality
        #w_return_mean = tf.reduce_sum(ratio_reward) / tf.cast(max_index, tf.float32)                                         # ---- No cluster cardinality
        w_return_mean = tf.reduce_sum(ratio_reward) / tf.cast(n_episodes, tf.float32)                                         # ---- Cluster cardinality
        # Divergences
        ess_classic = tf.linalg.norm(ratio_reward, 1) ** 2 / tf.linalg.norm(ratio_reward, 2) ** 2
        sqrt_ess_classic = tf.linalg.norm(ratio_reward, 1) / tf.linalg.norm(ratio_reward, 2)
        ess_renyi = n_episodes / empirical_d2
        # Summaries
        losses_with_name.extend([(tf.reduce_max(ratio_clustered), 'MaxIW'),
                                 (tf.reduce_min(ratio_clustered), 'MinIW'),
                                 (tf.reduce_mean(ratio_clustered), 'MeanIW'),
                                 (U.reduce_std(ratio_clustered), 'StdIW'),
                                 (1-(max_index / n_episodes), 'RewardCompression'),
                                 (ess_classic, 'ESSClassic'),
                                 (ess_renyi, 'ESSRenyi')])
    else:
        raise NotImplementedError()

    if bound == 'J':
        bound_ = w_return_mean
    elif bound == 'std-d2':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_std
    elif bound == 'max-d2':
        var_estimate = tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_abs_max
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_abs_max
    elif bound == 'max-ess':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / delta) / sqrt_ess_classic * return_abs_max
    elif bound == 'std-ess':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / delta) / sqrt_ess_classic * return_std
    elif bound == 'pdis-max-d2':
        # Discount factor
        if gamma >= 1:
            discounter = [float(1+2*(horizon-t-1)) for t in range(0, horizon)]
        else:
            def f(t):
                return pow(gamma, 2*t) + (2*pow(gamma,t)*(pow(gamma, t+1) - pow(gamma, horizon))) / (1-gamma)
            discounter = [f(t) for t in range(0, horizon)]
        discounter_tf = tf.constant(discounter)
        mean_episode_d2 = tf.reduce_sum(d2_w_0t, axis=0) / (tf.reduce_sum(mask_split, axis=0) + 1e-24)
        discounted_d2 = mean_episode_d2 * discounter_tf # Discounted d2
        discounted_total_d2 = tf.reduce_sum(discounted_d2, axis=0) # Sum over time
        bound_ = w_return_mean - tf.sqrt((1-delta) * discounted_total_d2 / (delta*n_episodes)) * return_step_max
    elif bound == 'pdis-mean-d2':
        # Discount factor
        if gamma >= 1:
            discounter = [float(1+2*(horizon-t-1)) for t in range(0, horizon)]
        else:
            def f(t):
                return pow(gamma, 2*t) + (2*pow(gamma,t)*(pow(gamma, t+1) - pow(gamma, horizon))) / (1-gamma)
            discounter = [f(t) for t in range(0, horizon)]
        discounter_tf = tf.constant(discounter)
        mean_episode_d2 = tf.reduce_sum(d2_w_0t, axis=0) / (tf.reduce_sum(mask_split, axis=0) + 1e-24)
        discounted_d2 = mean_episode_d2 * discounter_tf # Discounted d2
        discounted_total_d2 = tf.reduce_sum(discounted_d2, axis=0) # Sum over time
        bound_ = w_return_mean - tf.sqrt((1-delta) * discounted_total_d2 / (delta*n_episodes)) * return_step_mean
    else:
        raise NotImplementedError()

    # Policy entropy for exploration
    ent = pi.pd.entropy()
    meanent = tf.reduce_mean(ent)
    losses_with_name.append((meanent, 'MeanEntropy'))
    # Add policy entropy bonus
    if entropy != 'none':
        scheme, v1, v2 = entropy.split(':')
        if scheme == 'step':
            entcoeff = tf.cond(iter_number_ < int(v2), lambda: float(v1), lambda: float(0.0))
            losses_with_name.append((entcoeff, 'EntropyCoefficient'))
            entbonus = entcoeff * meanent
            bound_ = bound_ + entbonus
        elif scheme == 'lin':
            ip = tf.cast(iter_number_ / max_iters, tf.float32)
            entcoeff_decay = tf.maximum(0.0, float(v2) + (float(v1) - float(v2)) * (1.0 - ip))
            losses_with_name.append((entcoeff_decay, 'EntropyCoefficient'))
            entbonus = entcoeff_decay * meanent
            bound_ = bound_ + entbonus
        elif scheme == 'exp':
            ent_f = tf.exp(-tf.abs(tf.reduce_mean(iw) - 1) * float(v2)) * float(v1)
            losses_with_name.append((ent_f, 'EntropyCoefficient'))
            bound_ = bound_ + ent_f * meanent
        else:
            raise Exception('Unrecognized entropy scheme.')

    losses_with_name.append((w_return_mean, 'ReturnMeanIW'))
    losses_with_name.append((bound_, 'Bound'))
    losses, loss_names = map(list, zip(*losses_with_name))

    if use_natural_gradient:
        p = tf.placeholder(dtype=tf.float32, shape=[None])
        target_logpdf_episode = tf.reduce_sum(target_log_pdf_split * mask_split, axis=1)
        grad_logprob = U.flatgrad(tf.stop_gradient(iwn) * target_logpdf_episode, var_list)
        dot_product = tf.reduce_sum(grad_logprob * p)
        hess_logprob = U.flatgrad(dot_product, var_list)
        compute_linear_operator = U.function([p, ob_, ac_, disc_rew_, mask_], [-hess_logprob])

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    assert_ops = tf.group(*tf.get_collection('asserts'))
    print_ops = tf.group(*tf.get_collection('prints'))

    compute_lossandgrad = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_], losses + [U.flatgrad(bound_, var_list), assert_ops, print_ops])
    compute_grad = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_], [U.flatgrad(bound_, var_list), assert_ops, print_ops])
    compute_bound = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_], [bound_, assert_ops, print_ops])
    compute_losses = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_], losses)
    #compute_temp = U.function([ob_, ac_, rew_, disc_rew_, mask_], [ratio_cumsum, discounted_ratio])

    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, n_episodes, horizon, stochastic=True)
        sampler = type("SequentialSampler", (object,), {"collect": lambda self, _: seg_gen.__next__()})()

    U.initialize()

    # Starting optimizing

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=n_episodes)
    rewbuffer = deque(maxlen=n_episodes)

    while True:

        iters_so_far += 1

        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        if callback:
            callback(locals(), globals())

        if iters_so_far >= max_iters:
            print('Finised...')
            break

        logger.log('********** Iteration %i ************' % iters_so_far)

        theta = get_parameter()

        with timed('sampling'):
            seg = sampler.collect(theta)

        add_disc_rew(seg, gamma)

        lens, rets = seg['ep_lens'], seg['ep_rets']
        lenbuffer.extend(lens)
        rewbuffer.extend(rets)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        args = ob, ac, rew, disc_rew, mask, iter_number = seg['ob'], seg['ac'], seg['rew'], seg['disc_rew'], seg['mask'], iters_so_far
        assign_old_eq_new()

        def evaluate_loss():
            loss = compute_bound(*args)
            return loss[0]

        def evaluate_gradient():
            gradient = compute_grad(*args)
            return gradient[0]

        if use_natural_gradient:
            def evaluate_fisher_vector_prod(x):
                return compute_linear_operator(x, *args)[0] + fisher_reg * x

            def evaluate_natural_gradient(g):
                return cg(evaluate_fisher_vector_prod, g, cg_iters=10, verbose=0)
        else:
            evaluate_natural_gradient = None

        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("InitialBound", evaluate_loss())
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        if save_weights:
            logger.record_tabular('Weights', str(get_parameter()))
            import pickle
            file = open('checkpoint.pkl', 'wb')
            pickle.dump(theta, file)

        with timed("offline optimization"):
            theta, improvement = optimize_offline(theta,
                                                  set_parameter,
                                                  line_search,
                                                  evaluate_loss,
                                                  evaluate_gradient,
                                                  evaluate_natural_gradient,
                                                  max_offline_ite=max_offline_iters)

        set_parameter(theta)

        with timed('summaries after'):
            if env.spec.id == 'LQG1D-v0':
                mu1 = pi.eval_mean([[1]])
                mu01 = pi.eval_mean([[0.1]])
                sigma = pi.eval_std()
                logger.record_tabular("LQGmu1", mu1)
                logger.record_tabular("LQGmu01", mu01)
                logger.record_tabular("LQGsigma", sigma)
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        logger.dump_tabular()

    env.close()
