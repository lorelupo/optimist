import os
import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import colorize
from contextlib import contextmanager
from baselines import logger
from baselines.plotting_tools import plot3D_bound_profile, plot_bound_profile, \
    render, plot_ess


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize(
        'done in %.3f seconds' % (time.time() - tstart), color='magenta'))


def eval_trajectory(env, pol, gamma, horizon, feature_fun, rescale_ep_return):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ob = np.reshape(ob, newshape=s.shape)
        ret += r
        disc_ret += gamma**t * r
        t += 1
        if rescale_ep_return:
            # Rescale episodic return in [0, 1]
            if env.spec.id == 'LQG1D-v0':
                # Rescale episodic return in [0, 1]
                # (Hp: r takes values in [0, 1])
                ret = ret / horizon
                max_disc_ret = (1 - gamma**(horizon)) / (1 - gamma)
                disc_ret = disc_ret / max_disc_ret
            else:
                raise NotImplementedError

    return ret, disc_ret, t


def generate_grid(grid_size, grid_dimension, trainable_std,
                  mu_min=-0.2, mu_max=0.2, logstd_min=-4, logst_max=0):
    # mu1 = np.linspace(-1, 1, grid_size)
    # mu2 = np.linspace(0, 20, grid_size)
    # mu3 = np.linspace(-10, 0, grid_size)
    # mu4 = np.linspace(-2, 2, grid_size)
    # gain_xyz = [mu1, mu2]
    gain_xyz = np.linspace(mu_min, mu_max, grid_size)
    gain_xyz = [gain_xyz for i in range(grid_dimension)]
    if trainable_std:
        logstd_xyz = np.linspace(logstd_min, logst_max, grid_size)
        logstd_xyz = [logstd_xyz for i in range(grid_dimension)]
        xyz = gain_xyz + logstd_xyz
        xyz = np.array(np.meshgrid(*xyz))
        XYZ = xyz.reshape(xyz.shape[0], (np.prod(xyz[0].shape)))
    else:
        xyz = np.array(np.meshgrid(*gain_xyz))
        XYZ = xyz.reshape(len(xyz), (np.prod(xyz[0].shape)))

    return list(zip(*XYZ)), gain_xyz, xyz


def best_of_grid(policy, grid_size_1d, mu_min, mu_max, grid_dimension,
                 trainable_std, rho_init, old_rhos_list,
                 iters_so_far, mask_iters,
                 set_parameters, set_parameters_old,
                 delta_cst, renyi_components_sum,
                 evaluate_behav, den_mise,
                 evaluate_behav_last_sample,
                 evaluate_bound, evaluate_renyi, evaluate_roba,
                 filename, plot_bound, plot_ess_profile, delta_t, new_grid):

    # Compute MISE's denominator and Renyi bound
    # evaluate the last behav over all samples and add to the denominator
    set_parameters_old(old_rhos_list[-1])
    behav_t = evaluate_behav()
    den_mise = (den_mise + np.exp(behav_t)) * mask_iters
    # print(den_mise)
    for i in range(len(old_rhos_list) - 1):
        # evaluate all the behavs (except the last) over the last sample
        set_parameters_old(old_rhos_list[i])
        behav = evaluate_behav_last_sample()
        # print('behhaaaaavvvv', np.exp(behav))
        den_mise[iters_so_far-1] = den_mise[iters_so_far-1] + np.exp(behav)

    # Compute the log of MISE's denominator
    eps = 1e-24  # to avoid inf weights and nan bound
    den_mise_it = (den_mise + eps) / iters_so_far
    den_mise_log = np.log(den_mise_it) * mask_iters

    # Calculate the grid of parameters to evaluate
    rho_grid, gain_grid, xyz = \
        generate_grid(grid_size_1d, grid_dimension, trainable_std,
                      mu_min=mu_min, mu_max=mu_max)
    logger.record_tabular("GridSize", len(rho_grid))

    # Evaluate the set of parameters and retain the best one
    bound = []
    mise = []
    bonus = []
    ess_d2 = []
    ess_miw = []
    bound_best = 0
    renyi_bound_best = 0
    # print('rho_grid', rho_grid)
    if new_grid and delta_t == 'continuous':
        print(colorize('computing renyi bound from scratch', color='magenta'))
    for i, rho in enumerate(rho_grid):
        set_parameters(rho)
        if new_grid and delta_t == 'continuous':
            for old_rho in old_rhos_list:
                set_parameters_old(old_rho)
                renyi_component = evaluate_renyi()
                renyi_components_sum[i] += 1 / renyi_component
            renyi_bound = 1 / renyi_components_sum[i]
        else:
            set_parameters_old(old_rhos_list[-1])
            renyi_component = evaluate_renyi()
            renyi_components_sum[i] += 1 / renyi_component
            renyi_bound = 1 / renyi_components_sum[i]

        bound_rho = evaluate_bound(den_mise_log, renyi_bound)
        bound.append(bound_rho)
        if bound_rho > bound_best:
            bound_best = bound_rho
            rho_best = rho
            renyi_bound_best = renyi_bound
        if plot_bound == 1:
            # Evaluate bounds' components for plotting
            mise_rho, bonus_rho, ess_d2_rho, ess_miw_rho = \
                evaluate_roba(den_mise_log, renyi_bound)
            mise.append(mise_rho)
            bonus.append(bonus_rho)
            ess_d2.append(ess_d2_rho)
            ess_miw.append(ess_miw_rho)

    # Calculate improvement
    # set_parameters(rho_init)
    # improvement = bound_best - evaluate_bound(den_mise_log, renyi_bound)
    improvement = 0

    # Plot the profile of the bound and its components
    if plot_bound == 2:
        bound = np.array(bound).reshape((grid_size_1d, grid_size_1d))
        # mise = np.array(mise).reshape((grid_size_std, grid_size))
        plot3D_bound_profile(xyz[0], xyz[1], bound, rho_best,
                             bound_best, iters_so_far, filename)
    elif plot_bound == 1:
        plot_bound_profile(gain_grid[0], bound, mise, bonus, rho_best[0],
                           bound_best, iters_so_far, filename)
        # plot_ess(gain_grid, ess_d2, iters_so_far, 'd2_' + filename)
        # plot_ess(gain_grid, ess_miw, iters_so_far, 'miw_' + filename)

    return rho_best, improvement, den_mise_log, den_mise, \
        renyi_components_sum, renyi_bound_best


def learn(env_name, make_env, seed, make_policy, *,
          max_iters,
          horizon,
          drho,
          delta,
          gamma,
          multiple_init=None,
          sampler=None,
          feature_fun=None,
          iw_norm='none',
          bound_type='max-ess',
          max_offline_iters=10,
          save_weights=False,
          render_after=None,
          grid_size_1d=None,
          mu_min=None,
          mu_max=None,
          truncated_mise=True,
          delta_t=None,
          k=2,
          filename=None,
          find_optimal_arm=False,
          plot_bound=False,
          plot_ess_profile=False,
          trainable_std=False,
          rescale_ep_return=False):
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
    losses_with_name = []

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    env.seed(seed)

    # Build the higher level target and behavioral policies
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)
    logger.record_tabular('NumTrainableParams', int(pi._n_higher_params))

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('higher')]
    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    d = sum(shapes)
    # Get all oldpi's learnable parameters
    all_var_list_old = oldpi.get_trainable_variables()
    var_list_old = \
        [v for v in all_var_list_old
         if v.name.split('/')[1].startswith('higher')]
    # Get hyperpolicy's logstd
    higher_logstd_list = [pi.get_higher_logstd()]

    # My Placeholders
    actor_params_ = tf.placeholder(shape=[max_iters, pi._n_actor_weights],
                                   name='actor_params', dtype=tf.float32)
    last_actor_param_ = tf.placeholder(shape=(pi._n_actor_weights),
                                       name='last_actor_params',
                                       dtype=tf.float32)
    den_mise_log_ = tf.placeholder(shape=[max_iters], dtype=tf.float32,
                                   name='den_mise')
    renyi_bound_ = tf.placeholder(dtype=tf.float32, name='renyi_bound')
    ret_ = tf.placeholder(dtype=tf.float32, shape=(max_iters), name='ret')
    disc_ret_ = tf.placeholder(dtype=tf.float32, shape=(max_iters),
                               name='disc_ret')
    n_ = tf.placeholder(dtype=tf.float32, name='iter_number')
    n_int = tf.cast(n_, dtype=tf.int32)
    mask_iters_ = tf.placeholder(dtype=tf.float32, shape=(max_iters),
                                 name='mask_iters')
    # grad_ = tf.placeholder(dtype=tf,.float32,
    #                            shape=(d, 1), name='grad')

    # Multiple importance weights (with balance heuristic)
    target_log_pdf = tf.reduce_sum(
        pi.pd.independent_logps(actor_params_), axis=1)
    behavioral_log_pdf = tf.reduce_sum(
        oldpi.pd.independent_logps(actor_params_), axis=1)
    behavioral_log_pdf_last_sample = tf.reduce_sum(
        oldpi.pd.independent_logps(last_actor_param_))
    log_ratio = target_log_pdf - den_mise_log_
    miw = tf.exp(log_ratio) * mask_iters_

    den_mise_log_mean = tf.reduce_sum(den_mise_log_) / n_
    den_mise_log_last = den_mise_log_[n_int-1]
    losses_with_name.extend([(den_mise_log_mean, 'DenMISEMeanLog'),
                             (den_mise_log_[0], 'DenMISELogFirst'),
                             (den_mise_log_last, 'DenMISELogLast'),
                             (miw[0], 'IWFirstEpisode'),
                             (miw[n_int-1], 'IWLastEpisode'),
                             (tf.reduce_sum(miw)/n_, 'IWMean'),
                             (tf.reduce_max(miw), 'IWMax'),
                             (tf.reduce_min(miw), 'IWMin')])

    # Return
    ep_return = disc_ret_
    return_mean = tf.reduce_sum(ep_return) / n_
    return_last = ep_return[n_int - 1]
    return_max = tf.reduce_max(ep_return[:n_int])
    return_min = tf.reduce_min(ep_return[:n_int])
    return_abs_max = tf.reduce_max(tf.abs(ep_return[:n_int]))
    regret = n_ * 5 - tf.reduce_sum(ep_return)
    regret_over_t = 5 - return_mean

    losses_with_name.extend([(return_mean, 'ReturnMean'),
                             (return_max, 'ReturnMax'),
                             (return_min, 'ReturnMin'),
                             (return_last, 'ReturnLastEpisode'),
                             (return_abs_max, 'ReturnAbsMax'),
                             (regret, 'Regret'),
                             (regret_over_t, 'Regret/t')])

    # Regret
    # Exponentiated Renyi divergence between the target and one behavioral
    renyi_component = pi.pd.renyi(oldpi.pd)
    renyi_component = tf.cond(tf.is_nan(renyi_component),
                              lambda: tf.constant(np.inf),
                              lambda: renyi_component)
    renyi_component = tf.cond(renyi_component < 0.,
                              lambda: tf.constant(np.inf),
                              lambda: renyi_component)
    renyi_component = tf.exp(renyi_component)

    if truncated_mise:
        # Bound to d2(target || mixture of behaviorals)/n
        mn = tf.sqrt((n_**2 * renyi_bound_) / tf.log(1 / delta))
        mn_broadcasted = \
            tf.ones(shape=miw.get_shape().as_list(), dtype=np.float32) * mn
        min = tf.where(tf.less(miw, mn_broadcasted), miw, mn_broadcasted)
        mise = tf.reduce_sum(min * ep_return * mask_iters_)/n_
    else:
        # MISE
        mise = tf.reduce_sum(miw * ep_return * mask_iters_)/n_
        losses_with_name.append((mise, 'MISE'))

    # Bounds
    if delta_t == 'continuous':
        tau = tf.ceil(n_**(1 / k))
        delta_cst = delta
        delta = 6 * delta / ((np.pi * n_)**2 * (1 + tau**d))
    elif delta_t == 'discrete':
        delta_cst = delta
        delta = 3 * delta / ((np.pi * n_)**2 * grid_size_1d)
    elif delta_t is None:
        grid_size_1d = 100  # ToDo correggiiiiiiiiiiiiiiii
        delta_cst = delta
        delta = tf.constant(delta)
    else:
        raise NotImplementedError
    losses_with_name.append((delta, 'Delta'))

    if bound_type == 'J':
        bound = mise
    elif bound_type == 'max-renyi':
        if truncated_mise:
            const = return_abs_max * (np.sqrt(2) + 1 / 3) \
                * tf.sqrt(tf.log(1 / delta))
            exploration_bonus = const * tf.sqrt(renyi_bound_)
            bound = mise + exploration_bonus
        else:
            const = return_abs_max * tf.sqrt(1 / delta - 1)
            exploration_bonus = const * tf.sqrt(renyi_bound_)
            bound = mise + exploration_bonus
    else:
        raise NotImplementedError
    losses_with_name.append((mise, 'BoundMISE'))
    losses_with_name.append((exploration_bonus, 'BoundBonus'))
    losses_with_name.append((bound, 'Bound'))

    # ESS estimation by d2
    ess_d2 = n_ / renyi_bound_
    # ESS estimation by miw norms
    eps = 1e-18  # for eps<1e-18 miw_2=0 if weights are zero
    miw_ess = (tf.exp(log_ratio) + eps) * mask_iters_
    miw_1 = tf.linalg.norm(miw_ess, ord=1)
    miw_2 = tf.linalg.norm(miw_ess, ord=2)
    ess_miw = miw_1**2 / miw_2**2

    # Infos
    losses, loss_names = map(list, zip(*losses_with_name))

    # TF functions
    set_parameters = U.SetFromFlat(var_list)
    get_parameters = U.GetFlat(var_list)
    set_parameters_old = U.SetFromFlat(var_list_old)
    # set_higher_logstd = U.SetFromFlat(higher_logstd_list)
    # set_higher_logstd(np.log([0.15, 0.2]))

    compute_behav = U.function(
        [actor_params_], behavioral_log_pdf)
    compute_behav_last_sample = U.function(
        [last_actor_param_], behavioral_log_pdf_last_sample)
    compute_renyi = U.function(
        [], renyi_component)
    compute_bound = U.function(
        [actor_params_, disc_ret_, ret_, n_,
         mask_iters_, den_mise_log_, renyi_bound_], bound)
    compute_grad = U.function(
        [actor_params_, disc_ret_, ret_, n_,
         mask_iters_, den_mise_log_, renyi_bound_],
        U.flatgrad(bound, var_list))
    compute_return_mean = U.function(
        [actor_params_, disc_ret_, ret_, n_,
         mask_iters_], return_mean)
    compute_losses = U.function(
        [actor_params_, disc_ret_, ret_, n_,
         mask_iters_, den_mise_log_, renyi_bound_], losses)
    compute_roba = U.function(
        [actor_params_, disc_ret_, ret_, n_,
         mask_iters_, den_mise_log_, renyi_bound_],
        [mise, exploration_bonus, ess_d2, ess_miw])

    # Tf initialization
    U.initialize()

    # Store behaviorals' params and their trajectories
    old_rhos_list = []
    all_eps = {}
    all_eps['actor_params'] = np.zeros(shape=[max_iters, pi._n_actor_weights])
    all_eps['disc_ret'] = np.zeros(max_iters)
    all_eps['ret'] = np.zeros(max_iters)
    mask_iters = np.zeros(max_iters)
    # Set learning loop variables
    den_mise = np.zeros(mask_iters.shape).astype(np.float32)
    if delta_t == 'continuous':
        renyi_components_sum = None
    else:
        renyi_components_sum = np.zeros(grid_size_1d**d)
    new_grid = True
    grid_size_1d_old = 0
    iters_so_far = 0
    lens = []
    tstart = time.time()
    # Sample actor's params before entering the learning loop
    rho = get_parameters()
    theta = pi.resample()
    all_eps['actor_params'][iters_so_far, :] = theta
    # Establish grid dimension if needed
    grid_dimension = ob_space.shape[0]
    # Learning loop ###########################################################
    while True:
        iters_so_far += 1
        mask_iters[:iters_so_far] = 1

        # Render one episode
        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        # Exit loop in the end
        if iters_so_far - 1 >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iters_so_far)

        # Generate one trajectory
        with timed('sampling'):
            # Sample a trajectory with the newly parametrized actor
            ret, disc_ret, ep_len = eval_trajectory(
                env, pi, gamma, horizon, feature_fun, rescale_ep_return)
            all_eps['ret'][iters_so_far-1] = ret
            all_eps['disc_ret'][iters_so_far-1] = disc_ret
            lens.append(ep_len)

        # Store the parameters of the behavioral hyperpolicy
        old_rhos_list.append(rho)

        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("NumTrajectories", iters_so_far)
            logger.record_tabular("TimestepsSoFar", np.sum(lens))
            logger.record_tabular('AvgEpLen', np.mean(lens))
            logger.record_tabular('MinEpLen', np.min(lens))
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Save policy parameters to disk
        if save_weights:
            logger.record_tabular('Weights', str(get_parameters()))
            import pickle
            file = open('checkpoint.pkl', 'wb')
            pickle.dump(rho, file)

        # Tensor evaluations

        def evaluate_behav():
            return compute_behav(all_eps['actor_params'])

        def evaluate_behav_last_sample():
            args_behav_last = [all_eps['actor_params'][iters_so_far - 1]]
            return compute_behav_last_sample(*args_behav_last)

        def evaluate_renyi_component():
            return compute_renyi()

        args = all_eps['actor_params'], all_eps['disc_ret'], \
            all_eps['ret'], iters_so_far, mask_iters

        def evaluate_bound(den_mise_log, renyi_bound):
            args_bound = args + (den_mise_log, renyi_bound, )
            return compute_bound(*args_bound)

        def evaluate_grad(den_mise_log, renyi_bound):
            args_grad = args + (den_mise_log, renyi_bound, )
            return compute_grad(*args_grad)

        def evaluate_roba(den_mise_log, renyi_bound):
            args_roba = args + (den_mise_log, renyi_bound, )
            return compute_roba(*args_roba)

        if bound_type == 'J':
            evaluate_renyi = None
        elif bound_type == 'max-renyi':
            evaluate_renyi = evaluate_renyi_component
        else:
            raise NotImplementedError

        with timed("Optimization"):
            if find_optimal_arm:
                pass
            elif multiple_init:
                bound = 0
                improvement = 0
                check = False
                for i in range(multiple_init):
                    rho_init = [np.arctanh(np.random.uniform(
                        pi.min_mean, pi.max_mean))]
                    rho_i, improvement_i, den_mise_log_i, bound_i = \
                        optimize_offline(evaluate_roba, pi,
                                         rho_init, drho,
                                         old_rhos_list,
                                         iters_so_far,
                                         mask_iters, set_parameters,
                                         set_parameters_old,
                                         evaluate_behav, evaluate_renyi,
                                         evaluate_bound,
                                         evaluate_grad,
                                         max_offline_ite=max_offline_iters)
                    if bound_i > bound:
                        check = True
                        rho = rho_i
                        improvement = improvement_i
                        den_mise_log = den_mise_log_i
                if not check:
                    den_mise_log = den_mise_log_i
            else:
                if delta_t == 'continuous':
                    grid_size_1d = int(np.ceil(iters_so_far**(1 / k)))
                    if grid_size_1d > grid_size_1d_old:
                        new_grid = True
                        renyi_components_sum = np.zeros(grid_size_1d**d)
                        grid_size_1d_old = grid_size_1d
                rho, improvement, den_mise_log, den_mise, \
                    renyi_components_sum, renyi_bound = \
                    best_of_grid(pi, grid_size_1d, mu_min, mu_max,
                                 grid_dimension, trainable_std,
                                 rho, old_rhos_list,
                                 iters_so_far, mask_iters,
                                 set_parameters, set_parameters_old,
                                 delta_cst, renyi_components_sum,
                                 evaluate_behav, den_mise,
                                 evaluate_behav_last_sample,
                                 evaluate_bound, evaluate_renyi, evaluate_roba,
                                 filename, plot_bound,
                                 plot_ess_profile, delta_t, new_grid)
                new_grid = False
            set_parameters(rho)

        with timed('summaries after'):
            # Sample actor's parameters from hyperpolicy and assign to actor
            if iters_so_far < max_iters:
                theta = pi.resample()
                all_eps['actor_params'][iters_so_far, :] = theta

            if env.spec is not None:
                if env.spec.id == 'LQG1D-v0':
                    mu1_actor = pi.eval_actor_mean([[1]])[0][0]
                    mu1_higher = pi.eval_higher_mean()[0]
                    sigma = pi.eval_higher_std()[0]
                    logger.record_tabular("LQGmu1_actor", mu1_actor)
                    logger.record_tabular("LQGmu1_higher", mu1_higher)
                    logger.record_tabular("LQGsigma_higher", sigma)
                elif env.spec.id == 'MountainCarContinuous-v0':
                    ac1 = pi.eval_actor_mean([[1, 1]])[0][0]
                    mu1_higher = pi.eval_higher_mean()
                    sigma = pi.eval_higher_std()
                    logger.record_tabular("ActionIn1", ac1)
                    logger.record_tabular("MountainCar_mu0_higher", mu1_higher[0])
                    logger.record_tabular("MountainCar_mu1_higher", mu1_higher[1])
                    logger.record_tabular("MountainCar_std0_higher", sigma[0])
                    logger.record_tabular("MountainCar_std1_higher", sigma[1])
            elif env.id is not None:
                if env.id == 'inverted_pendulum':
                    ac1 = pi.eval_actor_mean([[1, 1, 1, 1]])[0][0]
                    mu1_higher = pi.eval_higher_mean()
                    sigma = pi.eval_higher_std()
                    logger.record_tabular("ActionIn1", ac1)
                    logger.record_tabular("InvPendulum_mu0_higher", mu1_higher[0])
                    logger.record_tabular("InvPendulum_mu1_higher", mu1_higher[1])
                    logger.record_tabular("InvPendulum_mu2_higher", mu1_higher[2])
                    logger.record_tabular("InvPendulum_mu3_higher", mu1_higher[3])
                    logger.record_tabular("InvPendulum_std0_higher", sigma[0])
                    logger.record_tabular("InvPendulum_std1_higher", sigma[1])
                    logger.record_tabular("InvPendulum_std2_higher", sigma[2])
                    logger.record_tabular("InvPendulum_std3_higher", sigma[3])
            if find_optimal_arm:
                ret_mean = compute_return_mean(*args)
                logger.record_tabular('ReturnMean', ret_mean)
            else:
                args_losses = args + (den_mise_log, renyi_bound, )
                meanlosses = np.array(compute_losses(*args_losses))
                for (lossname, lossval) in zip(loss_names, meanlosses):
                    logger.record_tabular(lossname, lossval)

        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

def optimize_offline(evaluate_roba, pi,
                     rho_init, drho, old_rhos_list,
                     iters_so_far, mask_iters,
                     set_parameters, set_parameters_old,
                     evaluate_behav, evaluate_renyi,
                     evaluate_bound, evaluate_grad,
                     evaluate_natural_grad=None,
                     grad_tol=1e-4, bound_tol=1e-10, max_offline_ite=10):

    # Compute MISE's denominator and Renyi bound
    den_mise = np.zeros(mask_iters.shape).astype(np.float32)
    renyi_components_sum = 0
    for i in range(len(old_rhos_list)):
        set_parameters_old(old_rhos_list[i])
        behav = evaluate_behav()
        den_mise = den_mise + np.exp(behav)
        renyi_component = evaluate_renyi()
        renyi_components_sum += 1 / renyi_component
    renyi_bound = 1 / renyi_components_sum
    renyi_bound_old = renyi_bound

    # Compute the log of MISE's denominator
    eps = 1e-24  # to avoid inf weights and nan bound
    den_mise = (den_mise + eps) / iters_so_far
    den_mise_log = np.log(den_mise) * mask_iters

    # Set optimization variables
    rho = rho_old = rho_init
    improvement = 0
    improvement_old = 0.

    # Calculate initial bound
    bound = evaluate_bound(den_mise_log, renyi_bound)
    if np.isnan(bound):
        print('Got NaN bound! Stopping!')
        set_parameters(rho_old)
        return rho, improvement, den_mise_log, bound
    bound_old = bound
    print('Initial bound after last sampling:', bound)

    # Print infos about optimization loop
    fmtstr = '%6i %10.3g  %18.3g %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s  %18s %16s %18s %18s %18s'
    print(titlestr % ('iter', 'step size', 'grad norm',
                      'delta rho', 'delta bound ite', 'delta bound tot'))

    # Optimization loop
    if max_offline_ite > 0:
        for i in range(max_offline_ite):

            # Gradient
            grad = evaluate_grad(den_mise_log, renyi_bound)
            # Sanity check for the grad
            if np.any(np.isnan(grad)):
                print('Got NaN grad! Stopping!')
                set_parameters(rho_old)
                return rho_old, improvement, den_mise_log, bound_old

            # Natural gradient
            if pi.diagonal:
                natgrad = grad / (pi.eval_fisher() + 1e-24)
            else:
                raise NotImplementedError
            assert np.dot(grad, natgrad) >= 0

            grad_norm = np.sqrt(np.dot(grad, natgrad))
            # Check that the grad norm is not too small
            if grad_norm < grad_tol:
                print('stopping - grad norm < grad_tol')
                # print('rho', rho)
                # print('rho_old', rho_old)
                # print('rho_init', rho_init)
                return rho, improvement, den_mise_log, bound, renyi_bound

            # Save old values
            rho_old = rho
            improvement_old = improvement
            # Make an optimization step
            alpha = drho / grad_norm**2
            delta_rho = alpha * natgrad
            rho = rho + delta_rho
            set_parameters(rho)
            # Update bounds
            renyi_components_sum = 0
            for i in range(len(old_rhos_list)):
                set_parameters_old(old_rhos_list[i])
                renyi_component = evaluate_renyi()
                renyi_components_sum += 1 / renyi_component
            renyi_bound = 1 / renyi_components_sum
            # Sanity check for the bound
            bound = evaluate_bound(den_mise_log, renyi_bound)
            if np.isnan(bound):
                print('Got NaN bound! Stopping!')
                set_parameters(rho_old)
                return rho_old, improvement_old, den_mise_log, \
                    renyi_bound_old, bound_old
                delta_bound = bound - bound_old

            improvement = improvement + delta_bound
            bound_old = bound
            renyi_bound_old = renyi_bound

            print(fmtstr % (i+1, alpha, grad_norm,
                            delta_rho[0], delta_bound, improvement))

    print('Max number of offline iterations reached.')
    return rho, improvement, den_mise_log, renyi_bound, bound
