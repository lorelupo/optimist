# coding: utf-8
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import baselines.common.tf_util as U
import time
from baselines import logger
from baselines.plotting_tools import plot3D_bound_profile, plot_bound_profile


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
                  mu_min=-2, mu_max=2, logstd_min=-4, logst_max=0):
    # mu1 = np.linspace(-1, 1, grid_size)
    # mu2 = np.linspace(-10, 10, grid_size)
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


def learn(seed, make_env,
          make_policy,
          horizon,
          delta,
          gamma=0.99,
          max_iters=1000,
          filename=None,
          grid_size_1d=100,
          mu_min=None,
          mu_max=None,
          feature_fun=None,
          plot_bound=False,
          trainable_std=False,
          rescale_ep_return=False):

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space
    env.seed(seed)

    # Build the higher level policy
    pi = make_policy('pi', ob_space, ac_space)

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('higher')]
    # Get hyperpolicy's logstd
    higher_logstd_list = [pi.get_higher_logstd()]

    # TF functions
    set_parameters = U.SetFromFlat(var_list)
    set_higher_logstd = U.SetFromFlat(higher_logstd_list)
    # set_higher_logstd(np.log([0.15, 1.5]))

    # Calculate the grid of parameters to evaluate
    grid_dimension = ob_space.shape[0]
    rho_grid, gain_grid, xyz = \
        generate_grid(grid_size_1d, grid_dimension, trainable_std,
                      mu_min=mu_min, mu_max=mu_max)

    # Learning loop
    regret = 0
    iter = 0
    ret_mu = np.array([0. for _ in range(len(rho_grid))])
    ret_sigma = np.array([0.5 for _ in range(len(rho_grid))])
    selected_rhos = []
    selected_disc_rets = []
    lens = []
    tstart = time.time()
    while True:
        iter += 1

        # Exit loop in the end
        if iter - 1 >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iter)

        # Select the bound maximizing arm
        beta = \
            2 * np.log((np.abs(len(rho_grid)) * (iter * np.pi)**2) / 6 * delta)
        bonus = ret_sigma * np.sqrt(beta)
        bound = ret_mu + bonus
        i_best = np.argmax(bound)
        bound_best = bound[i_best]
        rho_best = rho_grid[i_best]
        selected_rhos.append(rho_best)
        # Sample actor's parameters from chosen arm
        set_parameters(rho_best)
        _ = pi.resample()
        # Sample a trajectory with the newly parametrized actor
        _, disc_ret, ep_len = eval_trajectory(
            env, pi, gamma, horizon, feature_fun, rescale_ep_return)
        selected_disc_rets.append(disc_ret)
        lens.append(ep_len)
        regret += (5 - disc_ret)
        # Create GP regressor and fit it to the arms' returns
        gp = GaussianProcessRegressor()
        gp.fit(selected_rhos, selected_disc_rets)
        ret_mu, ret_sigma = gp.predict(rho_grid, return_std=True)

        # Store info about variables of interest
        if env.spec.id == 'LQG1D-v0':
            mu1_actor = pi.eval_actor_mean([[1]])[0][0]
            mu1_higher = pi.eval_higher_mean()[0]
            sigma_higher = pi.eval_higher_std()[0]
            logger.record_tabular("LQGmu1_actor", mu1_actor)
            logger.record_tabular("LQGmu1_higher", mu1_higher)
            logger.record_tabular("LQGsigma_higher", sigma_higher)
        elif env.spec.id == 'MountainCarContinuous-v0':
            ac1 = pi.eval_actor_mean([[1, 1]])[0][0]
            mu1_higher = pi.eval_higher_mean()
            sigma = pi.eval_higher_std()
            logger.record_tabular("ActionIn1", ac1)  # optimum ~2.458
            logger.record_tabular("MountainCar_mu0_higher", mu1_higher[0])
            logger.record_tabular("MountainCar_mu1_higher", mu1_higher[1])
            logger.record_tabular("MountainCar_std0_higher", sigma[0])
            logger.record_tabular("MountainCar_std1_higher", sigma[1])
        logger.record_tabular("ReturnLastEpisode", disc_ret)
        logger.record_tabular("ReturnMean", sum(selected_disc_rets) / iter)
        logger.record_tabular('AvgEpLen', np.mean(lens))
        logger.record_tabular('MinEpLen', np.min(lens))
        logger.record_tabular("Regret", regret)
        logger.record_tabular("Regret/t", regret / iter)
        logger.record_tabular("Iteration", iter)
        logger.record_tabular("NumTrajectories", iter)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("GridSize", len(rho_grid))

        # Plot the profile of the bound and its components
        if plot_bound == 2:
            bound = np.array(bound).reshape((grid_size_1d, grid_size_1d))
            # mise = np.array(mise).reshape((grid_size_std, grid_size))
            plot3D_bound_profile(xyz[0], xyz[1], bound, rho_best,
                                 bound_best, iter, filename)
        elif plot_bound == 1:
            plot_bound_profile(gain_grid[0], bound,
                               bound_best, iter, filename)

        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()
