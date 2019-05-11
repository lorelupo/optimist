# Implementing UCB
import numpy as np
import baselines.common.tf_util as U
import time
from baselines import logger
from baselines.plotting_tools import plot3D_bound_profile, plot_bound_profile


def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        # ob = np.reshape(ob, newshape=s.shape)
        ret += r
        disc_ret += gamma**t * r
        t += 1
        # Rescale episodic return in [0, 1] (Hp: r takes values in [0, 1])
        ret_rescaled = ret / horizon
        max_disc_ret = (1 - gamma**(horizon)) / (1 - gamma)  # r =1,1,...
        disc_ret_rescaled = disc_ret / max_disc_ret

    return ret_rescaled, disc_ret_rescaled, t


def learn(make_env,
          make_policy,
          horizon,
          gamma=0.99,
          max_iters=1000,
          filename=None,
          grid_size=100,
          feature_fun=None,
          plot_bound=False):

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Build the higher level policy
    pi = make_policy('pi', ob_space, ac_space)

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('higher')]

    # TF functions
    set_parameters = U.SetFromFlat(var_list)
    get_parameters = U.GetFlat(var_list)

    # Generate the grid of parameters to evaluate
    gain_grid = np.linspace(-1, 1, grid_size)
    rho = get_parameters()
    std_too = (len(rho) == 2)
    if std_too:
        grid_size_std = int(grid_size)
        logstd_grid = np.linspace(-4, 0, grid_size_std)
        x, y = np.meshgrid(gain_grid, logstd_grid)
        X = x.reshape((np.prod(x.shape),))
        Y = y.reshape((np.prod(y.shape),))
        rho_grid = list(zip(X, Y))
    else:
        rho_grid = [[x] for x in gain_grid]

    # initialize loop variables
    n_selections = np.zeros(len(rho_grid))
    ret_sums = np.zeros(len(rho_grid))
    regret = 0
    iter = 0

    # Learning loop
    tstart = time.time()
    while True:
        iter += 1

        # Exit loop in the end
        if iter - 1 >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iter)

        ub = []
        ub_best = 0
        i_best = 0
        average_ret = []
        bonus = []
        for i, rho in enumerate(rho_grid):
            if n_selections[i] > 0:
                average_ret_rho = ret_sums[i] / n_selections[i]
                bonus_rho = np.sqrt(2 * np.log(iter) / n_selections[i])
                ub_rho = average_ret_rho + bonus_rho
                ub.append(ub_rho)
                if not std_too:
                    average_ret.append(average_ret_rho)
                    bonus.append(bonus_rho)
            else:
                ub_rho = 1e100
                ub.append(ub_rho)
                average_ret.append(0)
                bonus.append(1e100)
            if ub_rho > ub_best:
                ub_best = ub_rho
                rho_best = rho
                i_best = i
        # Sample actor's parameters from chosen arm
        set_parameters(rho_best)
        _ = pi.resample()

        # Sample a trajectory with the newly parametrized actor
        _, disc_ret, _ = eval_trajectory(
            env, pi, gamma, horizon, feature_fun)
        ret_sums[i_best] += disc_ret
        regret += (0.96512 - disc_ret)
        n_selections[i_best] += 1

        # Store info about variables of interest
        if env.spec.id == 'LQG1D-v0':
            mu1_actor = pi.eval_actor_mean([[1]])[0][0]
            mu1_higher = pi.eval_higher_mean()[0]
            sigma = pi.eval_higher_std()[0]
            logger.record_tabular("LQGmu1_actor", mu1_actor)
            logger.record_tabular("LQGmu1_higher", mu1_higher)
            logger.record_tabular("LQGsigma_higher", sigma)
        logger.record_tabular("ReturnLastEpisode", disc_ret)
        logger.record_tabular("ReturnMean", sum(ret_sums) / iter)
        logger.record_tabular("Regret", regret)
        logger.record_tabular("Regret/t", regret / iter)
        logger.record_tabular("Iteration", iter)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Plot the profile of the bound and its components
        if plot_bound:
            if std_too:
                ub = np.array(ub).reshape((grid_size_std, grid_size))
                plot3D_bound_profile(x, y, ub, rho_best, ub_best,
                                     iter, filename)
            else:
                plot_bound_profile(gain_grid, ub, average_ret, bonus, rho_best,
                                   ub_best, iter, filename)
        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()
