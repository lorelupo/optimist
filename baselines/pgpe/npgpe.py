#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from baselines import logger
import baselines.common.tf_util as U

"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
    Optimal baseline: Zhao, Tingting, et al. "Analysis and
        improvement of policy gradient estimation." Advances in Neural
        Information Processing Systems. 2011.
"""


def eval_trajectory(env, pol, gamma, task_horizon, feature_fun=None):
    ret = disc_ret = 0

    t = 0
    ob = env.reset()
    done = False
    while not done and t < task_horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ob = np.reshape(ob, newshape=s.shape)
        ret += r
        disc_ret += gamma**t * r
        t += 1

    return ret, disc_ret, t


def learn(make_env, seed, make_policy, *,
          horizon=500,
          max_iters=1000,
          verbose=True,
          step_size_strategy=None,
          use_baseline=True,
          step_size=0.1,
          batch_size=100,
          gamma=0.99,
          trainable_std=False):

    # Logging
    format_strs = []
    if verbose:
        format_strs.append('stdout')

    # Build the environment and the policy
    env = make_env()
    env.seed(seed)
    ob_space = env.observation_space
    ac_space = env.action_space
    pol = make_policy('pol', ob_space, ac_space)
    # Hard-set higher logst
    higher_logstd_list = [pol.get_higher_logstd()]
    set_higher_logstd = U.SetFromFlat(higher_logstd_list)
    # set_higher_logstd(np.log([0.15, 1.5]))

    # Learning iteration
    all_disc_rets, lens = [], []
    n_trajectories = 0
    for it in range(max_iters):
        rho = pol.eval_params()  # Higher-order-policy parameters

        # Batch of episodes
        actor_params, rets, disc_rets = [], [], []
        for ep in range(batch_size):
            theta = pol.resample()
            actor_params.append(theta)
            ret, disc_ret, ep_len = eval_trajectory(
                env, pol, gamma, horizon)
            rets.append(ret)
            disc_rets.append(disc_ret)
            all_disc_rets.append(disc_ret)
            lens.append(ep_len)
            n_trajectories += 1

        logger.log('\n********** Iteration %i ************' % it)
        if verbose > 1:
            print('Higher-order parameters:', rho)
            # print('Fisher diagonal:', pol.eval_fisher())
            # print('Renyi:', pol.renyi(pol))
        logger.record_tabular('ReturnMean', np.mean(all_disc_rets))
        logger.record_tabular('ReturnMax', np.max(all_disc_rets))
        logger.record_tabular('AvgRet', np.mean(rets))
        logger.record_tabular('J', np.mean(disc_rets))
        logger.record_tabular('VarJ', np.var(disc_rets, ddof=1)/batch_size)
        logger.record_tabular('BatchSize', batch_size)
        logger.record_tabular('AvgEpLen', np.mean(lens))
        logger.record_tabular('MinEpLen', np.min(lens))
        logger.record_tabular('TimestepsSoFar', np.sum(lens))
        logger.record_tabular('Iteration', it+1)
        logger.record_tabular('NumTrajectories', n_trajectories)

        if env.spec is not None:
            if env.spec.id == 'MountainCarContinuous-v0':
                ac1 = pol.eval_actor_mean([[1, 1]])[0][0]
                mu1_higher = pol.eval_higher_mean()
                sigma = pol.eval_higher_std()
                logger.record_tabular("ActionIn1", ac1)
                logger.record_tabular("MountainCar_mu0_higher", mu1_higher[0])
                logger.record_tabular("MountainCar_mu1_higher", mu1_higher[1])
                logger.record_tabular("MountainCar_std0_higher", sigma[0])
                logger.record_tabular("MountainCar_std1_higher", sigma[1])
        elif env.id is not None:
            if env.id == 'inverted_pendulum':
                ac1 = pol.eval_actor_mean([[1, 1, 1, 1]])[0][0]
                mu1_higher = pol.eval_higher_mean()
                sigma = pol.eval_higher_std()
                logger.record_tabular("ActionIn1", ac1)
                logger.record_tabular("InvPendulum_mu0_higher", mu1_higher[0])
                logger.record_tabular("InvPendulum_mu1_higher", mu1_higher[1])
                logger.record_tabular("InvPendulum_mu2_higher", mu1_higher[2])
                logger.record_tabular("InvPendulum_mu3_higher", mu1_higher[3])
                logger.record_tabular("InvPendulum_std0_higher", sigma[0])
                logger.record_tabular("InvPendulum_std1_higher", sigma[1])
                logger.record_tabular("InvPendulum_std2_higher", sigma[2])
                logger.record_tabular("InvPendulum_std3_higher", sigma[3])

        # Update higher-order policy
        grad = pol.eval_gradient(actor_params, disc_rets,
                                 use_baseline=use_baseline)
        natgrad = pol.eval_natural_gradient(actor_params, disc_rets,
                                            use_baseline=use_baseline)
        if verbose > 1:
            print('natGrad:', natgrad)

        grad2norm = np.linalg.norm(natgrad, 2)
        gradmaxnorm = np.linalg.norm(natgrad, np.infty)

        step_size_it = {'const': step_size,
                        'norm': step_size/np.sqrt(np.dot(grad, natgrad)) if grad2norm > 0 else 0,
                        'vanish': step_size/np.sqrt(it+1)
                        }.get(step_size_strategy, step_size)
        delta_rho = step_size_it * natgrad
        pol.set_params(rho + delta_rho)

        logger.record_tabular('StepSize', step_size_it)
        logger.record_tabular('NatGradInftyNorm', gradmaxnorm)
        logger.record_tabular('NatGrad2Norm', grad2norm)
        logger.dump_tabular()
