#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018

@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
"""

import numpy as np
from baselines import logger
import warnings
from contextlib import contextmanager
import time
from baselines.common import colorize

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

def eval_trajectory(env, pol, gamma, task_horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t<task_horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma**t * r
        t+=1
        
    return ret, disc_ret, t

#BINARY line search
def line_search_binary(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2):
    rho_init = newpol.eval_params()
    low = 0.
    high = None
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    #old_delta_bound = 0.
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = 0.
    epsilon_opt = 0.
    epsilon = 1.
    
    for i in range(max_search_ite):
        rho = rho_init + epsilon * natgrad * alpha
        newpol.set_params(rho)
        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        delta_bound = bound - bound_init        
        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            rho_opt = rho
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        old_epsilon = epsilon
        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.
        if abs(old_epsilon - epsilon) < 1e-6:
            break
    
    return rho_opt, epsilon_opt, delta_bound_opt, i_opt+1

def line_search_parabola(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2):
    epsilon = 1.
    epsilon_old = 0.
    max_increase=2. 
    delta_bound_tol=1e-4
    delta_bound_old = -np.inf
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    rho_old = rho_init = newpol.eval_params()

    for i in range(max_search_ite):

        rho = rho_init + epsilon * alpha * natgrad
        newpol.set_params(rho)

        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return rho_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        
        if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
            epsilon = epsilon_old * max_increase
        else:
            epsilon = epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))
        
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return rho_init, 0., 0., i+1
            else:
                return rho_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        rho_old = rho

    return rho_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(pol, newpol, actor_params, rets, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100, 
                     normalize=True, 
                     use_rmax=True,
                     use_renyi=True,
                     max_search_ite=30,
                     rmax=None, delta=0.2, use_parabola=False):
    improvement = 0.
    rho = pol.eval_params()
    
    
    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", 
                      "gradient norm", "delta bound ite", "delta bound tot"))
    
    natgrad = None
    
    for i in range(max_offline_ite):
        #Candidate policy
        newpol.set_params(rho)
        
        #Bound with gradient
        bound, grad = newpol.eval_bound_and_grad(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement
        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement     

            
        #Natural gradient
        if newpol.diagonal: 
            natgrad = grad/(newpol.eval_fisher() + 1e-24)
        else:
            raise NotImplementedError
        assert np.dot(grad, natgrad) >= 0

        grad_norm = np.sqrt(np.dot(grad, natgrad))
        if grad_norm < grad_tol:
            print("stopping - gradient norm < gradient_tol")
            print(rho)
            return rho, improvement
        
        #Step size search
        alpha = 1. / grad_norm ** 2
        line_search = line_search_parabola if use_parabola else line_search_binary
        rho, epsilon, delta_bound, num_line_search = line_search(pol, 
                                                                 newpol, 
                                                                 actor_params, 
                                                                 rets, 
                                                                 alpha, 
                                                                 natgrad, 
                                                                 normalize=normalize,
                                                                 use_rmax=use_rmax,
                                                                 use_renyi=use_renyi,
                                                                 max_search_ite=max_search_ite,
                                                                 rmax=rmax,
                                                                 delta=delta)
        newpol.set_params(rho)
        improvement+=delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, grad_norm, delta_bound, improvement))
        if delta_bound < bound_tol:
            print("stopping - delta bound < bound_tol")
            print(rho)
            return rho, improvement
    
    return rho, improvement


def learn(env, pol_maker, gamma, initial_batch_size, task_horizon, max_iterations, 
          feature_fun=None, 
          rmax=None,
          normalize=True, 
          use_rmax=True, 
          use_renyi=True,
          max_offline_ite=100, 
          max_search_ite=30,
          verbose=True, 
          save_to=None,
          delta=0.2,
          shift=False,
          reuse=False,
          use_parabola=False):
    
    #Logging
    format_strs = []
    if verbose: format_strs.append('stdout')
    if save_to: format_strs.append('csv')
    logger.configure(dir=save_to, format_strs=format_strs)

    pol = pol_maker('pol')
    newpol = pol_maker('oldpol')
    newpol.set_params(pol.eval_params())
    batch_size = initial_batch_size
    
    #Learning iteration
    actor_params, rets, disc_rets, lens = [], [], [], []
    old_perf = -np.inf
    for it in range(max_iterations):
        logger.log('\n********** Iteration %i ************' % it)
        rho = pol.eval_params() #Higher-order-policy parameters
        if verbose>1:
            logger.log('Higher-order parameters: ', rho)
        if save_to: np.save(save_to + '/weights_' + str(it), rho)
            
        #Batch of episodes
        #TODO: try symmetric sampling
        with timed('Sampling'):
            for ep in range(initial_batch_size):
                frozen_pol = pol.freeze()
                theta = frozen_pol.resample()
                actor_params.append(theta)
                ret, disc_ret, ep_len = eval_trajectory(env, frozen_pol, gamma, task_horizon, feature_fun)
                rets.append(ret)
                disc_rets.append(disc_ret)
                lens.append(ep_len)
        complete = len(rets)>=batch_size
        norm_disc_rets = np.array(disc_rets)
        if shift:
            norm_disc_rets = norm_disc_rets - np.mean(norm_disc_rets)
        rmax = np.max(abs(norm_disc_rets))
        perf = np.mean(norm_disc_rets)
        logger.log('Performance: ', np.mean(perf))
        #if save_to: np.save(save_to + '/rets_' + str(it), rets)
        
        if complete and perf<old_perf and batch_size<5*initial_batch_size:
            #Try with more trajectories
            iter_type = 0
            if verbose: logger.log('Performance loss! Adding more trajectories')
            batch_size+=initial_batch_size
            old_perf = -np.inf #After adding 100, go on anyway
            newpol.set_params(rho)
            complete = False #Policy does not change, so keep the trajectories
        elif complete:
            #When you have enough data, optimize
            iter_type = 1
            if verbose: logger.log('Optimizing')
            with timed('Optimizing offline'):
                rho, improvement = optimize_offline(pol, newpol, actor_params, norm_disc_rets,
                                                    normalize=normalize,
                                                    use_rmax=use_rmax,
                                                    use_renyi=use_renyi,
                                                    max_offline_ite=max_offline_ite,
                                                    max_search_ite=max_search_ite,
                                                    rmax=rmax,
                                                    delta=delta,
                                                    use_parabola=use_parabola)
            newpol.set_params(rho)
            assert(improvement>=0.)
            old_perf = perf
        else:
            iter_type = 2
            if verbose: logger.log('Collecting more data (%d/%d)' % (len(rets), batch_size))
            newpol.set_params(rho)
        
        logger.log('Recap of iteration %i' % it)
        unn_iws = newpol.eval_iws(actor_params, behavioral=pol, normalize=False)
        iws = unn_iws/np.sum(unn_iws)
        ess = np.linalg.norm(unn_iws, 1) ** 2 / np.linalg.norm(unn_iws, 2) ** 2
        J, varJ = newpol.eval_performance(actor_params, norm_disc_rets, behavioral=pol)
        eRenyi = np.exp(newpol.eval_renyi(pol))
        logger.record_tabular('IterType', iter_type)
        logger.record_tabular('Bound', newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta))
        logger.record_tabular('ESSClassic', ess)
        logger.record_tabular('ESSRenyi', batch_size/eRenyi)
        logger.record_tabular('MaxVanillaIw', np.max(unn_iws))
        logger.record_tabular('MinVanillaIw', np.min(unn_iws))
        logger.record_tabular('AvgVanillaIw', np.mean(unn_iws))
        logger.record_tabular('VarVanillaIw', np.var(unn_iws, ddof=1))
        logger.record_tabular('MaxNormIw', np.max(iws))
        logger.record_tabular('MinNormIw', np.min(iws))
        logger.record_tabular('AvgNormIw', np.mean(iws))
        logger.record_tabular('VarNormIw', np.var(iws, ddof=1))
        logger.record_tabular('eRenyi2', eRenyi)
        logger.record_tabular('AvgRet', np.mean(rets))
        logger.record_tabular('VanillaAvgRet', np.mean(rets))
        logger.record_tabular('VarRet', np.var(rets, ddof=1))
        logger.record_tabular('VarDiscRet', np.var(norm_disc_rets, ddof=1))
        logger.record_tabular('AvgDiscRet', np.mean(norm_disc_rets))
        logger.record_tabular('J', J)
        logger.record_tabular('VarJ', varJ)
        logger.record_tabular('BatchSize', batch_size)
        logger.record_tabular('EpsThisIter', initial_batch_size)
        logger.record_tabular('AvgEpLen', np.mean(lens))
        logger.dump_tabular()
        
        #Update
        pol.set_params(newpol.eval_params()) 
        if complete:
            actor_params, rets, disc_rets, lens = [], [], [], []