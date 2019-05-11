#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:37:23 2018

@author: matteo
"""
from time import sleep

def replay(env, pol, gamma, task_horizon, n_episodes, feature_fun=None, tau=0):
    for ep in range(n_episodes):
        ret = disc_ret = 0
        
        t = 0
        ob = env.reset()
        done = False
        while not done and t<task_horizon:
            
            env.render()
            sleep(tau)
            s = feature_fun(ob) if feature_fun else ob
            a = pol.act(s)
            ob, r, done, _ = env.step(a)
            ret += r
            disc_ret += gamma**t * r
            t+=1
        print('\nEpisode %i:\n\tLength: %d\n\tReturn: %f\n\tDiscounted Return:%f' %
              (ep+1, t, ret, disc_ret))