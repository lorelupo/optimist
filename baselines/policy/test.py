#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:36:13 2018

@author: matteo
"""

from baselines.policy.bounded_mlp_policy import MlpPolicy
import gym
import baselines.envs.lqg1d
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
import numpy as np


sess = U.make_session(1)
sess.__enter__()
set_global_seeds(0)

env = gym.make('LQG1D-v0')
pi = MlpPolicy('pi', env.observation_space, env.action_space, [2, 3], 2 ,
               )#max_mu=10, min_mu=-5, max_std=2, min_std = 0.1)

ob = np.array([0.2])[None]

a, _ = pi.act(True, ob)
print("------------------------")
print(a)
print(pi.max_std)

print(pi.eval_mean(ob))
print(pi.eval_std())