#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:04:32 2018

@author: matteo
"""

import numpy as np
import itertools
from gym import spaces

"""Feature functions"""

class RBF(object):
    def __init__(self, kernel_centers, std):
        self.kernel_centers = kernel_centers
        self.std = std
        self.n_kernels = len(kernel_centers)
        
    def feature_fun(self, s):
        s = np.ravel(s)
        feats = np.zeros(len(self.kernel_centers))
        for k, i in zip(self.kernel_centers, range(len(self.kernel_centers))):
            k = np.ravel(k)
            feats[i] = np.exp(-np.linalg.norm(s - k) / 
                          (2 * self.std**2))
        return feats
    
    def state_space(self):
        return spaces.Box(low=np.array([-1.]*self.n_kernels), 
                       high=np.array([1.]*self.n_kernels))

if __name__=='__main__':
    s = [0., 0.]
    xs = np.linspace(-1.2, 0.6, 3)
    ys = np.linspace(-0.07, 0.07, 4)
    ks = [np.array([x,y]) for (x,y) in itertools.product(xs, ys)]
    f = RBF(ks, 1.)
    print(f.feature_fun(s))
    print(f.state_space())