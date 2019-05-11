import tensorflow as tf
import numpy as np
from baselines.poise import run

def main1():
    '''
    basic idea for storing all params of behaviourals (with ARRAY)
    '''
    # array of OldPis
    weights = np.array([[1, 2, 3]], dtype=np.float32)
    # placeHolder of OldPis
    v1 = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    # cose brutte che faccio con oldPi
    out = tf.reduce_sum(v1, axis=0)
    # inizio il vero e proprio algo, finchè non converge
    for _ in range(5):
        with tf.Session() as sess:
            # ottimizzo i parametri della nuova policy
            reduced = sess.run(out, feed_dict={v1: weights})
            print(reduced)
            # aggiungo i parametri al benedetto array of OldPis
            reduced = np.array([reduced])
            weights = np.concatenate((weights, reduced), axis=0)

def main2():
    '''
    basic idea for storing all params of behaviourals (with LIST)
    '''
    # array of OldPis
    weights = []
    weights.append([1, 2, 3])
    # placeHolder of OldPis
    v1 = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    # cose brutte che faccio con oldPi
    out = tf.reduce_sum(v1, axis=0)
    # inizio il vero e proprio algo, finchè non converge
    for _ in range(5):
        with tf.Session() as sess:
            # ottimizzo i parametri della nuova policy
            reduced = sess.run(out, feed_dict={v1: weights})
            # print('weights=', weights)
            # print('reduced=', reduced)
            # aggiungo i parametri al benedetto array of OldPis
            weights.append(reduced)


def main3():
    '''
    Load params of behaviourals into policy one at a time
    '''

    weights = np.array([[1, 2, 3], [5, 6, 7]], dtype=np.float32)
    p1 = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    v1 = tf.get_variable(
        name='ass', shape=[3, ], initializer=tf.zeros_initializer)

    with tf.Session() as sess:
        for i in range(len(weights)):
            v1.load(weights[i])
            print(sess.run(v1))
            # reduced = sess.run(map, feed_dict={p1: weights})
            # print(reduced[0])


def main4():
    '''
    Assign params of behaviourals into policy one at a time
    '''

    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       dtype=np.float32)
    p1 = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    v1 = tf.get_variable(name='v1', shape=[3, ], dtype=tf.float32)
    out = tf.map_fn(lambda x: tf.assign(v1, x), p1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        sess.run(init)
        reduced = sess.run(out, feed_dict={p1: weights})
        print(len(reduced))
        print(reduced)
        # print('p1=',reduced[0], 'v1=',reduced[1], 'out=',reduced[2])


def main5():
    '''
    Check sums of tensors
    '''

    arr = np.array([1, 2, 3], dtype=np.float32)
    added = 5.
    p1 = tf.placeholder(shape=arr.shape, dtype=tf.float32)
    p2 = tf.placeholder(dtype=tf.float32)
    out = p1 + p2

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        sess.run(init)
        reduced = sess.run(out, feed_dict={p1: arr, p2: added})
        print(len(reduced))
        print(reduced)


def main6():
    import argparse

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.add_argument('--njobs', type=int, default=10)
        parser.set_defaults(**{name: default})

    parser = argparse.ArgumentParser(description="My parser")

    add_bool_arg(parser, 'feature', default=False)

    args = parser.parse_args()
    print(isinstance(args.njobs, int))
    print('feature', args.feature)


def main7():
    '''
    Check sums of tensors
    '''
    eps=1e-10
    arr = tf.constant(np.array([0, 0, 0], dtype=np.float32)) + eps
    norm1 = tf.linalg.norm(arr, ord=1)
    norm2 = tf.linalg.norm(arr, ord=2)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        sess.run(init)
        n1 = sess.run(norm1)
        n2 = sess.run(norm2)
        print('n1', n1)
        print('n2', n2)


def main8():
    '''
    Test tf.where usage
    '''
    c = tf.constant(2, dtype=tf.float32)
    arr1 = tf.constant(np.array([1, 1, 10], dtype=np.float32))
    arr2 = tf.ones(shape=arr1.get_shape().as_list(), dtype=np.float32) * c
    min = tf.where(tf.less(arr1, arr2), arr1, arr2)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        sess.run(init)
        n1 = sess.run(min)
        print('n1', n1)


if __name__ == "__main__":
    main8()


# ###################################################################
#
# # arrays
# import numpy as np
# import tensorflow as tf
#
# weights = np.array([[1, 2, 3]], dtype=np.float32)
# len(weights)
# len(weights.shape)
# weights = weights.reshape(3,)
# print(weights.T.T)
# weights = np.concatenate((weights, weights), axis=0).astype(dtype=np.int32)
# weights[0]
# isinstance(weights, (np.ndarray))
# np.zeros((5, 2))
# arr = np.array([1, 2, 3], dtype=np.float32)
# arr.shape
# # lists
# l1 = []
# l2 = [1,2,3]
# l2.append(l2)
# l1.append(l2)
# l2
# weights[:2] = 0
# weights
# np.exp(-700).astype(np.float32)
#
# b=str(1) + 'a'
# b
# x= np.array(-11021.833 * 1.292e-05)
# x.astype(np.float32)
# -0.14243974
# # tuples
# a = b = c = 1
# tupla = ()
# tupla
# tupla = tupla + (c,)
# tupla += (a,)
# len(tupla)
# tupla
# # tensors
# t = tf.constant([[1, 2, 3]])
# t = tf.concat((t, t), axis=0)
# t.shape.as_list()
# np.zeros(shape=[3,])
#
# oldt = tf.placeholder(shape=[None, 10],
#                              dtype=tf.float32, name='old_thetas')
# oldpi = tf.placeholder(shape=[None, 10],
#                              dtype=tf.float32, name='oldpi')
#
# tf.map_fn(lambda old: tf.assign(oldpi, old_thetas_))
#
#

#
#
# ##################################################################
#
# import tensorflow as tf
# pt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# numOne = tf.constant(1)
# xi = tf.Variable(0)
# out = tf.Variable(pt)
# tf.global_variables_initializer().run()
#
# xi_ = xi + numOne
# out_ = tf.scatter_update(out, [xi], [tf.cast(numOne, tf.float32)])
# out.assign(out_)
#
# step = tf.group(
#     xi.assign(xi_),
#     )
# with tf.Session() as sess:
#     for i in range(15):
#         sess.run(step)
#     print(out.eval())
#
# ###################################################################
#
# import gym
#
# env = gym.make('CartPole-v0')
# env.spec.id
#
# ###################################################################
#
# import math
#
# def frange(start, end=None, inc=None):
#     "A range function, that does accept float increments..."
#
#     if end == None:
#         end = start + 0.0
#         start = 0.0
#
#     if inc == None:
#         inc = 1.0
#
#     L = []
#     while 1:
#         next = start + len(L) * inc
#         if inc > 0 and next >= end:
#             break
#         elif inc < 0 and next <= end:
#             break
#         L.append(next)
#
#     return L
#
# i = [n/10 for n in range(1, 9)]
# i
#
# # define range() for floats
# special_args = []
# for i in [n/10 for n in range(1, 9)]:
#     for j in range(3):
#         special_args.append([str(i), str(j)])
# special_args
#
# ##########################################################3
#
# th = th_new = 2
# for _ in range(3):
#     print('th_new', th_new)
#     th_new +=1
#     print('th', th)
#
# import numpy as np
# np.array([np.random.uniform(low=-4, high=4)])
# np.array([4])
# a = -np.inf
# b=1e-4
# a+b
#
# np.exp(-55.241) + np.exp()
#
# import numpy as np
# a = np.array([1, 2])
# b = np.zeros(a.shape)
# b
# new = a
# old = new
# new
# old
# new += 3
# old
# id(new)
# id(old)
# a += 10
# old
# behav = np.array(-800).astype(np.float64)
# e = np.exp(behav)
# e
# np.log(e)
# abs(-10)
#
# import copy
# deep_new = copy.deepcopy(new)
# deep_new
# new += 10
# deep_new
#
# import time
# tt =int(str(time.time())[-5:])
# tt
# t = time.localtime(time.time())
# t
# y=t.tm_year
# m=t.tm_mon
# d=t.tm_mday
# '%s-%s-%s_%s%s%s_%s' % (
#     t.tm_hour, t.tm_min, t.tm_sec, t.tm_mday, t.tm_mon, t.tm_year, tt)
#
# s = time.asctime( time.localtime(time.time()) )
# s
# import numpy as np
# a=np.array([1,2,3])
# a*0 + 3
# for i in range(1, 11):
#     print(i/10)
# for i in range(0):
#     print(i)
#
# np.random.uniform(-1, 1, 10)
# np.linspace(-1, 1, 1000)
# #######################################################
#
# from joblib import Parallel, delayed
# from datetime import datetime
# from tqdm import tqdm
# import numpy as np
#
# class Foo():
#     def myfun(x, y):
#         return x**2 + np.dot(y, [1, 0])
#
# y = np.array([2, 2])
# tg = np.linspace(-1, 1, 100)
# tg
# results = Parallel(n_jobs=8)(delayed(Foo.myfun)(i, y) for i in tg)
# b = np.argmax(results)
# results
# isinstance(y, np.array)
#
# import pathos.pools as pp
# class someClass(object):
#         def __init__(self):
#             pass
#         def f(self, x):
#             return x*x
#         def go(self):
#             pool = pp.ProcessPool(4)
#             print(pool.map(self.f, range(10)))
#
# sc = someClass()
# sc.go()
#
#
# ##########################################################
#
# %matplotlib inline
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from matplotlib import animation
# from IPython.display import HTML
# import autograd.numpy as np
# from scipy.optimize import minimize
# from collections import defaultdict
# from itertools import zip_longest
# from functools import partial
#
#
# f  = lambda x: np.tanh(x**2)
# xmin, xmax, xstep = -1, 1, 0.1
# x = np.arange(xmin, xmax + xstep, xstep)
# z = f(x)
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# ax.plot(x, z)
#
# for a, b in enumerate([5, 5, 5]):
#     print(a)
# import numpy as np
# 1e-200/1e-100
# nan = np.nan
# 1/nan
#
# a = np.array(1e-46).astype(np.float32)
# a
# a/a
# miw =  np.array([1, 2, 3])
# mask = np.array([1, 0, 0])
# ep_r = np.array([2, 2, 2])
# mise = miw * ep_r
# mise
# mise * mask
#
# l = [1] + [0, 0]
# np.linspace(-1, 1, 20)
# lin = [[x, 0, np.log(0.11), 0] for x in np.linspace(-1, 1, 10)]
# lin
# count = 0
# for i in range(1000):
#     sample = np.random.normal(0, 0.1, 1)
#     if sample < -0.3 or sample > 0.3:
#         count += 1
# count
# a = 1
# b= 2
# args = a, b
# c = 3
# d = 4
# args_c = args + (c, d, )
# args_c

import numpy as np
import matplotlib.pyplot as plt
x1 = np.linspace(-10, np.log(1.0), 100)
x2 = np.linspace(-1, 1, 100)

y = np.exp(x1)
plt.plot(x1, y)

x1 = [1, 2]
x2 = [3, 4]
x, y = np.meshgrid(x1, x2)
x
y
x = x.reshape((np.prod(x.shape),))
y = y.reshape((np.prod(y.shape),))
x
y
ll = list(zip(x, y))
np.array(ll[0])

from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot([1.], [1.], [1.], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=100)


a = np.linspace(6, -6, 30)
np.exp(a)


# Calculate the grid of parameters to evaluate
grid_size=3
gain_grid = np.linspace(-1, 1, grid_size)
logstd_grid = np.linspace(-20, np.log(1.0), grid_size)
x, y = np.meshgrid(gain_grid, logstd_grid)
X = x.reshape((np.prod(x.shape),))
Y = y.reshape((np.prod(y.shape),))
rho_grid = list(zip(X, Y))
print(x)
print(y)
bound = []
for rho in rho_grid:
    bound_rho = rho[0]+rho[1]
    bound.append(bound_rho)
bound
np.array(bound).reshape((grid_size, grid_size))
for i, xx in enumerate(X):
    print(i)

len(X)

###############################################################################
import json

# as requested in comment
exDict = {1:1, 2:2, 3:3}
fout = "./here.txt"
fo = open(fout, "w")

for k, v in exDict.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo.close()

# dictionaries
from collections import defaultdict

all_seg = defaultdict()
for i in ["all_ob", "all_ac", "all_rew", "all_disc_rew", "all_mask"]:
    all_seg[i] = np.zeros(3)

newd = defaultdict()
newd[0.1] += 1

std_too = False
grid_size = 50
# Calculate the grid of parameters to evaluate
gain_grid = np.linspace(-1, 1, grid_size)
logstd_grid = np.linspace(-10, 0, grid_size)
gain_grid
if std_too:
    threeDplot = True
    x, y = np.meshgrid(gain_grid, logstd_grid)
    X = x.reshape((np.prod(x.shape),))
    Y = y.reshape((np.prod(y.shape),))
    rho_grid = list(zip(X, Y))
else:
    rho_grid = [[x] for x in gain_grid]

n_selections = defaultdict()
ret_sums = defaultdict()
for rho in rho_grid:
    n_selections[str(rho)] = 0
    ret_sums[str(rho)] = 0
n_selections

ll = [1, 2, 3]
sum(ll)
import numpy as np
import matplotlib.pyplot as plt
int(np.ceil(0.3))

k = 4
x = range(1, 5000)
y = np.power(x, (1 / 4))
y = np.ceil(y)
y = np.power(np.ceil(y), 4)
sl = sqrt*np.log(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(x, y)
ax.set_title('Total arms')

# ax.plot(x, sl, color='g')
sum([-1.5666753,  -1.2310921,  -0.04226363, -1.8523657])

plt.close(fig)

ob = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
np.atleast_2d(ob)[0]

import numpy as np

mu_min=-1; mu_max=+1; logstd_min=-4; logst_max=0; den=1;
grid_size = 5
grid_dimension = 4
trainable_std = False
len(rho_grid)


def generate_custom_grid(grid_size, grid_dimension, trainable_std,
                         mu_min=-1, mu_max=+1, logstd_min=-4, logst_max=0):
    mu1 = np.linspace(-1, 1, grid_size)
    mu2 = np.linspace(0, 4, grid_size)
    mu3 = np.linspace(-10, 0, grid_size)
    mu4 = np.linspace(-2, 2, grid_size)
    gain_xyz = [mu1, mu2, mu3, mu4]
    # gain_xyz = np.linspace(mu_min, mu_max, grid_size)
    # gain_xyz = [gain_xyz for i in range(grid_dimension)]
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

rho_grid = generate_grid(grid_size, grid_dimension, trainable_std)

################################
import numpy as np
import os

def how_many_arms(k, total=False):
    x = range(1, 5000)
    y = np.power(x, (1 / k))
    y = np.ceil(y)
    if total:
        y = np.power(np.ceil(y), 2)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(x, y)
    if total:
        ax.set_title('Total arms_k={}'.format(k))
        fname = 'total_arms_'
    else:
        ax.set_title('Arms per dimension_k={}'.format(k))
        fname = 'arms_per_dim_'.format(k)
    # Save plot to given dir
    dir = './how_many_k_mc/'
    siter = fname + 'k_{}'.format(k)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)

for k in [2, 3, 4, 5]:
    how_many_arms(k, total=False)

np.linspace(0, 4, 18)

sum([0.07315415,  2.3851378])

#################################################################
import numpy as np

for i in range(10):
    grid_size = np.ceil(np.sqrt(i))

a = 1
b =2
new_grid=True
delta_t = 'continuous'
if delta_t == 'continuous' and new_grid:
    print(a)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
x = [0.,  1.5, 3. ]
y = [0.,  1.5, 3. ]
z = [[349.872, 338.562, 321.505]
 [336.053, 329.626, 321.158]
 [335.885, 333.606, 327.332]]

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot_surface(x, y, bound, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('1st param')
ax.set_ylabel('2nd param')
ax.set_zlabel('bound')
ax.invert_yaxis()
# y = np.exp(y)
# rho_best[1] = np.exp(rho_best[1])
ax.plot([rho_best[0]], [rho_best[1]], [bound_best],
        markerfacecolor='r', marker='o', markersize=5)
