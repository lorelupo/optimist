import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time


def plot_bound_profile(rho_grid, bound, first_term, second_term,
                       point_x, point_y, iter, filename):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(rho_grid, bound, label='bound', color='red', linewidth='2')
    ax.plot(rho_grid, first_term, label='performance', color='blue', linewidth='0.5')
    ax.plot(rho_grid, second_term, label='bonus', color='green', linewidth='0.5')
    ax.plot(point_x, point_y, 'o', color='orange')
    ax.legend(loc='upper right')
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def plot3D_bound_profile(x, y, bound, rho_best, bound_best, iter, filename):

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
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(fname)
    plt.close(fig)


def plot_ess(rho_grid, ess, iter, filename):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(rho_grid, ess, color='blue', linewidth='1')
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def how_many_arms(k, total=False):
    x = range(1, 5000)
    y = np.power(x, (1 / k))
    y = np.ceil(y)
    if total:
        y = np.power(np.ceil(y), 4)
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
    dir = './how_many_k/'
    siter = fname + 'k_{}'.format(k)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def render(env, pi, horizon):
    """
    Shows a test episode on the screen
        env: environment
        pi: policy
        horizon: episode length
    """
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
