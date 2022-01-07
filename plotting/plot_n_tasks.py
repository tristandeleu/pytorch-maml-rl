import os

import numpy as np
import matplotlib.pyplot as plt
import re
import pandas


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


NAME = ['90°', '180°', '270°', '360°']
LINES1 = ['-', '--', '-.', ':']


def plot(means, steps, prev):
    global b, e, c
    w = 15
    means = np.asarray(means)
    std = moving_average(np.std(means, axis=0), w)
    mean = moving_average(np.mean(means, axis=0), w)
    step = steps[0][:len(mean)]
    b = np.max([step[0], b])
    e = np.min([step[-1], e])
    plt.plot(step, mean, LINES1[c], label=prev)
    c += 1
    ci = 1.96 * std / np.sqrt(3)
    plt.fill_between(step, mean - ci, mean + ci, alpha=.2)


if __name__ == "__main__":
    b, e, c = -1, 15768000000, 0
    path = '/home/rafael/Documents/projects/FoRL/RL2/n_tasks/'
    prev = '15'
    means = []
    steps = []
    for dir in sorted(os.listdir(path)):
        type = re.split('_', dir)[5]
        dir = os.path.join(path, dir)
        if type != prev:
            plot(means, steps, prev)

            prev = type
            means = []
            steps = []
        file_path = os.path.join(dir, "progress.csv")
        with open(file_path, 'rb') as f:
            df = pandas.read_csv(f)
        steps.append(df['n_timesteps'].to_numpy())
        try:
            means.append(df['eval-AverageReturn'].to_numpy())
        except KeyError:
            means.append(df['train-AverageReturn'].to_numpy())
        f.close()

    plot(means, steps, prev)

    plt.legend()
    plt.title('Sparse Sampling')
    plt.xlabel('Samples Collected')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.xlim(b, e)
    plt.xscale('log')
    plt.savefig('ant_dir_rl2_n_tasks.pdf')
