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
    w = 25
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
    path = '/home/rafael/Documents/projects/FoRL/RL2/meta_batch_size/'
    prev = '100'
    means = []
    steps = []
    for dir in sorted(os.listdir(path)):
        type = re.split('_', dir)[4]
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
        means.append(df['train-AverageReturn'].to_numpy())
        f.close()

    plot(means, steps, prev)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0])))
    plt.legend(handles, labels)
    plt.title('Meta-Batch Size')
    plt.xlabel('Samples Collected')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.xlim(b, e)
    plt.xscale('log')
    plt.savefig('ant_dir_rl2_meta_batch_size.pdf')
