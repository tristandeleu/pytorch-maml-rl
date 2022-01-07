import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

TYPES = ['w/o Meta-Training', '90°', '180°', '270°', '360°']
LINES1 = ['-.', '-', '--', '-', '--']
LINES2 = ['o', '>', 'v', '<', '^']
SIZE = 17

if __name__ == "__main__":
    path = '../log'
    for dir in os.listdir(path):
        type = int(dir[-1])
        label = TYPES[type]
        dir = os.path.join(path, dir)
        means = []
        for subdir in os.listdir(dir):
            subdir = os.path.join(dir, subdir)
            try:
                with open(os.path.join(subdir, 'mean.pkl'), 'rb') as f:
                    df = pkl.load(f)
                    f.close()
                means.append(df['value'].to_numpy())
            except:
                continue
        means = np.asarray(means)[:, :SIZE]
        means = means / 6
        mean = np.mean(means, axis=0)
        std = np.std(means, axis=0)
        ticks = np.arange(SIZE)
        ci = 1.96 * std / np.sqrt(3)

        plt.plot(ticks, mean, LINES1[type], label=label)
        plt.fill_between(ticks, (mean - ci), (mean + ci), alpha=.2)
    plt.legend()
    plt.title('Ant-Dir Environment')
    plt.xlabel('Gradient Steps')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('ant_dir_maml_adapt.pdf')
