import os

import numpy as np
import matplotlib.pyplot as plt
import re
import pandas


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


COLORS = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
counter = 0


def plot(means, steps, prev):
    global counter
    w = 1
    means = np.asarray(means)
    std = moving_average(np.std(means, axis=0), w)
    mean = moving_average(np.mean(means, axis=0), w)
    step = steps[0][:len(mean)]
    plt.plot(step, mean, '-', color=COLORS[counter], label=prev)
    ci = 1.96 * std / np.sqrt(3)
    plt.fill_between(step, mean - ci, mean + ci, color=COLORS[counter], alpha=.2)
    plt.plot(step, [np.mean(mean[-10:])] * len(step), '--', color=COLORS[counter])
    counter += 1


if __name__ == "__main__":
    path = '/home/rafael/Downloads/temp (copy 1)/'
    for dir in sorted(os.listdir(path)):
        type = str.upper(dir)
        dir = os.path.join(path, dir)
        if type == 'PROMP':
            continue
        means = []
        steps = []
        for subdir in os.listdir(dir):
            file_path = os.path.join(dir, subdir, "progress.csv")
            with open(file_path, 'rb') as f:
                df = pandas.read_csv(f)
            if type == 'MAML-PPO' or type == 'MAML-TRPO':
                steps.append(df['n_timesteps'].to_numpy())
                try:
                    means.append(df['Eval Step_1-AverageReturn'].to_numpy())
                except KeyError:
                    means.append(df['Step_1-AverageReturn'].to_numpy())
                f.close()
            elif type == 'PEARL':
                steps.append(df['Number of env steps total'].to_numpy())
                means.append(df['AverageReturn_all_test_tasks'].to_numpy())
            elif type == 'RL2':
                steps.append(df['n_timesteps'].to_numpy())
                means.append(df['train-AverageReturn'].to_numpy())
        _min = 10000000
        for i in means:
            _min = np.min([len(i), _min])

        means = list(map(lambda x: x[:_min], means))
        steps = list(map(lambda x: x[:_min], steps))

        print(steps[0][-1])
        plot(means, steps, type)

    plt.plot([],[],'--',color='black',label='Final Performance')
    plt.legend(loc=4)
    plt.title('Performance of Selected Algorithms')
    plt.xlabel('Samples Collected')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.xscale('log')
    #plt.xticks([1e6,5e6],['10\U00002076','5\U000000D710\U00002076'])
   # plt.ticklabel_format(style='sci')
    plt.xlim(320000,1e9)
    plt.savefig('ant_dir_comparison.pdf')
