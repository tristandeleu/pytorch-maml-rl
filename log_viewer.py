import numpy as np


def main(args):
    logs = np.load(args.log_path)
    # for log in logs:
    #     print(log)
    valid_return_sum = 0
    for valid_return in (logs['valid_returns']):
        valid_return_sum += valid_return.mean()
    print(valid_return_sum / len(logs['valid_returns']))

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Log viewer')

    parser.add_argument('--log_path', type=str, required=True,
    help='path to the log file.')

    args = parser.parse_args()
    main(args)

