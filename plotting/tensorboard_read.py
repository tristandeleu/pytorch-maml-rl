#!/usr/bin/env python3

import glob
import os
import pprint
import traceback

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        event_list = event_acc.Scalars('Eval_Reward/Mean')
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": ['Eval_Reward/Mean'] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    path = '../log'
    for dir in os.listdir(path):
        dir = os.path.join(path,dir)
        for subdir in os.listdir(dir):
            subdir = os.path.join(dir,subdir)
            event_paths = glob.glob(os.path.join(subdir, "event*"))
            if event_paths:
                pp.pprint("Found tensorflow logs to process:")
                pp.pprint(event_paths)
                all_logs = many_logs2pandas(event_paths)
                pp.pprint("Head of created dataframe")
                pp.pprint(all_logs.head())
                print("saving to pickle file")
                out_file = os.path.join(subdir, "mean.pkl")
                print(out_file)
                all_logs.to_pickle(out_file)
            else:
                print("No event paths have been found.")
