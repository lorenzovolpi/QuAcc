import glob
import itertools as IT
import os

import pandas as pd

from exp.leap.config import PROBLEM, root_dir
from exp.trd.config import get_acc_names, get_CAP_method_names


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.json")


def load_results(filter_methods=None) -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.json"), recursive=True):
        dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def elab(res: pd.DataFrame):
    methods = get_CAP_method_names()
    accs = get_acc_names()

    dfs = []

    for acc_name, method in IT.product(accs, methods):
        df = res.loc[(res["method"] == method) & (res["acc_name"] == acc_name), :].copy()

        best_samples_idx = df.groupby(["uids", "dataset"])["estim_accs"].idxmin()
        df = df.loc[best_samples_idx].reset_index(drop=False)
        print(df)
