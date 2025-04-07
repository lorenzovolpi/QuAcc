import glob
import os

import pandas as pd

from exp.leap.config import PROBLEM, root_dir


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.json")


def load_results(filter_methods=None) -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.json"), recursive=True):
        dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)
