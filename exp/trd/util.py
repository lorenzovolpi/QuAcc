import glob
import itertools as IT
import os

import numpy as np
import pandas as pd

from exp.leap.config import sample_size
from exp.trd.config import PROBLEM, get_acc_names, get_CAP_method_names, get_classifier_names, root_dir


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.json")


def load_results(filter_methods=None) -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.json"), recursive=True):
        dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def CAP_model_selection(res: pd.DataFrame):
    methods = get_CAP_method_names()
    accs = get_acc_names()

    dfs = []
    # apply model selection for each method and each acc measure
    for acc_name, method in IT.product(accs, methods):
        # filter by method and acc and make a copy
        mdf = res.loc[(res["method"] == method) & (res["acc_name"] == acc_name), :].copy()
        # index data by dataset, sample_id (uids), and classifier
        mdf = mdf.set_index(["dataset", "uids", "classifier"])
        # group data by sample_id and dataset and take the index of the minimum in the "estim_accs" column
        best_idx = mdf.groupby(["uids", "dataset"])["estim_accs"].idxmin()
        # use the index to filter the data, resetting the index
        mdf = mdf.loc[best_idx, :].reset_index(drop=False)
        dfs.append(mdf)

    return pd.concat(dfs, axis=0)


def oracle_model_selection(res: pd.DataFrame):
    accs = get_acc_names()

    dfs = []
    for acc in accs:
        odf = res.loc[res["acc_name"] == acc, :].groupby(["dataset", "uids", "classifier"]).first()
        best_idx = odf.groupby(["uids", "dataset"])["true_accs"].idxmin()
        odf = odf.loc[best_idx, :].reset_index(drop=False)
        odf["method"] = ["oracle"] * len(odf)
        dfs.append(odf)

    return pd.concat(dfs, axis=0)


def no_model_selection(res: pd.DataFrame, only_default=False):
    accs = get_acc_names()
    classifiers = get_classifier_names()
    print(classifiers)

    dfs = []
    for acc, classifier in IT.product(accs, classifiers):
        ndf = (
            res.loc[(res["acc_name"] == acc) & (res["classifier"] == classifier), :]
            .groupby(["dataset", "uids"])
            .first()
            .reset_index(drop=False)
        )
        if only_default and not np.all(ndf["default_c"].to_numpy()):
            continue
        ndf["method"] = ndf["classifier"]
        dfs.append(ndf)

    return pd.concat(dfs, axis=0)


def model_selection(res, only_default=False):
    return pd.concat(
        [
            CAP_model_selection(res),
            oracle_model_selection(res),
            no_model_selection(res, only_default=only_default),
        ],
        axis=0,
    )


if __name__ == "__main__":
    res = load_results()
    print(CAP_model_selection(res))
    print(oracle_model_selection(res))
    print(no_model_selection(res))
