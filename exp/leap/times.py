import itertools as IT
import os

import pandas as pd

from exp.critdd import get_acc_names
from exp.leap.config import PROBLEM, get_dataset_names, get_method_names, root_dir
from exp.leap.util import load_results, rename_datasets


def times():
    res = load_results()

    classifiers = ["LR"]
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["LEAP(KDEy)", "PHD(KDEy)"]

    parent_dir = os.path.join(root_dir, "times")
    os.makedirs(parent_dir, exist_ok=True)

    for cls_name, acc in IT.product(classifiers, accs):
        df = res.loc[
            (res["classifier"] == cls_name)
            & (res["acc_name"] == acc)
            & (res["dataset"].isin(datasets))
            & (res["method"].isin(methods)),
            ["t_train", "t_test_ave", "method", "dataset"],
        ]

        pivot = pd.pivot_table(df, index="dataset", columns="method", values=["t_train", "t_test_ave"])
        with open(os.path.join(parent_dir, f"{cls_name}_{PROBLEM}.html"), "w") as f:
            pivot.to_html(f)


if __name__ == "__main__":
    times()
