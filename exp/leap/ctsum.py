import itertools as IT

import numpy as np

from exp.leap.config import get_acc_names, get_classifier_names, get_dataset_names
from exp.leap.util import load_results


def get_cts(df, method, cts_name):
    return np.array(df.loc[df["method"] == method, cts_name].to_list())


def ctsum():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    method = "PHD(KDEy)"

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        print(cls_name, acc, dataset)
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]

        _cts_sum: np.ndarray = get_cts(df, method, "estim_cts").sum(axis=(1, 2))
        _cts_sumno1 = np.nonzero(~np.isclose(_cts_sum, 1.0))[0]
        print(_cts_sumno1)


if __name__ == "__main__":
    ctsum()
