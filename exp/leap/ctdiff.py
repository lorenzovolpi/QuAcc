import itertools as IT

import numpy as np

from exp.leap.config import get_acc_names, get_classifier_names, get_dataset_names
from exp.leap.util import load_results


def get_cts(df, method, cts_name):
    return np.array(df.loc[df["method"] == method, cts_name].to_list())


def ctdfiff():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["LEAP(KDEy)", "OCE(KDEy)-SLSQP"]
    method_combos = list(IT.combinations(methods, 2))

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]
        for m1, m2 in method_combos:
            true_cts = get_cts(df, m1, "true_cts")
            assert np.all(true_cts == get_cts(df, m2, "true_cts"))
            estim_cts1 = get_cts(df, m1, "estim_cts")
            estim_cts2 = get_cts(df, m2, "estim_cts")
            ae1 = np.abs(estim_cts1 - true_cts).mean(axis=0)
            ae2 = np.abs(estim_cts2 - true_cts).mean(axis=0)
            comp = np.abs(estim_cts1 - estim_cts2).mean(axis=0)

        print(cls_name, acc, dataset)
        break


if __name__ == "__main__":
    ctdfiff()
