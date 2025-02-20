import itertools as IT

from exp.leap.config import get_acc_names, get_classifier_names, get_dataset_names
from exp.leap.util import load_results


def ctdfiff():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["LEAP(KDEy)", "OCE(KDEy)-SLSQP"]
    method_combos = [(m1, m2) for m1, m2 in IT.product(methods, methods) if m1 != m2]

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]
        for m1, m2 in method_combos:
            print(cls_name, acc, dataset)
            print(m1, df.loc[df["method"] == m1, "estim_cts"].to_numpy())
            print(m2, df.loc[df["method"] == m2, "estim_cts"].to_numpy())


if __name__ == "__main__":
    ctdfiff()
