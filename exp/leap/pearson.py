import itertools as IT
import os

import pandas as pd
from scipy.stats import pearsonr

from exp.leap.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, get_method_names, root_dir
from exp.leap.util import load_results


def pearson():
    res = load_results()

    classifiers = get_classifier_names()
    methods = get_method_names(with_oracle=False)
    accs = get_acc_names()

    data = {
        "classifier": [],
        "method": [],
        "rval": [],
        "pval": [],
    }
    for cls_name, acc_name, method in IT.product(classifiers, accs, methods):
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc_name) & (res["method"] == method), :]
        true_acc = df["true_accs"].to_numpy()
        ae = df["acc_err"].to_numpy()
        pson = pearsonr(true_acc, ae)
        rval, pval = float(pson.statistic), float(pson.pvalue)
        data["classifier"].append(cls_name)
        data["method"].append(method)
        data["rval"].append(rval)
        data["pval"].append(pval)

    df = pd.DataFrame.from_dict(data, orient="columns")
    pivot = pd.pivot_table(df, index="classifier", columns="method", values="rval")
    with open(os.path.join(root_dir, "tables", f"pearson_{PROBLEM}.html"), "w") as f:
        pivot.to_html(f)


if __name__ == "__main__":
    pearson()
