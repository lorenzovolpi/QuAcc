import itertools as IT
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

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
        "pearson-r": [],
        "pearson-p": [],
        "spearman-r": [],
        "spearman-p": [],
    }
    for cls_name, acc_name, method in IT.product(classifiers, accs, methods):
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc_name) & (res["method"] == method), :]
        true_acc = df["true_accs"].to_numpy()
        ae = df["acc_err"].to_numpy()
        person_corr = pearsonr(true_acc, ae)
        spearman_corr = spearmanr(true_acc, ae)
        pearson_r, pearson_p = float(person_corr.statistic), float(person_corr.pvalue)
        spearman_r, spearman_p = float(spearman_corr.statistic), float(spearman_corr.pvalue)
        data["classifier"].append(cls_name)
        data["method"].append(method)
        data["pearson-r"].append(pearson_r)
        data["pearson-p"].append(pearson_p)
        data["spearman-r"].append(spearman_r)
        data["spearman-p"].append(spearman_p)

    pearson_df = pd.DataFrame.from_dict(data, orient="columns")

    pearson_pivot = pd.pivot_table(df, index="classifier", columns="method", values="pearson-r")
    with open(os.path.join(root_dir, "tables", f"pearson_{PROBLEM}.html"), "w") as f:
        pearson_pivot.to_html(f)

    spearman_pivot = pd.pivot_table(df, index="classifier", columns="method", values="spearman-r")
    with open(os.path.join(root_dir, "tables", f"spaerman_{PROBLEM}.html"), "w") as f:
        spearman_pivot.to_html(f)


if __name__ == "__main__":
    pearson()
