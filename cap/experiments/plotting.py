import itertools as IT
from traceback import print_exception

import pandas as pd

from cap.experiments.run import CSV_SEP, PROBLEM, PROJECT, log, root_dir
from cap.experiments.util import load_results
from cap.plot.seaborn import plot_diagonal, plot_shift


def plots(df: pd.DataFrame):
    configs = [
        {
            "problem": "binary",
            "classifier": "LR",
            "datasets": ["IMDB", "CCAT", "GCAT"],
            "methods": ["ATC-MC", "DoC", "LEAP(KDEy)", "QuAcc(SLD)", "QuAcc(KDEy)"],
            "accs": ["vanilla_accuracy"],
            "plot": "shift",
        },
        {
            "problem": "binary",
            "classifier": "LR",
            "datasets": ["IMDB", "CCAT", "GCAT"],
            "methods": ["ATC-MC", "DoC", "LEAP(KDEy)", "QuAcc(SLD)", "QuAcc(KDEy)"],
            "train_prevs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "accs": ["vanilla_accuracy"],
            "plot": "diagonal",
        },
    ]
    acc_rename = {
        "vanilla_accuracy": "Vanilla Accuracy",
        "macro-F1": "F1",
    }
    method_rename = {
        "LEAP(KDEy)": "LEAP",
    }

    for cf in configs:
        if PROBLEM != cf["problem"]:
            continue

        classifier, methods = cf["classifier"], cf["methods"]
        train_prevs = cf.get("train_prevs", [None])
        for dataset, acc, tp in IT.product(cf["datasets"], cf["accs"], train_prevs):
            cf_df = df.loc[
                (df["classifier"] == classifier)
                & (df["dataset"] == dataset)
                & (df["method"].isin(methods))
                & (df["acc_name"] == acc),
                :,
            ]
            if tp is not None:
                cf_df = cf_df.loc[cf_df["train_prev"] == tp, :]

            acc_name = acc_rename[acc]
            # rename methods
            for m in methods:
                cf_df.loc[cf_df["method"] == m, "method"] = [
                    method_rename.get(m, m)
                ] * len(cf_df.loc[cf_df["method"] == m])

            if cf["plot"] == "shift":
                plot_shift(
                    cf_df,
                    cls_name=classifier,
                    acc_name=acc,
                    dataset_name=dataset,
                    basedir=PROJECT,
                    problem=PROBLEM,
                    linewidth=2,
                    x_label="Amount of Prior Probability Shift",
                    y_label=f"Prediction Error for {acc_name}",
                )

            if cf["plot"] == "diagonal":
                plot_diagonal(
                    cf_df,
                    cls_name=classifier,
                    acc_name=acc,
                    dataset_name=dataset,
                    basedir=PROJECT,
                    problem=PROBLEM,
                    x_label=f"True {acc_name}",
                    y_label=f"Estimated {acc_name}",
                    file_name=f"{str(int(tp * 100))}_diagonal",
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        results = load_results(PROBLEM, root_dir, CSV_SEP)
        plots(results)
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
