import itertools as IT

import pandas as pd

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_bin_datasets, gen_classifiers, get_method_names
from quacc.experiments.report import Report

PROBLEM = "binary"
basedir = PROBLEM
plots_basedir = PROBLEM

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def rename_methods(df: pd.DataFrame, methods: dict):
    for om, nm in methods.items():
        df.loc[df["method"] == om, "method"] = nm


def plot_grid_of_diagonals():
    methods = {
        "ATC-MC": "ATC",
        "DoC": "DoC",
        # "Naive",
        "N2E(ACC-h0)": "P$h$D",
        "N2E(KDEy-h0)": "P$h$D*",
    }
    classifiers = [
        # "KNN_10",
        "LR",
        # "SVM(rbf)",
    ]
    for cls_name, (acc_name, _) in IT.product(classifiers, gen_acc_measure()):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        dataset_names = [dataset_name for dataset_name, _ in gen_datasets(only_names=True)]
        rep = Report.load_results(
            basedir, cls_name, acc_name, dataset_name=dataset_names, method_name=list(methods.keys())
        )
        df = rep.table_data(mean=False)
        rename_methods(df, methods)
        qc.plot.seaborn.plot_diagonal_grid(
            df,
            cls_name,
            acc_name,
            dataset_names,
            basedir=plots_basedir,
            n_cols=5,
            x_label="True Accuracy",
            y_label="Estimated Accuracy",
        )
        print(f"{cls_name}-{acc_name} plots generated")


if __name__ == "__main__":
    plot_grid_of_diagonals()
