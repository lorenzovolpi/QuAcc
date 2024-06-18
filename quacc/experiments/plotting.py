import itertools as IT

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_classifiers, get_method_names
from quacc.experiments.report import Report
from quacc.experiments.run import PROBLEM, basedir, gen_datasets, plots_basedir


def plot_grid_of_diagonals():
    methods = [
        "ATC-MC",
        "DoC",
        "Naive",
        "N2E(ACC-h0)",
        "N2E(KDEy-h0)",
    ]
    classifiers = [
        "KNN_10",
        "LR",
        "SVM(rbf)",
    ]
    for cls_name, (acc_name, _) in IT.product(classifiers, gen_acc_measure()):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        dataset_names = [dataset_name for dataset_name, _ in gen_datasets(only_names=True)]
        rep = Report.load_results(basedir, cls_name, acc_name, dataset_name=dataset_names, method_name=methods)
        df = rep.table_data(mean=False)
        qc.plot.seaborn.plot_diagonal_grid(df, cls_name, acc_name, dataset_names, basedir=plots_basedir, n_cols=5)
        print(f"{cls_name}-{acc_name} plots generated")


if __name__ == "__main__":
    plot_grid_of_diagonals()
