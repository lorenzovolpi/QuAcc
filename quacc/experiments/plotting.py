import itertools as IT

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_classifiers, get_method_names
from quacc.experiments.report import Report
from quacc.experiments.run import PROBLEM, basedir, gen_datasets, plots_basedir


def plot_grid_of_diagonals():
    for (cls_name, _), (acc_name, _) in IT.product(gen_classifiers(), gen_acc_measure()):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        methods = get_method_names(PROBLEM)
        dataset_names = [dataset_name for dataset_name, _ in gen_datasets(only_names=True)]
        reps = [
            Report.load_results(basedir, cls_name, acc_name, dataset_name=dataset_name, method_name=methods)
            for dataset_name in dataset_names
        ]
        dfs = [r.diagonal_plot_data() for r in reps]
        qc.plot.seaborn.plot_diagonal_grid(dfs, cls_name, acc_name, dataset_names, basedir=plots_basedir, n_cols=4)
        print(f"{cls_name}-{acc_name} plots generated")
