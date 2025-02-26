import itertools as IT
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import close

from exp.leap.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.leap.util import load_results


def get_cts(df, method, cts_name):
    return np.array(df.loc[df["method"] == method, cts_name].to_list())


def draw_heatmap(data, plot_names, cbars, **kwargs):
    col = data["col"].to_numpy()[0]
    plot_name = plot_names[col]
    cbar = cbars[col]
    data = data.drop(["col", "row"], axis=1).to_numpy()
    plot = sns.heatmap(data, cbar=cbar, **kwargs)
    plot.set_title(plot_name)


def ctdfiff():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["LEAP(KDEy)", "PHD(KDEy)", "OCE(KDEy)-SLSQP"]
    method_combos = list(IT.combinations(methods, 2))

    parent_dir = os.path.join(root_dir, "plots", PROBLEM)
    os.makedirs(parent_dir, exist_ok=True)

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        print(cls_name, acc, dataset)
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]
        for m1, m2 in method_combos:
            true_cts = get_cts(df, m1, "true_cts")
            assert np.all(true_cts == get_cts(df, m2, "true_cts"))
            estim_cts1 = get_cts(df, m1, "estim_cts")
            estim_cts2 = get_cts(df, m2, "estim_cts")
            ae1 = np.abs(estim_cts1 - true_cts).mean(axis=0)
            ae2 = np.abs(estim_cts2 - true_cts).mean(axis=0)
            comp = np.abs(estim_cts1 - estim_cts2).mean(axis=0)

            _diff = np.vstack([comp, ae1, ae2])
            hm = pd.DataFrame(_diff)
            hm["col"] = np.repeat(np.arange(3), _diff.shape[1])
            hm["row"] = "diff"
            plot = sns.FacetGrid(hm, col="col", row="row")
            plot.map_dataframe(
                draw_heatmap,
                plot_names=[f"{m1} - {m2}", f"{m1} ae", f"{m2} ae"],
                cbars=[False, False, True],
                vmin=np.min(_diff),
                vmax=np.max(_diff),
                cmap="rocket_r",
                annot=_diff.shape[1] <= 4,
            )

            path = os.path.join(parent_dir, f"[{cls_name} - {dataset}]{m1}_vs_{m2}.png")
            plot.figure.savefig(path)
            plot.figure.clear()
            close(plot.figure)


if __name__ == "__main__":
    ctdfiff()
