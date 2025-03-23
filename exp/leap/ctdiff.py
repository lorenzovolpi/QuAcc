import itertools as IT
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import close

from exp.leap.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.leap.util import load_results, rename_datasets, rename_methods

N_COLS = 4

method_map = {
    "LEAP(ACC)": "LEAP$_{\\mathrm{ACC}}$",
    "LEAP(KDEy)": "LEAP$_{\\mathrm{KDEy}}$",
    "PHD(KDEy)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "OCE(KDEy)-SLSQP": "O-LEAP$_{\\mathrm{KDEy}}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def _savefig(plot, path):
    exts = ["png", "pdf"]
    paths = [f"{path}.{ext}" for ext in exts]
    for p in paths:
        plot.figure.savefig(p)
    plot.figure.clear()
    close(plot.figure)


def get_cts(df, method, cts_name):
    return np.array(df.loc[df["method"] == method, cts_name].to_list())


def draw_heatmap(data, plot_names, **kwargs):
    col = data["col"].to_numpy()[0]
    row = data["row"].to_numpy()[0]
    plot_name = plot_names[col + N_COLS * row]
    data = data.drop(["col", "row"], axis=1).to_numpy()
    plot = sns.heatmap(data, **kwargs)
    plot.set_title(plot_name)


def ctdfiff_couples():
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
            hm["row"] = 0
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

            path = os.path.join(parent_dir, f"[{cls_name} - {dataset}]{m1}_vs_{m2}")
            _savefig(plot, path)


def ctdfiff_true_acc():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    # methods = ["Naive", "LEAP(ACC)", "LEAP(KDEy)", "PHD(KDEy)", "OCE(KDEy)-SLSQP"]
    methods = ["LEAP(ACC)", "LEAP(KDEy)", "PHD(KDEy)", "OCE(KDEy)-SLSQP"]

    parent_dir = os.path.join(root_dir, "ctdiffs", PROBLEM)
    os.makedirs(parent_dir, exist_ok=True)

    res, datasets = rename_datasets(dataset_map, res, datasets)
    res, methods = rename_methods(method_map, res, methods)

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        print(cls_name, acc, dataset)
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]

        cnt = 0

        # true_cts = get_cts(df, methods[0], "true_cts").mean(axis=0)
        # true_df = pd.DataFrame(true_cts)
        # true_df["col"] = 0
        # true_df["row"] = 0
        #
        # cnt += 1
        #
        # plot_names = ["True Acc."]
        # cbars = [False]
        # vmin, vmax = np.min(true_cts), np.max(true_cts)
        # annot = true_cts.shape[1] <= 4

        plot_names, cbars = [], []
        vmin, vmax = 1, 0
        annot = False

        mdfs = []
        for m in methods:
            true_cts = get_cts(df, m, "true_cts")
            estim_cts = get_cts(df, m, "estim_cts")
            ae = np.abs(estim_cts - true_cts).mean(axis=0)

            mdf = pd.DataFrame(ae)
            mdf["col"] = cnt % N_COLS
            # mdf["row"] = cnt // N_COLS
            mdf["row"] = 0
            mdfs.append(mdf)
            plot_names.append(m)
            cbars.append(cnt == len(methods) - 1)
            ae_min, ae_max = np.min(ae), np.max(ae)
            vmin = ae_min if ae_min < vmin else vmin
            vmax = ae_max if ae_max > vmax else vmax
            annot = ae.shape[1] <= 4
            cnt += 1

        # hmdf = pd.concat([true_df] + mdfs, axis=0)
        hmdf = pd.concat(mdfs, axis=0)
        plot = sns.FacetGrid(hmdf, col="col", row="row")
        cbar_ax = plot.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plot = plot.map_dataframe(
            draw_heatmap,
            plot_names=plot_names,
            vmin=vmin,
            vmax=vmax,
            cmap="rocket_r",
            annot=annot,
            cbar_ax=cbar_ax,
        )
        plot.fig.subplots_adjust(right=0.9)

        path = os.path.join(parent_dir, f"{cls_name} - {dataset}")
        _savefig(plot, path)


if __name__ == "__main__":
    ctdfiff_true_acc()
