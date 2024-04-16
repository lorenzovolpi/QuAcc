import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from quacc.utils.commons import get_plots_path

sns.set_theme(style="whitegrid")


def _save_figure(plot: Axes, basedir, cls_name, acc_name, dataset_name, plot_type):
    plotsubdir = "all" if dataset_name == "*" else dataset_name
    file = get_plots_path(basedir, cls_name, acc_name, plotsubdir, plot_type)
    os.makedirs(Path(file).parent, exist_ok=True)
    plot.figure.savefig(file, bbox_inches="tight")
    plot.figure.clear()


def _config_legend(plot: Axes):
    sns.move_legend(plot, "lower center", bbox_to_anchor=(1, 0.5), ncol=1)


def plot_diagonal(df: pd.DataFrame, cls_name, acc_name, dataset_name, *, basedir=None, file_name=None):
    plot = sns.scatterplot(data=df, x="true_accs", y="estim_accs", hue="method", alpha=0.5)

    _config_legend(plot)
    return _save_figure(
        plot,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        "diagonal" if file_name is None else file_name,
    )


def plot_shift(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_name,
    *,
    n_bins=20,
    basedir=None,
    file_name=None,
):
    # binning on shift values
    # sh_min, sh_max = np.min(df.loc[:, "shifts"]), np.max(df.loc[:, "shifts"])
    sh_min, sh_max = 0, 1
    bins = np.linspace(sh_min, sh_max, n_bins + 1)
    binwidth = (sh_max - sh_min) / n_bins
    shifts_bin_idx = np.digitize(df.loc[:, "shifts"], bins, right=True)
    bins[1:] = bins[1:] - binwidth / 2
    df.loc[:, "shifts_bin"] = bins[shifts_bin_idx]

    plot = sns.lineplot(
        data=df,
        x="shifts_bin",
        y="acc_err",
        hue="method",
        estimator="mean",
        errorbar=None,
    )

    _config_legend(plot)
    return _save_figure(
        plot,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        "shift" if file_name is None else file_name,
    )


# def plot_delta(
#     df: pd.DataFrame,
#     cls_name,
#     acc_name,
#     dataset_name,
#     *,
#     bins=10,
#     basedir=None,
#     stdev=False,
# ):
#     plot = sns.lineplot(
#         data=df,
#         x="prevs",
#         y="acc_err",
#         hue="method",
#         estimator="mean",
#         errorbar=("sd" if stdev else None),
#     )

#     _config_legend(plot)
#     return _save_figure(
#         plot, basedir, cls_name, acc_name, dataset_name, "stdev" if stdev else "delta"
#     )
