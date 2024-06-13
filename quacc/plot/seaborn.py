import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from quacc.plot.utils import get_binned_values
from quacc.utils.commons import get_plots_path

sns.set_theme(style="whitegrid")


def _save_figure(plot: Axes, basedir, cls_name, acc_name, dataset_name, plot_type):
    plotsubdir = "all" if dataset_name == "*" else dataset_name
    file = get_plots_path(basedir, cls_name, acc_name, plotsubdir, plot_type)
    os.makedirs(Path(file).parent, exist_ok=True)
    plot.figure.savefig(file, bbox_inches="tight")
    plot.figure.clear()


def _config_legend(plot: Axes):
    plot.legend(title="")
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


def plot_diagonal_grid(
    dfs: list[pd.DataFrame], cls_name, acc_name, dataset_names, *, basedir=None, file_name=None, n_cols=1
):
    pass


def plot_shift(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_name,
    *,
    n_bins=20,
    basedir=None,
    file_name=None,
    linewidth=1,
    **kwargs,
):
    # binning on shift values
    df.loc[:, "shifts_bin"] = get_binned_values(df, "shifts", n_bins)

    plot = sns.lineplot(
        data=df,
        x="shifts_bin",
        y="acc_err",
        hue="method",
        estimator="mean",
        errorbar=None,
        linewidth=linewidth,
    )

    _config_legend(plot)
    if "x_label" in kwargs:
        plot.set_xlabel(kwargs["x_label"])
    if "y_label" in kwargs:
        plot.set_ylabel(kwargs["y_label"])
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
