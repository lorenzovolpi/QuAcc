import functools
import json
import os
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import pandas as pd
import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import quacc as qc


def combine_dataframes(dfs, df_index=[]) -> pd.DataFrame:
    if len(dfs) < 1:
        raise ValueError
    if len(dfs) == 1:
        return dfs[0]
    df = dfs[0]
    for ndf in dfs[1:]:
        df = df.join(ndf.set_index(df_index), on=df_index)

    return df


def avg_group_report(df: pd.DataFrame) -> pd.DataFrame:
    def _reduce_func(s1, s2):
        return {(n1, n2): v + s2[(n1, n2)] for ((n1, n2), v) in s1.items()}

    lst = df.to_dict(orient="records")[1:-1]
    summed_series = functools.reduce(_reduce_func, lst)
    idx = df.columns.drop([("base", "T"), ("base", "F")])
    avg_report = {(n1, n2): (v / len(lst)) for ((n1, n2), v) in summed_series.items() if n1 != "base"}
    return pd.DataFrame([avg_report], columns=idx)


def fmt_line_md(s):
    return f"> {s}  \n"


def get_quacc_home():
    home = qc.env["QUACC_DATA"]
    os.makedirs(home, exist_ok=True)
    return home


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, downloaded_path: Path):
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=downloaded_path.name,
    ) as t:
        urlretrieve(url, filename=downloaded_path, reporthook=t.update_to)


def save_json_file(path, data):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_file(path, object_hook=None):
    if not os.path.exists(path):
        raise ValueError("Ivalid path for json file")
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def get_results_path(basedir, cls_name, acc_name, dataset_name, method_name):
    return os.path.join(
        qc.env["OUT_DIR"],
        "results",
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        method_name + ".json",
    )


def get_plots_path(basedir, cls_name, acc_name, dataset_name, plot_type):
    return os.path.join(
        qc.env["OUT_DIR"],
        "plots",
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        plot_type + ".svg",
    )


def get_njobs(n_jobs):
    return qc.env["N_JOBS"] if n_jobs is None else n_jobs


def true_acc(h: BaseEstimator, acc_fn: Callable, U: LabelledCollection):
    y_pred = h.predict(U.X)
    y_true = U.y
    conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=U.classes_)
    return acc_fn(conf_table)


def save_dataset_stats(dataset_name, test_prot, L, V):
    path = os.path.join(qc.env["OUT_DIR"], "dataset_stats", f"{dataset_name}.json")
    test_prevs = [Ui.prevalence() for Ui in test_prot()]
    shifts = [qp.error.ae(L.prevalence(), Ui_prev) for Ui_prev in test_prevs]
    info = {
        "n_classes": L.n_classes,
        "n_train": len(L),
        "n_val": len(V),
        "train_prev": L.prevalence().tolist(),
        "val_prev": V.prevalence().tolist(),
        "test_prevs": [x.tolist() for x in test_prevs],
        "shifts": [x.tolist() for x in shifts],
        "sample_size": test_prot.sample_size,
        "num_samples": test_prot.total(),
    }
    save_json_file(path, info)
