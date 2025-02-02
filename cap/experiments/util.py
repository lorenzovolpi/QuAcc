import glob
import itertools as IT
import logging
import os.path
from time import time

import numpy as np
import pandas as pd
import quapy as qp
from quapy.protocol import UPP

import cap
from cap.models.base import ClassifierAccuracyPrediction
from cap.models.cont_table import LabelledCollection
from cap.models.model_selection import GridSearchCAP


def method_can_switch(method):
    return (
        method is not None
        and hasattr(method, "switch")
        and not isinstance(method, GridSearchCAP)
    )


def fit_or_switch(
    method: ClassifierAccuracyPrediction, V, V_posteriors, acc_fn, is_fit
):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit or isinstance(method, GridSearchCAP):
            tinit = time()
            method.fit(V, V_posteriors)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V, V_posteriors)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_predictions(
    method: ClassifierAccuracyPrediction, test_prot, test_prot_posteriors, oracle=False
):
    tinit = time()
    if not oracle:
        estim_accs = method.batch_predict(test_prot, test_prot_posteriors)
    else:
        oracles = [Ui.prevalence() for Ui in test_prot()]
        estim_accs = method.batch_predict(
            test_prot, test_prot_posteriors, oracle_prevs=oracles
        )
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, t_test_ave


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return tuple(prev[1:].tolist())
    else:
        return float(prev[-1])


def prevs_from_prot(prot):
    return [get_plain_prev(Ui.prevalence()) for Ui in prot()]


def get_acc_name(acc_name):
    return {
        "Vanilla Accuracy": "vanilla_accuracy",
        "Macro F1": "macro-F1",
    }


def split_validation(V: LabelledCollection, ratio=0.6):
    v_train, v_val = V.split_stratified(ratio, random_state=qp.environ["_R_SEED"])
    val_prot = UPP(v_val, repeats=100, return_type="labelled_collection")
    return v_train, val_prot


def get_logger(id="quacc"):
    _name = f"{id}_log"
    _parent = cap.env["OUT_DIR"]
    os.makedirs(_parent, exist_ok=True)
    _path = os.path.join(_parent, f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_results(PROBLEM, root_dir, CSV_SEP) -> pd.DataFrame:
    # load results
    dfs = []
    for path in glob.glob(
        os.path.join(root_dir, PROBLEM, "**", "*.csv"), recursive=True
    ):
        dfs.append(pd.read_csv(path, sep=CSV_SEP))
    df = pd.concat(dfs, axis=0)

    # merge quacc results
    merges = {
        "QuAcc(CC)": ["QuAcc(CC)1xn2", "QuAcc(CC)nxn", "QuAcc(CC)1xnp1"],
        "QuAcc(SLD)": ["QuAcc(SLD)1xn2", "QuAcc(SLD)nxn", "QuAcc(SLD)1xnp1"],
        "QuAcc(KDEy)": ["QuAcc(KDEy)1xn2", "QuAcc(KDEy)nxn", "QuAcc(KDEy)1xnp1"],
    }

    classifiers = df["classifier"].unique()
    datasets = df["dataset"].unique()
    acc_names = df["acc_name"].unique()
    train_prevs = df["train_prev"].unique()

    new_dfs = []
    for cls_name, dataset, acc_name, train_prev in IT.product(
        classifiers, datasets, acc_names, train_prevs
    ):
        _df = df.loc[
            (df["classifier"] == cls_name)
            & (df["dataset"] == dataset)
            & (df["acc_name"] == acc_name)
            & (df["train_prev"] == train_prev),
            :,
        ]
        if _df.empty:
            continue

        for new_method, methods in merges.items():
            scores = (
                _df.loc[_df["method"].isin(methods), ["method", "fit_score"]]
                .groupby(["method"])
                .mean()
            )
            if len(scores.index) == 0:
                continue
            best_method = scores.idxmin()["fit_score"]

            best_method_tbl = _df.loc[_df["method"] == best_method, :].copy()
            best_method_tbl["method"] = new_method
            best_method_tbl["best_method"] = best_method
            new_dfs.append(best_method_tbl)

    results = pd.concat([df] + new_dfs, axis=0)

    return results
