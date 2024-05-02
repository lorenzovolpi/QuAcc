import logging
import os.path
from time import time

import numpy as np
import quapy as qp
from quapy.protocol import UPP

import quacc as qc
from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.cont_table import CAPContingencyTable, LabelledCollection
from quacc.models.model_selection import GridSearchCAP


def method_can_switch(method):
    return method is not None and hasattr(method, "switch") and not isinstance(method, GridSearchCAP)


def fit_or_switch(method: ClassifierAccuracyPrediction, V, acc_fn, is_fit):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit or isinstance(method, GridSearchCAP):
            tinit = time()
            method.fit(V)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_predictions(method: ClassifierAccuracyPrediction, test_prot, oracle=False):
    tinit = time()
    if not oracle:
        estim_accs = [method.predict(Ui.X) for Ui in test_prot()]
    else:
        estim_accs = [method.predict(Ui.X, oracle_prev=Ui.prevalence()) for Ui in test_prot()]
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, t_test_ave


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return tuple(prev[1:])
    else:
        return prev[-1]


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
    _path = os.path.join(qc.env["OUT_DIR"], f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
