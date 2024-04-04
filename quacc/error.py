from functools import wraps
from typing import List

import numpy as np
import quapy as qp
from sklearn.metrics import accuracy_score, f1_score

from quacc.legacy.data import ExtendedPrev


def from_name(err_name):
    assert err_name in ERROR_NAMES, f"unknown error {err_name}"
    callable_error = globals()[err_name]
    return callable_error


# def f1(prev):
#     # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
#     if prev[0] == 0 and prev[1] == 0 and prev[2] == 0:
#         return 1.0
#     elif prev[0] == 0 and prev[1] > 0 and prev[2] == 0:
#         return 0.0
#     elif prev[0] == 0 and prev[1] == 0 and prev[2] > 0:
#         return float('NaN')
#     else:
#         recall = prev[0] / (prev[0] + prev[1])
#         precision = prev[0] / (prev[0] + prev[2])
#         return 2 * (precision * recall) / (precision + recall)


def nae(prevs: np.ndarray, prevs_hat: np.ndarray) -> np.ndarray:
    _ae = qp.error.ae(prevs, prevs_hat)
    # _zae = (2.0 * (1.0 - prevs.min())) / prevs.shape[1]
    _zae = 2.0 / prevs.shape[1]
    return _ae / _zae


def f1(prev: np.ndarray | ExtendedPrev) -> float:
    if isinstance(prev, ExtendedPrev):
        prev = prev.A

    def _score(idx):
        _tp = prev[idx, idx]
        _fn = prev[idx, :].sum() - _tp
        _fp = prev[:, idx].sum() - _tp
        _den = 2.0 * _tp + _fp + _fn
        return 0.0 if _den == 0.0 else (2.0 * _tp) / _den

    if prev.shape[0] == 2:
        return _score(1)
    else:
        _idxs = np.arange(prev.shape[0])
        return np.array([_score(idx) for idx in _idxs]).mean()


def f1e(prev):
    return 1 - f1(prev)


def acc(prev: np.ndarray | ExtendedPrev) -> float:
    if isinstance(prev, ExtendedPrev):
        prev = prev.A
    return np.diag(prev).sum() / prev.sum()


def accd(
    true_prevs: List[np.ndarray | ExtendedPrev],
    estim_prevs: List[np.ndarray | ExtendedPrev],
) -> np.ndarray:
    a_tp = np.array([acc(tp) for tp in true_prevs])
    a_ep = np.array([acc(ep) for ep in estim_prevs])
    return np.abs(a_tp - a_ep)


def maccd(
    true_prevs: List[np.ndarray | ExtendedPrev],
    estim_prevs: List[np.ndarray | ExtendedPrev],
) -> float:
    return accd(true_prevs, estim_prevs).mean()


def from_contingency_table(param1, param2):
    if (
        param2 is None
        and isinstance(param1, np.ndarray)
        and param1.ndim == 2
        and (param1.shape[0] == param1.shape[1])
    ):
        return True
    elif (
        isinstance(param1, np.ndarray)
        and isinstance(param2, np.ndarray)
        and param1.shape == param2.shape
    ):
        return False
    else:
        raise ValueError("parameters for evaluation function not understood")


def vanilla_acc_fn(param1, param2=None):
    if from_contingency_table(param1, param2):
        return _vanilla_acc_from_ct(param1)
    else:
        return accuracy_score(param1, param2)


def macrof1_fn(param1, param2=None):
    if from_contingency_table(param1, param2):
        return macro_f1_from_ct(param1)
    else:
        return f1_score(param1, param2, average="macro")


def _vanilla_acc_from_ct(cont_table):
    return np.diag(cont_table).sum() / cont_table.sum()


def _f1_bin(tp, fp, fn):
    if tp + fp + fn == 0:
        return 1
    else:
        return (2 * tp) / (2 * tp + fp + fn)


def macro_f1_from_ct(cont_table):
    n = cont_table.shape[0]

    if n == 2:
        tp = cont_table[1, 1]
        fp = cont_table[0, 1]
        fn = cont_table[1, 0]
        return _f1_bin(tp, fp, fn)

    f1_per_class = []
    for i in range(n):
        tp = cont_table[i, i]
        fp = cont_table[:, i].sum() - tp
        fn = cont_table[i, :].sum() - tp
        f1_per_class.append(_f1_bin(tp, fp, fn))

    return np.mean(f1_per_class)


def microf1(cont_table):
    n = cont_table.shape[0]

    if n == 2:
        tp = cont_table[1, 1]
        fp = cont_table[0, 1]
        fn = cont_table[1, 0]
        return _f1_bin(tp, fp, fn)

    tp, fp, fn = 0, 0, 0
    for i in range(n):
        tp += cont_table[i, i]
        fp += cont_table[:, i] - tp
        fn += cont_table[i, :] - tp
    return _f1_bin(tp, fp, fn)


ACCURACY_ERROR = {maccd}
ACCURACY_ERROR_SINGLE = {accd}
ACCURACY_ERROR_NAMES = {func.__name__ for func in ACCURACY_ERROR}
ACCURACY_ERROR_SINGLE_NAMES = {func.__name__ for func in ACCURACY_ERROR_SINGLE}
ERROR_NAMES = ACCURACY_ERROR_NAMES | ACCURACY_ERROR_SINGLE_NAMES
