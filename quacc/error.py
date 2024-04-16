from functools import wraps
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def from_name(err_name):
    assert err_name in ERROR_NAMES, f"unknown error {err_name}"
    callable_error = globals()[err_name]
    return callable_error


def from_contingency_table(param1, param2):
    if param2 is None and isinstance(param1, np.ndarray) and param1.ndim == 2 and (param1.shape[0] == param1.shape[1]):
        return True
    elif isinstance(param1, np.ndarray) and isinstance(param2, np.ndarray) and param1.shape == param2.shape:
        return False
    else:
        raise ValueError("parameters for evaluation function not understood")


def vanilla_acc(param1, param2=None):
    if from_contingency_table(param1, param2):
        return _vanilla_acc_from_ct(param1)
    else:
        return accuracy_score(param1, param2)


def macrof1(param1, param2=None):
    if from_contingency_table(param1, param2):
        return _macro_f1_from_ct(param1)
    else:
        return f1_score(param1, param2, average="macro")


def _vanilla_acc_from_ct(cont_table):
    return np.diag(cont_table).sum() / cont_table.sum()


def _f1_bin(tp, fp, fn):
    if tp + fp + fn == 0:
        return 1
    else:
        return (2 * tp) / (2 * tp + fp + fn)


def _macro_f1_from_ct(cont_table):
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


ACCURACY_MEASURE = {vanilla_acc, macrof1}
ACCURACY_MEASURE_NAMES = {acc_fn.__name__ for acc_fn in ACCURACY_MEASURE}
ERROR_NAMES = ACCURACY_MEASURE_NAMES
