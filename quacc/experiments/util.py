from time import time

import numpy as np
from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix


def fit_method(method, V):
    tinit = time()
    method.fit(V)
    t_train = time() - tinit
    return method, t_train


def predictionsCAP(method, test_prot, oracle=False):
    tinit = time()
    if not oracle:
        estim_accs = [method.predict(Ui.X) for Ui in test_prot()]
    else:
        estim_accs = [
            method.predict(Ui.X, oracle_prev=Ui.prevalence()) for Ui in test_prot()
        ]
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, t_test_ave


def predictionsCAPcont_table(method, test_prot, gen_acc_measure, oracle=False):
    estim_accs_dict = {}
    tinit = time()
    if not oracle:
        estim_tables = [method.predict_ct(Ui.X) for Ui in test_prot()]
    else:
        estim_tables = [
            method.predict_ct(Ui.X, oracle_prev=Ui.prevalence()) for Ui in test_prot()
        ]
    for acc_name, acc_fn in gen_acc_measure():
        estim_accs_dict[acc_name] = [acc_fn(cont_table) for cont_table in estim_tables]
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs_dict, t_test_ave


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return tuple(prev[1:])
    else:
        return prev[-1]


def prevs_from_prot(prot):
    return [get_plain_prev(Ui.prevalence()) for Ui in prot()]


def true_acc(h: BaseEstimator, acc_fn: callable, U: LabelledCollection):
    y_pred = h.predict(U.X)
    y_true = U.y
    conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=U.classes_)
    return acc_fn(conf_table)


def get_acc_name(acc_name):
    return {
        "Vanilla Accuracy": "vanilla_accuracy",
        "Macro F1": "macro-F1",
    }
