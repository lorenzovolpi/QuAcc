from time import time

import numpy as np
from pytest import TestReport

from quacc.models.cont_table import CAPContingencyTable
from quacc.models.direct import CAPDirect
from quacc.models.model_selection import GridSearchAE


def fit_method(method, V):
    tinit = time()
    method.fit(V)
    t_train = time() - tinit
    return method, t_train


def get_intermediate_res(method, test_prot, oracle=False):
    if isinstance(method, CAPContingencyTable):
        tinit = time()
        if not oracle:
            estim_tables = [method.predict_ct(Ui.X) for Ui in test_prot()]
        else:
            estim_tables = [method.predict_ct(Ui.X, oracle_prev=Ui.prevalence()) for Ui in test_prot()]
        t_interm = (time() - tinit) / test_prot.total()

        return estim_tables, t_interm

    return None, 0


def get_predictions(method, estim_inter, test_prot, acc_fn, oracle=False):
    if isinstance(method, CAPDirect):
        tinit = time()
        if not oracle:
            estim_accs = [method.predict(Ui.X) for Ui in test_prot()]
        else:
            estim_accs = [method.predict(Ui.X, oracle_prev=Ui.prevalence()) for Ui in test_prot()]
        t_test_ave = (time() - tinit) / test_prot.total()
        return estim_accs, t_test_ave
    elif isinstance(method, CAPContingencyTable):
        estim_tables = estim_inter
        tinit = time()
        estim_accs = [acc_fn(cont_table) for cont_table in estim_tables]
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


def cache_method(report: TestReport, cache):
    if isinstance(report.method, CAPContingencyTable) and not isinstance(report.method, GridSearchAE):
        cache[report.method_name] = report
