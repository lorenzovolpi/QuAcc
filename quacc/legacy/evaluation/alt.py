from functools import wraps

import numpy as np
import quapy.functional as F
import sklearn.metrics as metrics
from quapy.method.aggregative import ACC, EMQ
from sklearn import clone
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.legacy.evaluation.report import EvaluationReport

_alts = {}


def alt(func):
    @wraps(func)
    def wrapper(c_model, validation, protocol):
        return func(c_model, validation, protocol)

    wrapper.name = func.__name__
    _alts[func.__name__] = wrapper

    return wrapper


@alt
def cross(c_model, validation, protocol):
    y_val = validation.labels
    y_hat_val = c_model.predict(validation.instances)

    qcls = clone(c_model)
    qcls.fit(*validation.Xy)

    er = EvaluationReport(name="cross")
    for sample in protocol():
        y_hat = c_model.predict(sample.instances)
        y = sample.labels
        ground_acc = (y_hat == y).mean()
        ground_f1 = metrics.f1_score(y, y_hat, zero_division=0)

        q = EMQ(qcls)
        q.fit(validation, fit_classifier=False)

        M_hat = ACC.getPteCondEstim(validation.classes_, y_val, y_hat_val)
        p_hat = q.quantify(sample.instances)
        cont_table_hat = p_hat * M_hat

        acc_score = qc.error.acc(cont_table_hat)
        f1_score = qc.error.f1(cont_table_hat)

        meta_acc = abs(acc_score - ground_acc)
        meta_f1 = abs(f1_score - ground_f1)
        er.append_row(
            sample.prevalence(),
            acc=meta_acc,
            f1=meta_f1,
            acc_score=acc_score,
            f1_score=f1_score,
        )

    return er


@alt
def cross2(c_model, validation, protocol):
    classes = validation.classes_
    y_val = validation.labels
    y_hat_val = c_model.predict(validation.instances)
    M_hat = ACC.getPteCondEstim(classes, y_val, y_hat_val)
    pos_prev_val = validation.prevalence()[1]

    er = EvaluationReport(name="cross2")
    for sample in protocol():
        y_test = sample.labels
        y_hat_test = c_model.predict(sample.instances)
        ground_acc = (y_hat_test == y_test).mean()
        ground_f1 = metrics.f1_score(y_test, y_hat_test, zero_division=0)
        pos_prev_cc = F.prevalence_from_labels(y_hat_test, classes)[1]
        tpr_hat = M_hat[1, 1]
        fpr_hat = M_hat[1, 0]
        tnr_hat = M_hat[0, 0]
        pos_prev_test_hat = (pos_prev_cc - fpr_hat) / (tpr_hat - fpr_hat)
        pos_prev_test_hat = np.clip(pos_prev_test_hat, 0, 1)

        if pos_prev_val > 0.5:
            # in this case, the tpr might be a more reliable estimate than tnr
            A = np.asarray(
                [[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, tpr_hat, 0, tpr_hat - 1]]
            )
        else:
            # in this case, the tnr might be a more reliable estimate than tpr
            A = np.asarray(
                [[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [tnr_hat - 1, 0, tnr_hat, 0]]
            )

        b = np.asarray([pos_prev_cc, pos_prev_test_hat, 1, 0])

        tn, fn, fp, tp = np.linalg.solve(A, b)
        cont_table_hat = np.array([[tn, fp], [fn, tp]])

        acc_score = qc.error.acc(cont_table_hat)
        f1_score = qc.error.f1(cont_table_hat)

        meta_acc = abs(acc_score - ground_acc)
        meta_f1 = abs(f1_score - ground_f1)
        er.append_row(
            sample.prevalence(),
            acc=meta_acc,
            f1=meta_f1,
            acc_score=acc_score,
            f1_score=f1_score,
        )

    return er
