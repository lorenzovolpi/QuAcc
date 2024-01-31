from functools import wraps
from statistics import mean

import numpy as np
import sklearn.metrics as metrics
from quapy.data import LabelledCollection
from quapy.protocol import APP, AbstractStochasticSeededProtocol
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

import baselines.atc as atc
import baselines.doc as doclib
import baselines.gde as gdelib
import baselines.impweight as iw
import baselines.mandoline as mandolib
import baselines.rca as rcalib
from baselines.utils import clone_fit
from quacc.environment import env

from .report import EvaluationReport

_baselines = {}


def baseline(func):
    @wraps(func)
    def wrapper(c_model, validation, protocol):
        return func(c_model, validation, protocol)

    wrapper.name = func.__name__
    _baselines[func.__name__] = wrapper

    return wrapper


@baseline
def kfcv(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    scoring = ["accuracy", "f1_macro"]
    scores = cross_validate(c_model, validation.X, validation.y, scoring=scoring)
    acc_score = mean(scores["test_accuracy"])
    f1_score = mean(scores["test_f1_macro"])

    report = EvaluationReport(name="kfcv")
    for test in protocol():
        test_preds = c_model_predict(test.X)
        meta_acc = abs(acc_score - metrics.accuracy_score(test.y, test_preds))
        meta_f1 = abs(
            f1_score - metrics.f1_score(test.y, test_preds, average=f1_average)
        )
        report.append_row(
            test.prevalence(),
            acc_score=acc_score,
            f1_score=f1_score,
            acc=meta_acc,
            f1=meta_f1,
        )

    return report


@baseline
def naive(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    val_preds = c_model_predict(validation.X)
    val_acc = metrics.accuracy_score(validation.y, val_preds)
    val_f1 = metrics.f1_score(validation.y, val_preds, average=f1_average)

    report = EvaluationReport(name="naive")
    for test in protocol():
        test_preds = c_model_predict(test.X)
        acc_score = metrics.accuracy_score(test.y, test_preds)
        f1_score = metrics.f1_score(test.y, test_preds, average=f1_average)
        meta_acc = abs(val_acc - acc_score)
        meta_f1 = abs(val_f1 - f1_score)
        report.append_row(
            test.prevalence(),
            acc_score=acc_score,
            f1_score=f1_score,
            acc=meta_acc,
            f1=meta_f1,
        )

    return report


@baseline
def ref(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
):
    c_model_predict = getattr(c_model, "predict")
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    report = EvaluationReport(name="ref")
    for test in protocol():
        test_preds = c_model_predict(test.X)
        report.append_row(
            test.prevalence(),
            acc_score=metrics.accuracy_score(test.y, test_preds),
            f1_score=metrics.f1_score(test.y, test_preds, average=f1_average),
        )

    return report


@baseline
def atc_mc(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    """garg"""
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = atc.get_max_conf(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)
    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)

    report = EvaluationReport(name="atc_mc")
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = atc.get_max_conf(test_probs)
        atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)
        meta_acc = abs(atc_accuracy - metrics.accuracy_score(test.y, test_preds))
        f1_score = atc.get_ATC_f1(
            atc_thres, test_scores, test_probs, average=f1_average
        )
        meta_f1 = abs(
            f1_score - metrics.f1_score(test.y, test_preds, average=f1_average)
        )
        report.append_row(
            test.prevalence(),
            acc=meta_acc,
            acc_score=atc_accuracy,
            f1_score=f1_score,
            f1=meta_f1,
        )

    return report


@baseline
def atc_ne(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    """garg"""
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = atc.get_entropy(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)
    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)

    report = EvaluationReport(name="atc_ne")
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = atc.get_entropy(test_probs)
        atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)
        meta_acc = abs(atc_accuracy - metrics.accuracy_score(test.y, test_preds))
        f1_score = atc.get_ATC_f1(
            atc_thres, test_scores, test_probs, average=f1_average
        )
        meta_f1 = abs(
            f1_score - metrics.f1_score(test.y, test_preds, average=f1_average)
        )
        report.append_row(
            test.prevalence(),
            acc=meta_acc,
            acc_score=atc_accuracy,
            f1_score=f1_score,
            f1=meta_f1,
        )

    return report


@baseline
def doc(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"

    val1, val2 = validation.split_stratified(train_prop=0.5, random_state=env._R_SEED)
    val1_probs = c_model_predict(val1.X)
    val1_mc = np.max(val1_probs, axis=-1)
    val1_preds = np.argmax(val1_probs, axis=-1)
    val1_acc = metrics.accuracy_score(val1.y, val1_preds)
    val1_f1 = metrics.f1_score(val1.y, val1_preds, average=f1_average)
    val2_protocol = APP(
        val2,
        n_prevalences=21,
        repeats=100,
        return_type="labelled_collection",
    )
    val2_prot_mc = []
    val2_prot_preds = []
    val2_prot_y = []
    for v2 in val2_protocol():
        _probs = c_model_predict(v2.X)
        _mc = np.max(_probs, axis=-1)
        _preds = np.argmax(_probs, axis=-1)
        val2_prot_mc.append(_mc)
        val2_prot_preds.append(_preds)
        val2_prot_y.append(v2.y)

    val_scores = np.array([doclib.get_doc(val1_mc, v2_mc) for v2_mc in val2_prot_mc])
    val_targets_acc = np.array(
        [
            val1_acc - metrics.accuracy_score(v2_y, v2_preds)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_acc = LinearRegression().fit(val_scores[:, np.newaxis], val_targets_acc)
    val_targets_f1 = np.array(
        [
            val1_f1 - metrics.f1_score(v2_y, v2_preds, average=f1_average)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_f1 = LinearRegression().fit(val_scores[:, np.newaxis], val_targets_f1)

    report = EvaluationReport(name="doc")
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_mc = np.max(test_probs, axis=-1)
        acc_score = (
            val1_acc
            - reg_acc.predict(np.array([[doclib.get_doc(val1_mc, test_mc)]]))[0]
        )
        f1_score = (
            val1_f1 - reg_f1.predict(np.array([[doclib.get_doc(val1_mc, test_mc)]]))[0]
        )
        meta_acc = abs(acc_score - metrics.accuracy_score(test.y, test_preds))
        meta_f1 = abs(
            f1_score - metrics.f1_score(test.y, test_preds, average=f1_average)
        )
        report.append_row(
            test.prevalence(),
            acc=meta_acc,
            acc_score=acc_score,
            f1=meta_f1,
            f1_score=f1_score,
        )

    return report


@baseline
def doc_feat(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    val_probs, val_labels = c_model_predict(validation.X), validation.y
    val_scores = np.max(val_probs, axis=-1)
    val_preds = np.argmax(val_probs, axis=-1)
    v1acc = np.mean(val_preds == val_labels) * 100

    report = EvaluationReport(name="doc_feat")
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = np.max(test_probs, axis=-1)
        score = (v1acc + doc.get_doc(val_scores, test_scores)) / 100.0
        meta_acc = abs(score - metrics.accuracy_score(test.y, test_preds))
        report.append_row(test.prevalence(), acc=meta_acc, acc_score=score)

    return report


@baseline
def rca(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    """elsahar19"""
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"
    val1, val2 = validation.split_stratified(train_prop=0.5, random_state=env._R_SEED)
    val1_pred1 = c_model_predict(val1.X)

    val2_protocol = APP(
        val2,
        n_prevalences=21,
        repeats=100,
        return_type="labelled_collection",
    )
    val2_prot_preds = []
    val2_rca = []
    val2_prot_preds = []
    val2_prot_y = []
    for v2 in val2_protocol():
        _preds = c_model_predict(v2.X)
        try:
            c_model2 = clone_fit(c_model, v2.X, _preds)
            c_model2_predict = getattr(c_model2, predict_method)
            val1_pred2 = c_model2_predict(val1.X)
            rca_score = 1.0 - rcalib.get_score(val1_pred1, val1_pred2, val1.y)
            val2_rca.append(rca_score)
            val2_prot_preds.append(_preds)
            val2_prot_y.append(v2.y)
        except ValueError:
            pass

    val_targets_acc = np.array(
        [
            metrics.accuracy_score(v2_y, v2_preds)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_acc = LinearRegression().fit(np.array(val2_rca)[:, np.newaxis], val_targets_acc)
    val_targets_f1 = np.array(
        [
            metrics.f1_score(v2_y, v2_preds, average=f1_average)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_f1 = LinearRegression().fit(np.array(val2_rca)[:, np.newaxis], val_targets_f1)

    report = EvaluationReport(name="rca")
    for test in protocol():
        try:
            test_preds = c_model_predict(test.X)
            c_model2 = clone_fit(c_model, test.X, test_preds)
            c_model2_predict = getattr(c_model2, predict_method)
            val1_pred2 = c_model2_predict(val1.X)
            rca_score = 1.0 - rcalib.get_score(val1_pred1, val1_pred2, val1.y)
            acc_score = reg_acc.predict(np.array([[rca_score]]))[0]
            f1_score = reg_f1.predict(np.array([[rca_score]]))[0]
            meta_acc = abs(acc_score - metrics.accuracy_score(test.y, test_preds))
            meta_f1 = abs(
                f1_score - metrics.f1_score(test.y, test_preds, average=f1_average)
            )
            report.append_row(
                test.prevalence(),
                acc=meta_acc,
                acc_score=acc_score,
                f1=meta_f1,
                f1_score=f1_score,
            )
        except ValueError:
            report.append_row(
                test.prevalence(),
                acc=np.nan,
                acc_score=np.nan,
                f1=np.nan,
                f1_score=np.nan,
            )

    return report


@baseline
def rca_star(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    """elsahar19"""
    c_model_predict = getattr(c_model, predict_method)
    f1_average = "binary" if validation.n_classes == 2 else "macro"
    validation1, val2 = validation.split_stratified(
        train_prop=0.5, random_state=env._R_SEED
    )
    val11, val12 = validation1.split_stratified(
        train_prop=0.5, random_state=env._R_SEED
    )

    val11_pred = c_model_predict(val11.X)
    c_model1 = clone_fit(c_model, val11.X, val11_pred)
    c_model1_predict = getattr(c_model1, predict_method)
    val12_pred1 = c_model1_predict(val12.X)

    val2_protocol = APP(
        val2,
        n_prevalences=21,
        repeats=100,
        return_type="labelled_collection",
    )
    val2_prot_preds = []
    val2_rca = []
    val2_prot_preds = []
    val2_prot_y = []
    for v2 in val2_protocol():
        _preds = c_model_predict(v2.X)
        try:
            c_model2 = clone_fit(c_model, v2.X, _preds)
            c_model2_predict = getattr(c_model2, predict_method)
            val12_pred2 = c_model2_predict(val12.X)
            rca_score = 1.0 - rcalib.get_score(val12_pred1, val12_pred2, val12.y)
            val2_rca.append(rca_score)
            val2_prot_preds.append(_preds)
            val2_prot_y.append(v2.y)
        except ValueError:
            pass

    val_targets_acc = np.array(
        [
            metrics.accuracy_score(v2_y, v2_preds)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_acc = LinearRegression().fit(np.array(val2_rca)[:, np.newaxis], val_targets_acc)
    val_targets_f1 = np.array(
        [
            metrics.f1_score(v2_y, v2_preds, average=f1_average)
            for v2_y, v2_preds in zip(val2_prot_y, val2_prot_preds)
        ]
    )
    reg_f1 = LinearRegression().fit(np.array(val2_rca)[:, np.newaxis], val_targets_f1)

    report = EvaluationReport(name="rca_star")
    for test in protocol():
        try:
            test_pred = c_model_predict(test.X)
            c_model2 = clone_fit(c_model, test.X, test_pred)
            c_model2_predict = getattr(c_model2, predict_method)
            val12_pred2 = c_model2_predict(val12.X)
            rca_star_score = 1.0 - rcalib.get_score(val12_pred1, val12_pred2, val12.y)
            acc_score = reg_acc.predict(np.array([[rca_star_score]]))[0]
            f1_score = reg_f1.predict(np.array([[rca_score]]))[0]
            meta_acc = abs(acc_score - metrics.accuracy_score(test.y, test_pred))
            meta_f1 = abs(
                f1_score - metrics.f1_score(test.y, test_pred, average=f1_average)
            )
            report.append_row(
                test.prevalence(),
                acc=meta_acc,
                acc_score=acc_score,
                f1=meta_f1,
                f1_score=f1_score,
            )
        except ValueError:
            report.append_row(
                test.prevalence(),
                acc=np.nan,
                acc_score=np.nan,
                f1=np.nan,
                f1_score=np.nan,
            )

    return report


@baseline
def gde(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
) -> EvaluationReport:
    c_model_predict = getattr(c_model, predict_method)
    val1, val2 = validation.split_stratified(train_prop=0.5, random_state=env._R_SEED)
    c_model1 = clone_fit(c_model, val1.X, val1.y)
    c_model1_predict = getattr(c_model1, predict_method)
    c_model2 = clone_fit(c_model, val2.X, val2.y)
    c_model2_predict = getattr(c_model2, predict_method)

    report = EvaluationReport(name="gde")
    for test in protocol():
        test_pred = c_model_predict(test.X)
        test_pred1 = c_model1_predict(test.X)
        test_pred2 = c_model2_predict(test.X)
        score = gdelib.get_score(test_pred1, test_pred2)
        meta_score = abs(score - metrics.accuracy_score(test.y, test_pred))
        report.append_row(test.prevalence(), acc=meta_score, acc_score=score)

    return report


@baseline
def mandoline(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
) -> EvaluationReport:
    c_model_predict = getattr(c_model, predict_method)

    val_probs = c_model_predict(validation.X)
    val_preds = np.argmax(val_probs, axis=1)
    D_val = mandolib.get_slices(val_probs)
    emprical_mat_list_val = (1.0 * (val_preds == validation.y))[:, np.newaxis]

    report = EvaluationReport(name="mandoline")
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_pred = np.argmax(test_probs, axis=1)
        D_test = mandolib.get_slices(test_probs)
        wp = mandolib.estimate_performance(D_val, D_test, None, emprical_mat_list_val)
        score = wp.all_estimates[0].weighted[0]
        meta_score = abs(score - metrics.accuracy_score(test.y, test_pred))
        report.append_row(test.prevalence(), acc=meta_score, acc_score=score)

    return report


@baseline
def logreg(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)

    val_preds = c_model_predict(validation.X)

    report = EvaluationReport(name="logreg")
    for test in protocol():
        wx = iw.logreg(validation.X, validation.y, test.X)
        test_preds = c_model_predict(test.X)
        estim_acc = iw.get_acc(val_preds, validation.y, wx)
        true_acc = metrics.accuracy_score(test.y, test_preds)
        meta_score = abs(estim_acc - true_acc)
        report.append_row(test.prevalence(), acc=meta_score, acc_score=estim_acc)

    return report


@baseline
def kdex2(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)

    val_preds = c_model_predict(validation.X)
    log_likelihood_val = iw.kdex2_lltr(validation.X)
    Xval = validation.X.toarray() if issparse(validation.X) else validation.X

    report = EvaluationReport(name="kdex2")
    for test in protocol():
        Xte = test.X.toarray() if issparse(test.X) else test.X
        wx = iw.kdex2_weights(Xval, Xte, log_likelihood_val)
        test_preds = c_model_predict(Xte)
        estim_acc = iw.get_acc(val_preds, validation.y, wx)
        true_acc = metrics.accuracy_score(test.y, test_preds)
        meta_score = abs(estim_acc - true_acc)
        report.append_row(test.prevalence(), acc=meta_score, acc_score=estim_acc)

    return report
