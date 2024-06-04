import os

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import (
    TWITTER_SENTIMENT_DATASETS_TEST,
    UCI_MULTICLASS_DATASETS,
)
from quapy.method._kdey import KDEyML
from quapy.method.aggregative import ACC, EMQ, PACC
from quapy.protocol import UPP
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import quacc as qc
from quacc.dataset import RCV1_BINARY_DATASETS, RCV1_MULTICLASS_DATASETS
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import (
    N2E,
    CAPContingencyTable,
    NaiveCAP,
    QuAcc1xN2,
    QuAcc1xNp1,
    QuAccNxN,
)
from quacc.models.direct import ATC, CAPDirect, DoC, PabloCAP, PrediQuant
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.models.requa import ReQua
from quacc.utils.commons import get_results_path

SLD = True
KDEy = False

MAE = True
MSE = True

VANILLA = True
F1 = False


def sld():
    return EMQ(LR(), val_split=5)


def pacc():
    return PACC(LR())


def kdey():
    return KDEyML(LR())


def gen_classifiers():
    param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}

    yield "LR", LR()
    yield "LR-opt", GridSearchCV(LR(), param_grid, cv=5, n_jobs=qc.env["N_JOBS"])
    # yield 'NB', GaussianNB()
    # yield 'SVM(rbf)', SVC()
    # yield 'SVM(linear)', LinearSVC()


def gen_multi_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # yields the UCI multiclass datasets
    for dataset_name in [d for d in UCI_MULTICLASS_DATASETS if d not in ["wine-quality", "letter"]]:
        yield dataset_name, None if only_names else DP.uci_multiclass(dataset_name)

    # yields the 20 newsgroups dataset
    # yield "20news", None if only_names else DP.news20()

    # yields the T1B@LeQua2022 (training) dataset
    # yield "T1B-LeQua2022", None if only_names else DP.t1b_lequa2022()

    # yields the RCV1 multiclass datasets
    for dataset_name in RCV1_MULTICLASS_DATASETS:
        yield dataset_name, None if only_names else DP.rcv1_multiclass(dataset_name)


def gen_tweet_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    for dataset_name in TWITTER_SENTIMENT_DATASETS_TEST:
        if only_names:
            yield dataset_name, None
        else:
            yield dataset_name, DP.twitter(dataset_name)


def gen_bin_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # rcv1
    for dn in RCV1_BINARY_DATASETS:
        dval = None if only_names else DP.rcv1_binary(dn)
        yield dn, dval
    # imdb
    yield "imdb", None if only_names else DP.imdb()


def gen_product(gen1, gen2):
    for v1 in gen1():
        for v2 in gen2():
            yield v1, v2


def requa_params(h, acc_fn, reg, q_class, config):
    quaccs = [
        QuAcc1xN2(h, acc_fn, q_class),
        QuAccNxN(h, acc_fn, q_class),
    ]
    if config == "binary":
        quaccs.append(QuAcc1xNp1(h, acc_fn, q_class))

    quacc_params = {
        # "q_class__classifier__C": np.logspace(-3, 3, 7),
        # "q_class__classifier__class_weight": [None, "balanced"],
        "add_X": [True, False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }

    sample_size = qp.environ["SAMPLE_SIZE"]

    return h, acc_fn, reg, quaccs, quacc_params, sample_size


def gen_CAP_direct(h, acc_fn, config, with_oracle=False) -> [str, CAPDirect]:
    ### CAP methods ###
    # yield 'SebCAP', SebastianiCAP(h, acc_fn, ACC)
    # yield 'SebCAPweight', SebastianiCAP(h, acc_fn, ACC, alpha=0)
    # yield "PrediQuant(ACC)", PrediQuant(h, acc_fn, ACC)
    # yield "PrediQuantWeight(ACC)", PrediQuant(h, acc_fn, ACC, alpha=0)
    # yield 'PabCAP', PabloCAP(h, acc_fn, ACC)
    if SLD:
        # yield 'SebCAP-SLD', SebastianiCAP(h, acc_fn, EMQ, predict_train_prev=not with_oracle)
        # yield 'PabCAP-SLD-median', PabloCAP(h, acc_fn, EMQ, aggr='median')
        if MAE:
            yield "PrediQuant(SLD-ae)", PrediQuant(h, acc_fn, EMQ)
            yield "PrediQuantWeight(SLD-ae)", PrediQuant(h, acc_fn, EMQ, alpha=0)
        if MSE:
            yield "PrediQuant(SLD-se)", PrediQuant(h, acc_fn, EMQ, error=qc.error.se)
            yield "PrediQuantWeight(SLD-se)", PrediQuant(h, acc_fn, EMQ, alpha=0, error=qc.error.se)
        yield "ReQua(SLD-LinReg)", ReQua(*requa_params(h, acc_fn, LinReg(), sld(), config))
        yield "ReQua(SLD-LinReg)-conf", ReQua(*requa_params(h, acc_fn, LinReg(), sld(), config), add_conf=True)
        yield "ReQua(SLD-Ridge)", ReQua(*requa_params(h, acc_fn, Ridge(), sld(), config))
        yield "ReQua(SLD-Ridge)-conf", ReQua(*requa_params(h, acc_fn, Ridge(), sld(), config), add_conf=True)
        yield "ReQua(SLD-KRR)", ReQua(*requa_params(h, acc_fn, KRR(), sld(), config))
        yield "ReQua(SLD-KRR)-conf", ReQua(*requa_params(h, acc_fn, KRR(), sld(), config), add_conf=True)
    if KDEy:
        # yield 'SebCAP-KDE', SebastianiCAP(h, acc_fn, KDEyML)
        if MAE:
            yield "PrediQuant(KDEy-ae)", PrediQuant(h, acc_fn, KDEyML)
            yield "PrediQuantWeight(KDEy-ae)", PrediQuant(h, acc_fn, KDEyML, alpha=0)
        if MSE:
            yield "PrediQuant(KDEy-se)", PrediQuant(h, acc_fn, KDEyML, error=qc.error.se)
            yield "PrediQuantWeight(KDEy-se)", PrediQuant(h, acc_fn, KDEyML, alpha=0, error=qc.error.se)
        yield "ReQua(KDEy-LinReg)", ReQua(*requa_params(h, acc_fn, LinReg(), kdey(), config))
        yield "ReQua(KDEy-LinReg)-conf", ReQua(*requa_params(h, acc_fn, LinReg(), kdey(), config), add_conf=True)
        yield "ReQua(KDEy-Ridge)", ReQua(*requa_params(h, acc_fn, Ridge(), kdey(), config))
        yield "ReQua(KDEy-Ridge)-conf", ReQua(*requa_params(h, acc_fn, Ridge(), kdey(), config), add_conf=True)
        yield "ReQua(KDEy-KRR)", ReQua(*requa_params(h, acc_fn, KRR(), kdey(), config))
        yield "ReQua(KDEy-KRR)-conf", ReQua(*requa_params(h, acc_fn, KRR(), kdey(), config), add_conf=True)

    ### baselines ###
    yield "ATC-MC", ATC(h, acc_fn, scoring_fn="maxconf")
    # yield 'ATC-NE', ATC(h, acc_fn, scoring_fn='neg_entropy')
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])


# fmt: off
def gen_CAP_cont_table(h, acc_fn, config) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(h, acc_fn)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, acc_fn, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, acc_fn, ACC)
    if SLD:
        # yield "CT-PPS-SLD", ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()))
        # yield 'QuAcc(SLD)nxn-noX', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_posteriors=True, add_X=False)
        # yield 'QuAcc(SLD)nxn', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()))
        # yield "QuAcc(SLD)nxn-MC", QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True)
        # yield 'QuAcc(SLD)nxn-NE', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_negentropy=True)
        # yield 'QuAcc(SLD)nxn-MIS', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxinfsoft=True)
        # yield 'QuAcc(SLD)nxn-MC-MIS', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
        # yield 'QuAcc(SLD)1xn2', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()))
        # yield 'QuAcc(SLD)1xn2-MC', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True)
        # yield 'QuAcc(SLD)1xn2-NE', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_negentropy=True)
        # yield 'QuAcc(SLD)1xn2-MIS', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxinfsoft=True)
        # yield 'QuAcc(SLD)1xn2-MC-MIS', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
        # yield 'CT-PPSh-SLD', ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()), reuse_h=True)
        # yield 'Equations-SLD', NsquaredEquationsCAP(h, acc_fn, EMQ)
        yield "N2E(SLD)", N2E(h, acc_fn, sld())
    if KDEy:
        # yield 'CT-PPS-KDE', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
        # yield 'CT-PPS-KDE05', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.05))
        yield "N2E(KDEy)", N2E(h, acc_fn, kdey())
# fmt: on


# fmt: off
def gen_CAP_cont_table_opt(h, acc_fn, config, val_prot) -> [str, CAPContingencyTable]:
    pacc_lr_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        # "add_X": [True, False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }
    emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
    kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}

    # yield "QuAcc(PACC)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=True)
    # yield "QuAcc(PACC)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=True)
    # yield "QuAcc(PACC)1xn2-OPT-norefit", GSCAP(QuAcc1xN2(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=False)
    # yield "QuAcc(PACC)nxn-OPT-norefit", GSCAP(QuAccNxN(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=False)
    if SLD and MAE:
        yield "QuAcc(SLD)1xn2-OPT(mae)-norefit", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
        yield "QuAcc(SLD)1xn2-OPT(mae)", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=True)
        yield "QuAcc(SLD)nxn-OPT(mae)-norefit", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
        yield "QuAcc(SLD)nxn-OPT(mae)", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=True)
        if config == "binary":
            yield "QuAcc(SLD)1xnp1-OPT(mae)-norefit", GSCAP(QuAcc1xNp1(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
            yield "QuAcc(SLD)1xnp1-OPT(mae)", GSCAP(QuAcc1xNp1(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=True)
    if SLD and MSE:
        yield "QuAcc(SLD)1xn2-OPT(mse)-norefit", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
        yield "QuAcc(SLD)1xn2-OPT(mse)", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
        yield "QuAcc(SLD)nxn-OPT(mse)-norefit", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
        yield "QuAcc(SLD)nxn-OPT(mse)", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
        if config == "binary":
            yield "QuAcc(SLD)1xnp1-OPT(mse)-norefit", GSCAP(QuAcc1xNp1(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
            yield "QuAcc(SLD)1xnp1-OPT(mse)", GSCAP(QuAcc1xNp1(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
    if KDEy and MAE:
        yield "QuAcc(KDEy)1xn2-OPT(mae)", GSCAP(QuAcc1xN2(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=True)
        yield "QuAcc(KDEy)1xn2-OPT(mae)-norefit", GSCAP(QuAcc1xN2(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
        yield "QuAcc(KDEy)nxn-OPT(mae)", GSCAP(QuAccNxN(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=True)
        yield "QuAcc(KDEy)nxn-OPT(mae)-norefit", GSCAP(QuAccNxN(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
        if config == "binary":
            yield "QuAcc(KDEy)1xnp1-OPT(mae)-norefit", GSCAP(QuAcc1xNp1(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
            yield "QuAcc(KDEy)1xnp1-OPT(mae)", GSCAP(QuAcc1xNp1(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=True)
    if KDEy and MSE:
        yield "QuAcc(KDEy)1xn2-OPT(mse)", GSCAP(QuAcc1xN2(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
        yield "QuAcc(KDEy)1xn2-OPT(mse)-norefit", GSCAP(QuAcc1xN2(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
        yield "QuAcc(KDEy)nxn-OPT(mse)", GSCAP(QuAccNxN(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
        yield "QuAcc(KDEy)nxn-OPT(mse)-norefit", GSCAP(QuAccNxN(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
        if config == "binary":
            yield "QuAcc(KDEy)1xnp1-OPT-norefit", GSCAP(QuAcc1xNp1(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=False)
            yield "QuAcc(KDEy)1xnp1-OPT", GSCAP(QuAcc1xNp1(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, error=qc.error.mse, refit=True)
    # return
    # yield
# fmt: on


def gen_methods(h, V, config, with_oracle=False):
    config = "multiclass" if config is None else config

    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_direct(h, acc_fn, config, with_oracle):
        yield name, method, V
    for name, method in gen_CAP_cont_table(h, acc_fn, config):
        yield name, method, V

    V, val_prot = split_validation(V)
    for name, method in gen_CAP_cont_table_opt(h, acc_fn, config, val_prot):
        yield name, method, V


def get_method_names(config):
    mock_h = LR()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_val_prot = UPP(None)
    return (
        [m for m, _ in gen_CAP_direct(mock_h, mock_acc_fn, config)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn, config)]
        + [m for m, _ in gen_CAP_cont_table_opt(mock_h, mock_acc_fn, config, mock_val_prot)]
    )


def gen_acc_measure(multiclass=False):
    if VANILLA:
        yield "vanilla_accuracy", vanilla_acc
    if F1:
        yield "macro-F1", f1_macro if multiclass else f1


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
