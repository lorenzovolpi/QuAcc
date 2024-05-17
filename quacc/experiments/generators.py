import os
from typing import Callable

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from quacc.dataset import RCV1_BINARY_DATASETS, RCV1_MULTICLASS_DATASETS
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import (
    CAPContingencyTable,
    ContTableTransferCAP,
    NaiveCAP,
    NsquaredEquationsCAP,
    QuAcc1xN2,
    QuAccNxN,
)
from quacc.models.direct import ATC, CAPDirect, DoC, PabloCAP, PrediQuant
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import get_results_path


def gen_classifiers():
    param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}

    yield "LR", LogisticRegression()
    yield "LR-opt", GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
    # yield 'NB', GaussianNB()
    # yield 'SVM(rbf)', SVC()
    # yield 'SVM(linear)', LinearSVC()


def gen_multi_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # yields the UCI multiclass datasets
    for dataset_name in [d for d in UCI_MULTICLASS_DATASETS if d not in ["wine-quality"]]:
        yield dataset_name, None if only_names else DP.uci_multiclass(dataset_name)

    # yields the 20 newsgroups dataset
    yield "20news", None if only_names else DP.news20()

    # yields the T1B@LeQua2022 (training) dataset
    yield "T1B-LeQua2022", None if only_names else DP.t1b_lequa2022()

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
    # imdb
    yield "imdb", None if only_names else DP.imdb()
    # rcv1
    for dn in RCV1_BINARY_DATASETS:
        dval = None if only_names else DP.rcv1_binary(dn)
        yield dn, dval


def gen_product(gen1, gen2):
    for v1 in gen1():
        for v2 in gen2():
            yield v1, v2


def gen_CAP_direct(h, acc_fn, with_oracle=False) -> [str, CAPDirect]:
    ### CAP methods ###
    # yield 'SebCAP', SebastianiCAP(h, acc_fn, ACC)
    # yield 'SebCAP-SLD', SebastianiCAP(h, acc_fn, EMQ, predict_train_prev=not with_oracle)
    # yield 'SebCAP-KDE', SebastianiCAP(h, acc_fn, KDEyML)
    # yield 'SebCAPweight', SebastianiCAP(h, acc_fn, ACC, alpha=0)
    yield "PrediQuant-ACC", PrediQuant(h, acc_fn, ACC)
    yield "PrediQuant-SLD", PrediQuant(h, acc_fn, EMQ)
    # yield "PrediQuant-KDE", PrediQuant(h, acc_fn, KDEyML)
    yield "PrediQuantWeight-ACC", PrediQuant(h, acc_fn, ACC, alpha=0)
    yield "PrediQuantWeight-SLD", PrediQuant(h, acc_fn, EMQ, alpha=0)
    # yield "PrediQuantWeight-KDE", PrediQuant(h, acc_fn, KDEyML, alpha=0)
    # yield 'PabCAP', PabloCAP(h, acc_fn, ACC)
    # yield 'PabCAP-SLD-median', PabloCAP(h, acc_fn, EMQ, aggr='median')

    ### baselines ###
    # yield "ATC-MC", ATC(h, acc_fn, scoring_fn="maxconf")
    # yield 'ATC-NE', ATC(h, acc_fn, scoring_fn='neg_entropy')
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])


# fmt: off
def gen_CAP_cont_table(h, acc_fn) -> [str, CAPContingencyTable]:
    # yield "Naive", NaiveCAP(h, acc_fn)
    # yield "CT-PPS-EMQ", ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'CT-PPS-KDE', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
    # yield 'CT-PPS-KDE05', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.05))
    # yield 'QuAcc(EMQ)nxn-noX', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_posteriors=True, add_X=False)
    # yield 'QuAcc(EMQ)nxn', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()))
    # yield "QuAcc(EMQ)nxn-MC", QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True)
    # yield 'QuAcc(EMQ)nxn-NE', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_negentropy=True)
    # yield 'QuAcc(EMQ)nxn-MIS', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxinfsoft=True)
    # yield 'QuAcc(EMQ)nxn-MC-MIS', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
    # yield 'QuAcc(EMQ)1xn2', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'QuAcc(EMQ)1xn2-MC', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True)
    # yield 'QuAcc(EMQ)1xn2-NE', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_negentropy=True)
    # yield 'QuAcc(EMQ)1xn2-MIS', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxinfsoft=True)
    # yield 'QuAcc(EMQ)1xn2-MC-MIS', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
    # yield 'CT-PPSh-EMQ', ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()), reuse_h=True)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, acc_fn, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, acc_fn, ACC)
    # yield 'Equations-SLD', NsquaredEquationsCAP(h, acc_fn, EMQ)
    return 
    yield
# fmt: on


# fmt: off
def gen_CAP_cont_table_opt(h, acc_fn, val_prot) -> [str, CAPContingencyTable]:
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

    def sld():
        return EMQ(LogisticRegression(), val_split=5)
    def pacc():
        return PACC(LogisticRegression())
    def kde():
        return KDEyML(LogisticRegression())

    yield "QuAcc(EMQ)1xn2-OPT-norefit", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
    yield "QuAcc(EMQ)nxn-OPT-norefit", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
    yield "QuAcc(EMQ)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(EMQ)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(PACC)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(PACC)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(PACC)1xn2-OPT-norefit", GSCAP(QuAcc1xN2(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=False)
    yield "QuAcc(PACC)nxn-OPT-norefit", GSCAP(QuAccNxN(h, acc_fn, pacc()), pacc_lr_params, val_prot, acc_fn, refit=False)
    yield "QuAcc(KDEy)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, kde()), kde_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(KDEy)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, kde()), kde_lr_params, val_prot, acc_fn, refit=True)
    yield "QuAcc(KDEy)1xn2-OPT-norefit", GSCAP(QuAcc1xN2(h, acc_fn, kde()), kde_lr_params, val_prot, acc_fn, refit=False)
    yield "QuAcc(KDEy)nxn-OPT-norefit", GSCAP(QuAccNxN(h, acc_fn, kde()), kde_lr_params, val_prot, acc_fn, refit=False)
    # return
    # yield
# fmt: on


def gen_methods(h, V, with_oracle=False):
    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_direct(h, acc_fn, with_oracle):
        yield name, method, V
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, V

    V, val_prot = split_validation(V)
    for name, method in gen_CAP_cont_table_opt(h, acc_fn, val_prot):
        yield name, method, V


def get_method_names():
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_val_prot = UPP(None)
    return (
        [m for m, _ in gen_CAP_direct(mock_h, mock_acc_fn)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
        + [m for m, _ in gen_CAP_cont_table_opt(mock_h, mock_acc_fn, mock_val_prot)]
    )


def gen_acc_measure(multiclass=False):
    yield "vanilla_accuracy", vanilla_acc
    yield "macro-F1", f1_macro if multiclass else f1


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
