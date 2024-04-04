import os

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import (
    TWITTER_SENTIMENT_DATASETS_TEST,
    UCI_MULTICLASS_DATASETS,
)
from quapy.method.aggregative import EMQ
from sklearn.linear_model import LogisticRegression

from quacc.dataset import DatasetProvider as DP
from quacc.error import macrof1_fn, vanilla_acc_fn
from quacc.experiments.util import getpath
from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.baselines import ATC, DoC
from quacc.models.cont_table import CAPContingencyTable, ContTableTransferCAP, NaiveCAP


def gen_classifiers():
    param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}

    yield "LR", LogisticRegression()
    # yield 'LR-opt', GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
    # yield 'NB', GaussianNB()
    # yield 'SVM(rbf)', SVC()
    # yield 'SVM(linear)', LinearSVC()


def gen_multi_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    for dataset_name in np.setdiff1d(UCI_MULTICLASS_DATASETS, ["wine-quality"]):
        if only_names:
            yield dataset_name, None
        else:
            yield dataset_name, DP.uci_multiclass(dataset_name)

    # yields the 20 newsgroups dataset
    if only_names:
        yield "20news", None
    else:
        yield "20news", DP.news20()

    # yields the T1B@LeQua2022 (training) dataset
    if only_names:
        yield "T1B-LeQua2022", None
    else:
        yield "T1B-LeQua2022", DP.t1b_lequa2022()


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
    if only_names:
        for dataset_name in ["imdb", "CCAT", "GCAT", "MCAT"]:
            yield dataset_name, None
    else:
        yield "imdb", DP.imdb()
        for rcv1_name in [
            "CCAT",
            "GCAT",
            "MCAT",
        ]:
            yield rcv1_name, DP.rcv1(rcv1_name)


def gen_CAP(h, acc_fn, with_oracle=False) -> [str, ClassifierAccuracyPrediction]:
    ### CAP methods ###
    # yield 'SebCAP', SebastianiCAP(h, acc_fn, ACC)
    # yield 'SebCAP-SLD', SebastianiCAP(h, acc_fn, EMQ, predict_train_prev=not with_oracle)
    # yield 'SebCAP-KDE', SebastianiCAP(h, acc_fn, KDEyML)
    # yield 'SebCAPweight', SebastianiCAP(h, acc_fn, ACC, alpha=0)
    # yield 'PabCAP', PabloCAP(h, acc_fn, ACC)
    # yield 'PabCAP-SLD-median', PabloCAP(h, acc_fn, EMQ, aggr='median')

    ### baselines ###
    yield "ATC-MC", ATC(h, acc_fn, scoring_fn="maxconf")
    # yield 'ATC-NE', ATC(h, acc_fn, scoring_fn='neg_entropy')
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])


def gen_CAP_cont_table(h) -> [str, CAPContingencyTable]:
    acc_fn = None
    yield "Naive", NaiveCAP(h, acc_fn)
    yield "CT-PPS-EMQ", ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'CT-PPS-KDE', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
    # yield 'CT-PPS-KDE05', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.05))
    # yield 'QuAcc(EMQ)nxn-noX', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_posteriors=True, add_X=False)
    # yield 'QuAcc(EMQ)nxn', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'QuAcc(EMQ)nxn-MC', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxconf=True)
    # yield 'QuAcc(EMQ)nxn-NE', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_negentropy=True)
    # yield 'QuAcc(EMQ)nxn-MIS', QuAccNxN(h, acc_fn, EMQ(LogisticRegression()), add_maxinfsoft=True)
    # yield 'QuAcc(EMQ)1xn2', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'QuAcc(EMQ)1xn2', QuAcc1xN2(h, acc_fn, EMQ(LogisticRegression()))
    # yield 'CT-PPSh-EMQ', ContTableTransferCAP(h, acc_fn, EMQ(LogisticRegression()), reuse_h=True)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, acc_fn, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, acc_fn, ACC)
    # yield 'Equations-SLD', NsquaredEquationsCAP(h, acc_fn, EMQ)


def get_method_names():
    mock_h = LogisticRegression()
    return [m for m, _ in gen_CAP(mock_h, None)] + [
        m for m, _ in gen_CAP_cont_table(mock_h)
    ]


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc_fn
    yield "macro-F1", macrof1_fn


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(
            getpath(basedir, cls_name, acc_name, dataset_name, method_name)
        ):
            return True
    return False
