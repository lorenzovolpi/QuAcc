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
from quacc.error import f1, vanilla_acc
from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.cont_table import (
    CAPContingencyTable,
    ContTableTransferCAP,
    NaiveCAP,
    QuAcc1xN2,
    QuAccNxN,
)
from quacc.models.direct import ATC, CAPDirect, DoC
from quacc.utils.commons import get_results_path


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
    _IMDB = [
        "imdb",
    ]
    _RCV1 = [
        "CCAT",
        "GCAT",
        "MCAT",
    ]
    for dn in _IMDB:
        dval = None if only_names else DP.imdb()
        yield dn, dval
    for dn in _RCV1:
        dval = None if only_names else DP.rcv1(dn)
        yield dn, dval


def gen_product(gen1, gen2):
    for v1 in gen1():
        for v2 in gen2():
            yield v1, v2


def gen_CAP(h, acc_fn, with_oracle=False) -> [str, CAPDirect]:
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


# fmt: off
def gen_CAP_cont_table(h) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(h)
    # yield "CT-PPS-EMQ", ContTableTransferCAP(h, EMQ(LogisticRegression()))
    # yield 'CT-PPS-KDE', ContTableTransferCAP(h, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
    # yield 'CT-PPS-KDE05', ContTableTransferCAP(h, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.05))
    # yield 'QuAcc(EMQ)nxn-noX', QuAccNxN(h, EMQ(LogisticRegression()), add_posteriors=True, add_X=False)
    # yield 'QuAcc(EMQ)nxn', QuAccNxN(h, EMQ(LogisticRegression()))
    yield "QuAcc(EMQ)nxn-MC", QuAccNxN(h, EMQ(LogisticRegression()), add_maxconf=True)
    # yield 'QuAcc(EMQ)nxn-NE', QuAccNxN(h, EMQ(LogisticRegression()), add_negentropy=True)
    yield 'QuAcc(EMQ)nxn-MIS', QuAccNxN(h, EMQ(LogisticRegression()), add_maxinfsoft=True)
    yield 'QuAcc(EMQ)nxn-MC-MIS', QuAccNxN(h, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
    # yield 'QuAcc(EMQ)1xn2', QuAcc1xN2(h, EMQ(LogisticRegression()))
    yield 'QuAcc(EMQ)1xn2-MC', QuAcc1xN2(h, EMQ(LogisticRegression()), add_maxconf=True)
    # yield 'QuAcc(EMQ)1xn2-NE', QuAcc1xN2(h, EMQ(LogisticRegression()), add_negentropy=True)
    yield 'QuAcc(EMQ)1xn2-MIS', QuAcc1xN2(h, EMQ(LogisticRegression()), add_maxinfsoft=True)
    yield 'QuAcc(EMQ)1xn2-MC-MIS', QuAcc1xN2(h, EMQ(LogisticRegression()), add_maxconf=True, add_maxinfsoft=True)
    # yield 'CT-PPSh-EMQ', ContTableTransferCAP(h, EMQ(LogisticRegression()), reuse_h=True)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, ACC)
    # yield 'Equations-SLD', NsquaredEquationsCAP(h, EMQ)
# fmt: on


def gen_CAP_cont_table_opt(h, acc_fn) -> [str, CAPContingencyTable]:
    return
    yield


def gen_methods(h, acc_fn, with_oracle=False):
    for name, method in gen_CAP(h, acc_fn, with_oracle):
        yield name, method
    for name, method in gen_CAP_cont_table(h):
        yield name, method
    for name, method in gen_CAP_cont_table_opt(h, acc_fn):
        yield name, method


def get_method_names():
    mock_h = LogisticRegression()
    return [m for m, _ in gen_CAP(mock_h, None)] + [m for m, _ in gen_CAP_cont_table(mock_h)]


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc
    yield "macro-F1", f1


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
