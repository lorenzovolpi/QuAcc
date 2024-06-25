import os
from collections import defaultdict

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import TWITTER_SENTIMENT_DATASETS_TEST, UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method._kdey import KDEyML
from quapy.method.aggregative import ACC, EMQ, PACC
from quapy.protocol import UPP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

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
from quacc.models.direct import ATC, CAPDirect, DoC, PabloCAP, PrediQuant, KFCV
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.models.regression import ReQua, reDAN
from quacc.utils.commons import get_results_path

_ACC = defaultdict(lambda: False)
_ACC = dict(
    N2E=True,
)
_SLD = defaultdict(lambda: False)
_SLD = dict(
    reDAN=False,
    PQ=False,
    ReQua=False,
    N2E=False,
    QuAcc=False,
)
_KDEy = defaultdict(lambda: False)
_KDEy = dict(
    reDAN=False,
    PQ=False,
    ReQua=False,
    N2E=True,
    QuAcc=False,
)


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
    # yield "LR-opt", GridSearchCV(LR(), param_grid, cv=5, n_jobs=qc.env["N_JOBS"])
    yield "KNN", KNN(n_neighbors=5)
    yield "KNN_10", KNN(n_neighbors=10)
    yield "SVM(rbf)", SVC(probability=True)
    yield "RFC", RFC()
    yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=0)
    # yield 'NB', GaussianNB()
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
    # for dn in RCV1_BINARY_DATASETS:
    #     dval = None if only_names else DP.rcv1_binary(dn)
    #     yield dn, dval
    # imdb
    # yield "imdb", None if only_names else DP.imdb()
    # UCI
    _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
    _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
    for dn in _uci_names:
        dval = None if only_names else DP.uci_binary(dn)
        yield dn, dval


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


### baselines ###
def gen_CAP_baselines(h, acc_fn, config, with_oracle=False) -> [str, CAPDirect]:
    yield "ATC-MC", ATC(h, acc_fn, scoring_fn="maxconf")
    # yield 'ATC-NE', ATC(h, acc_fn, scoring_fn='neg_entropy')
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])
    yield "KFCV", KFCV(h, acc_fn)


# fmt: off
def gen_CAP_direct(h, acc_fn, config, with_oracle=False) -> [str, CAPDirect]:
    redan_q_params= {
        "classifier__C": np.logspace(-3, 3, 7),
        "classifier__class_weight": [None, "balanced"],
    }
    rdan_q_params_sld = redan_q_params | {"recalib": [None, "bcts"]}
    rdan_q_params_kdey = redan_q_params | {"bandwidth": np.linspace(0.01, 0.2, 5)}
    ### CAP methods ###
    # yield 'SebCAP', SebastianiCAP(h, acc_fn, ACC)
    # yield 'SebCAPweight', SebastianiCAP(h, acc_fn, ACC, alpha=0)
    # yield "PrediQuant(ACC)", PrediQuant(h, acc_fn, ACC)
    # yield "PrediQuantWeight(ACC)", PrediQuant(h, acc_fn, ACC, alpha=0)
    # yield 'PabCAP', PabloCAP(h, acc_fn, ACC)
    if _SLD["PQ"]:
        # yield 'SebCAP-SLD', SebastianiCAP(h, acc_fn, EMQ, predict_train_prev=not with_oracle)
        # yield 'PabCAP-SLD-median', PabloCAP(h, acc_fn, EMQ, aggr='median')
        yield "PrediQuant(SLD-ae)", PrediQuant(h, acc_fn, EMQ)
        yield "PrediQuantWeight(SLD-ae)", PrediQuant(h, acc_fn, EMQ, alpha=0)
    if _SLD["ReQua"]:
        yield "ReQua(SLD-LinReg)", ReQua(*requa_params(h, acc_fn, LinReg(), sld(), config))
        yield "ReQua(SLD-LinReg)-conf", ReQua(*requa_params(h, acc_fn, LinReg(), sld(), config), add_conf=True)
        yield "ReQua(SLD-Ridge)", ReQua(*requa_params(h, acc_fn, Ridge(), sld(), config))
        yield "ReQua(SLD-Ridge)-conf", ReQua(*requa_params(h, acc_fn, Ridge(), sld(), config), add_conf=True)
        yield "ReQua(SLD-KRR)", ReQua(*requa_params(h, acc_fn, KRR(), sld(), config))
        yield "ReQua(SLD-KRR)-conf", ReQua(*requa_params(h, acc_fn, KRR(), sld(), config), add_conf=True)
    if _SLD["reDAN"]:
        # yield "reDAN(SLD-LinReg)", reDAN(h, acc_fn, LinReg(), sld(), sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(SLD-LinReg)-OPT", reDAN(h, acc_fn, LinReg(), sld(), add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(SLD-Ridge)", reDAN(h, acc_fn, Ridge(), sld(), sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(SLD-Ridge)-OPT", reDAN(h, acc_fn, Ridge(), sld(), add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        yield "reDAN(SLD-KRR)", reDAN(h, acc_fn, KRR(), sld(), q_params=rdan_q_params_sld, sample_size=qp.environ["SAMPLE_SIZE"])
        yield "reDAN(SLD-KRR)-OPT", reDAN(h, acc_fn, KRR(), sld(), q_params=rdan_q_params_sld, add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        yield "reDAN(SLD-KRR)-OPT+", reDAN(h, acc_fn, KRR(), sld(), q_params=rdan_q_params_sld, add_n2e_opt=True, add_conf=True, sample_size=qp.environ["SAMPLE_SIZE"])
    if _KDEy["PQ"]:
        # yield 'SebCAP-KDE', SebastianiCAP(h, acc_fn, KDEyML)
        yield "PrediQuant(KDEy-ae)", PrediQuant(h, acc_fn, KDEyML)
        yield "PrediQuantWeight(KDEy-ae)", PrediQuant(h, acc_fn, KDEyML, alpha=0)
    if _KDEy["ReQua"]:
        yield "ReQua(KDEy-LinReg)", ReQua(*requa_params(h, acc_fn, LinReg(), kdey(), config))
        yield "ReQua(KDEy-LinReg)-conf", ReQua(*requa_params(h, acc_fn, LinReg(), kdey(), config), add_conf=True)
        yield "ReQua(KDEy-Ridge)", ReQua(*requa_params(h, acc_fn, Ridge(), kdey(), config))
        yield "ReQua(KDEy-Ridge)-conf", ReQua(*requa_params(h, acc_fn, Ridge(), kdey(), config), add_conf=True)
        yield "ReQua(KDEy-KRR)", ReQua(*requa_params(h, acc_fn, KRR(), kdey(), config))
        yield "ReQua(KDEy-KRR)-conf", ReQua(*requa_params(h, acc_fn, KRR(), kdey(), config), add_conf=True)
    if _KDEy["reDAN"]:
        # yield "reDAN(KDEy-LinReg)", reDAN(h, acc_fn, LinReg(), kdey(), sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(KDEy-LinReg)-OPT", reDAN(h, acc_fn, LinReg(), kdey(), add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(KDEy-Ridge)", reDAN(h, acc_fn, Ridge(), kdey(), sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(KDEy-Ridge)-OPT", reDAN(h, acc_fn, Ridge(), kdey(), add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        yield "reDAN(KDEy-KRR)", reDAN(h, acc_fn, KRR(), kdey(), sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(KDEy-KRR)-OPT", reDAN(h, acc_fn, KRR(), kdey(), q_params = rdan_q_params_kdey, add_n2e_opt=True, sample_size=qp.environ["SAMPLE_SIZE"])
        # yield "reDAN(KDEy-KRR)-OPT+", reDAN(h, acc_fn, KRR(), kdey(), q_params = rdan_q_params_kdey, add_n2e_opt=True, add_conf=True, sample_size=qp.environ["SAMPLE_SIZE"])

def gen_CAP_cont_table(h, acc_fn, config) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(h, acc_fn)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, acc_fn, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, acc_fn, ACC)
    if _ACC["N2E"]:
        yield "N2E(ACC-h0)", N2E(h, acc_fn, ACC(LR()), reuse_h=True)
    if _SLD["N2E"]:
        yield "N2E(SLD-h0)", N2E(h, acc_fn, sld(), reuse_h=True)
        yield "N2E(SLD-h+)", N2E(h, acc_fn, sld(), reuse_h=False)
    if _KDEy["N2E"]:
        # yield 'CT-PPS-KDE', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
        # yield 'CT-PPS-KDE05', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.05))
        yield "N2E(KDEy-h0)", N2E(h, acc_fn, kdey(), reuse_h=True)
        yield "N2E(KDEy-h+)", N2E(h, acc_fn, kdey(), reuse_h=False)
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
    n2e_sld_h0_params = {"q_class__recalib": [None, "bcts"]}
    n2e_sld_hplus_params = n2e_sld_h0_params | {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
    }
    n2e_kde_h0_params = {"q_class__bandwidth": np.linspace(0.01, 0.2, 20)}
    n2e_kde_hplus_params = n2e_kde_h0_params | {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
    }

    if _SLD["QuAcc"]:
        yield "QuAcc(SLD)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
        yield "QuAcc(SLD)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
        if config == "binary":
            yield "QuAcc(SLD)1xnp1-OPT", GSCAP(QuAcc1xNp1(h, acc_fn, sld()), emq_lr_params, val_prot, acc_fn, refit=False)
    if _SLD["N2E"]:
        yield "N2E(SLD-h0)-OPT", GSCAP(N2E(h, acc_fn, sld(), reuse_h=True), n2e_sld_h0_params, val_prot, acc_fn, refit=False)
        yield "N2E(SLD-h+)-OPT", GSCAP(N2E(h, acc_fn, sld(), reuse_h=False), n2e_sld_hplus_params, val_prot, acc_fn, refit=False)
    if _KDEy["QuAcc"]:
        yield "QuAcc(KDEy)1xn2-OPT", GSCAP(QuAcc1xN2(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
        yield "QuAcc(KDEy)nxn-OPT", GSCAP(QuAccNxN(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
        if config == "binary":
            yield "QuAcc(KDEy)1xnp1-OPT", GSCAP(QuAcc1xNp1(h, acc_fn, kdey()), kde_lr_params, val_prot, acc_fn, refit=False)
    if _KDEy["N2E"]:
        yield "N2E(KDEy-h0)-OPT", GSCAP(N2E(h, acc_fn, kdey(), reuse_h=True), n2e_kde_h0_params, val_prot, acc_fn, refit=False)
        yield "N2E(KDEy-h+)-OPT", GSCAP(N2E(h, acc_fn, kdey(), reuse_h=False), n2e_kde_hplus_params, val_prot, acc_fn, refit=False)
    # return
    # yield
# fmt: on


def gen_methods(h, V, config, with_oracle=False):
    config = "multiclass" if config is None else config

    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_baselines(h, acc_fn, config, with_oracle):
        yield name, method, V
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
        [m for m, _ in gen_CAP_baselines(mock_h, mock_acc_fn, config)]
        + [m for m, _ in gen_CAP_direct(mock_h, mock_acc_fn, config)]
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
