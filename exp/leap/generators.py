import numpy as np
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import EMQ, DistributionMatchingY, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

from exp.leap.config import PROBLEM, _toggle
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models.cont_table import LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, DoC


def sld():
    emq = EMQ(LogisticRegression(), val_split=5)
    emq.SUPPRESS_WARNINGS = True
    return emq


def kdey():
    return KDEyML(LogisticRegression())


def kdey_auto():
    return KDEyML(LogisticRegression(), bandwidth="auto")


def dmy():
    return DistributionMatchingY(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "kNN", KNN(n_neighbors=10)
    yield "SVM", SVC(kernel="rbf", probability=True)
    yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=qp.environ["_R_SEED"])


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        for dn in _uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        for dataset_name in _uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def gen_acc_measure():
    multiclass = PROBLEM == "multiclass"
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        yield "macro-F1", f1_macro if multiclass else f1


def gen_baselines(acc_fn):
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")


def gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
    yield "DoC", DoC(acc_fn, V2_prot, V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
    yield "OCE(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP")
    # yield "OCE(KDEy)-SLSQP-c", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP-c")
    # yield "OCE(KDEy)-L-BFGS-B", OCE(acc_fn, kdey(), reuse_h=h, optim_method="L-BFGS-B")
    yield "LEAP(KDEy-a)", LEAP(acc_fn, kdey_auto(), reuse_h=h, log_true_solve=True)
    yield "PHD(KDEy-a)", PHD(acc_fn, kdey_auto(), reuse_h=h)
    yield "OCE(KDEy-a)-SLSQP", OCE(acc_fn, kdey_auto(), reuse_h=h, optim_method="SLSQP")
    if PROBLEM == "binary":
        yield "LEAP(DMy)", LEAP(acc_fn, dmy(), reuse_h=h, log_true_solve=True)
        yield "PHD(DMy)", PHD(acc_fn, dmy(), reuse_h=h)
        yield "OCE(DMy)-SLSQP", OCE(acc_fn, dmy(), reuse_h=h, optim_method="SLSQP")


def gen_methods(h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, V, V_posteriors
    for name, method in gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, V, V_posteriors


def get_method_names():
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_V2_prot = UPP(None)
    mock_V2_post = np.empty((1,))
    return (
        [m for m, _ in gen_baselines(mock_acc_fn)]
        + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
    )
