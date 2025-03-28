import json
import os
from dataclasses import dataclass

import numpy as np
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import ACC, CC, EMQ, DistributionMatchingY, KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import quacc as qc
from exp.util import split_validation
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models._large_models import BaseEstimatorAdapter
from quacc.models.cont_table import LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, DoC
from quacc.models.utils import OracleQuantifier
from quacc.utils.commons import contingency_table

PROJECT = "leap"
root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

PROBLEM = "multiclass"

_toggle = {
    "vanilla": True,
    "f1": False,
}


@dataclass
class DatasetBundle:
    L_prevalence: np.ndarray
    V: LabelledCollection
    U: LabelledCollection
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None

    def create_bundle(self, h: BaseEstimator):
        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V, repeats=500)

        # precomumpute model posteriors for validation sets
        self.V_posteriors = h.predict_proba(self.V.X)
        self.V1_posteriors = h.predict_proba(self.V1.X)
        self.V2_prot_posteriors = []
        for sample in self.V2_prot():
            self.V2_prot_posteriors.append(h.predict_proba(sample.X))

        # precomumpute model posteriors for test samples
        self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
        for sample in self.test_prot():
            P = h.predict_proba(sample.X)
            self.test_prot_posteriors.append(P)
            y_hat = np.argmax(P, axis=-1)
            self.test_prot_y_hat.append(y_hat)
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self

    @classmethod
    def mock(cls):
        return DatasetBundle(None, None, None, test_prot=lambda: [])
        # mock_y = np.array([0, 1])
        # mock_X = mock_post = mock_cts = np.eye(2)
        # mock_lc = LabelledCollection(mock_X, mock_y)
        # mock_prot = UPP(mock_lc, repeats=1, sample_size=1, return_type="labelled_collection")
        # D = DatasetBundle(mock_y, mock_lc, mock_lc)
        # D.V1 = mock_lc
        # D.V2_prot = D.test_prot = mock_prot
        # D.test_prot = mock_prot
        # D.V1_posteriors = mock_post
        # D.V2_prot_posteriors = D.test_prot_posteriors = [mock_post]
        # D.test_prot_y_hat = [mock_y]
        # D.test_prot_true_cts = [mock_cts]


def sample_size(test_size):
    return 100
    # if test_size > 3000:
    #     return 500
    # elif test_size > 1000:
    #     return 300
    # elif test_size > 400:
    #     return 200
    # else:
    #     return 100


def cc():
    return CC(LogisticRegression())


def acc():
    return ACC(LogisticRegression())


def sld():
    emq = EMQ(LogisticRegression(), val_split=5)
    emq.SUPPRESS_WARNINGS = True
    return emq


def kdey():
    return KDEyML(LogisticRegression())


def kdey_auto():
    return KDEyML(LogisticRegression(), bandwidth="auto")


def kdey_mlp():
    return KDEyML(MLP())


def dmy():
    return DistributionMatchingY(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "kNN", KNN(n_neighbors=10)
    yield "SVM", SVC(kernel="rbf", probability=True)
    yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=qp.environ["_R_SEED"])


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_transformer_model_dataset(only_dataset_names=False, only_model_names=False):
    dataset_model = [
        ("imdb", "bert-base-uncased"),
    ]

    if only_dataset_names:
        return [d for d, _ in dataset_model]
    if only_model_names:
        return [m for _, m in dataset_model]

    for dataset_name, model_name in dataset_model:
        parent_dir = os.path.join(qc.env["OUT_DIR"], "trainsformers", "embeds", dataset_name, model_name)

        V_X = torch.load(os.path.join(parent_dir, "hidden_states.validation.pt")).numpy()
        V_logits = torch.load(os.path.join(parent_dir, "logits.validation.pt")).numpy()
        V_labels = torch.load(os.path.join(parent_dir, "labels.validation.pt")).numpy()
        U_X = torch.load(os.path.join(parent_dir, "hidden_states.test.pt")).numpy()
        U_logits = torch.load(os.path.join(parent_dir, "logits.test.pt")).numpy()
        U_labels = torch.load(os.path.join(parent_dir, "labels.test.pt")).numpy()

        V = LabelledCollection(V_X, V_labels, classes=np.unique(V_labels))
        U = LabelledCollection(U_X, U_labels, classes=np.unique(U_labels))

        model = BaseEstimatorAdapter(V_X, U_X, V_logits, U_logits)

        with open(os.path.join(parent_dir, "dataset_info.json")) as f:
            dataset_info = json.load(f)
            L_prev = np.array(dataset_info["L_prev"])

        yield (model_name, model), (dataset_name, (V, U), L_prev)


def gen_acc_measure():
    multiclass = PROBLEM == "multiclass"
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        yield "macro-F1", f1_macro if multiclass else f1


def gen_baselines(acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")
    # yield "QuAccNxN(KDEy)", QuAccNxN(acc_fn, kdey(), add_X=True, add_posteriors=True, add_maxinfsoft=True)
    # yield "QuAccNxN(KDEy-a)", QuAccNxN(acc_fn, kdey_auto(), add_X=True, add_posteriors=True, add_maxinfsoft=True)


def gen_baselines_vp(acc_fn, D):
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "LEAP(CC)", LEAP(acc_fn, cc(), reuse_h=h, log_true_solve=True)
    yield "LEAP(ACC)", LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
    yield "PHD(CC)", PHD(acc_fn, cc(), reuse_h=h)
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
    yield "OCE(CC)-SLSQP", OCE(acc_fn, cc(), reuse_h=h, optim_method="SLSQP")
    yield "OCE(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP")
    # yield "OCE(KDEy-MLP)-SLSQP", OCE(acc_fn, kdey_mlp(), optim_method="SLSQP")
    # yield "OCE(KDEy)-SLSQP-c", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP-c")
    # yield "OCE(KDEy)-L-BFGS-B", OCE(acc_fn, kdey(), reuse_h=h, optim_method="L-BFGS-B")
    # yield "LEAP(KDEy-a)", LEAP(acc_fn, kdey_auto(), reuse_h=h, log_true_solve=True)
    # yield "PHD(KDEy-a)", PHD(acc_fn, kdey_auto(), reuse_h=h)
    # yield "OCE(KDEy-a)-SLSQP", OCE(acc_fn, kdey_auto(), reuse_h=h, optim_method="SLSQP")
    # if PROBLEM == "binary":
    #     yield "LEAP(DMy)", LEAP(acc_fn, dmy(), reuse_h=h, log_true_solve=True)
    #     yield "PHD(DMy)", PHD(acc_fn, dmy(), reuse_h=h)
    #     yield "OCE(DMy)-SLSQP", OCE(acc_fn, dmy(), reuse_h=h, optim_method="SLSQP")


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
    yield "LEAP(oracle)", LEAP(acc_fn, oracle_q, reuse_h=h, log_true_solve=True)
    yield "PHD(oracle)", PHD(acc_fn, oracle_q, reuse_h=h)
    yield "OCE(oracle)-SLSQP", OCE(acc_fn, oracle_q, reuse_h=h, optim_method="SLSQP")
    # return
    # yield


def gen_methods(h, D):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_baselines_vp(acc_fn, D):
        yield name, method, D.V1, D.V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_methods_with_oracle(h, acc_fn, D):
        yield name, method, D.V, D.V_posteriors


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def get_tr_classifier_names():
    return gen_transformer_model_dataset(only_model_names=True)


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_tr_dataset_names():
    return gen_transformer_model_dataset(only_dataset_names=True)


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def get_method_names(with_oracle=True):
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()
    names = (
        [m for m, _ in gen_baselines(mock_acc_fn)]
        + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
    )
    if with_oracle:
        names += [m for m, _ in gen_methods_with_oracle(mock_h, mock_acc_fn, mock_D)]

    return names


def get_baseline_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()
    baselines_names = [m for m, _ in gen_baselines(mock_acc_fn)]
    baselines_vp_names = [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    return baselines_names[:2] + baselines_vp_names + baselines_names[2:]
