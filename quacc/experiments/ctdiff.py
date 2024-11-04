import itertools as IT

import numpy as np
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression as LR

from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import LEAP, PHD, CAPContingencyTable, LabelledCollection

PROBLEM = "binary"
MODEL_TYPE = "simple"

qp.environ["_R_SEED"] = 0

if PROBLEM == "binary":
    qp.environ["SAMPLE_SIZE"] = 100


def sld():
    return EMQ(LR(), val_split=5)


def kdey():
    return KDEyML(LR())


class PredictedSet:
    def __init__(self, set, posteriors):
        self.A = set
        self.post = posteriors


def gen_classifiers():
    if MODEL_TYPE == "simple":
        yield "LR", LR()


def gen_datasets() -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    if PROBLEM == "binary":
        if MODEL_TYPE == "simple":
            _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
            _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
            for dn in _uci_names:
                yield dn, fetch_UCIBinaryDataset(dn)


def gen_methods(h, V_ps, V1_ps, V2_prot_ps):
    acc_fn = vanilla_acc
    yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h), V_ps
    yield "PHD(SLD)", PHD(acc_fn, sld(), reuse_h=h), V_ps
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h), V_ps
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h), V_ps


def gen_classifier_dataset():
    for classifier in gen_classifiers():
        for dataset in gen_datasets():
            yield classifier, dataset


def get_cts(method: CAPContingencyTable, test_prot, test_prot_posteriors):
    cts_list = []
    for sample, sample_post in zip(test_prot(), test_prot_posteriors):
        cts_list.append(method.predict_ct(sample.X, sample_post))
    return np.asarray(cts_list)


def get_cts_diff_mean(cts1, cts2):
    cts_diff = np.abs(cts1 - cts2)
    return np.mean(cts_diff, axis=0)


def contingency_matrix(y, y_hat, n_classes):
    ct = np.zeros((n_classes, n_classes))
    for _c in range(n_classes):
        _idx = y == _c
        for _c1 in range(n_classes):
            ct[_c, _c1] = np.sum(y_hat[_idx] == _c1)

    return ct / y.shape[0]


def ctdiff():
    NUM_TEST = 1000

    results = {}
    for (cls_name, h), (dataset_name, (L, V, U)) in gen_classifier_dataset():
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # split validation set
        V1, V2_prot = split_validation(V)

        # precomumpute model posteriors for validation and test sets
        V_ps = PredictedSet(V, h.predict_proba(V.X))
        V1_ps = PredictedSet(V1, h.predict_proba(V1.X))
        V2_prot_ps = PredictedSet(V2_prot, [h.predict_proba(sample.X) for sample in V2_prot()])

        test_prot_posteriors, test_prot_y_hat, true_cts = [], [], []
        for sample in test_prot():
            P = h.predict_proba(sample.X)
            y_hat = np.argmax(P, axis=-1)
            true_cts.append(contingency_matrix(sample.y, y_hat, sample.n_classes))
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(y_hat)

        results_cts = {}
        for method_name, method, val_ps in gen_methods(h, V_ps, V1_ps, V2_prot_ps):
            val, val_posteriors = val_ps.A, val_ps.post
            method.fit(val, val_posteriors)
            estim_cts = get_cts(method, test_prot, test_prot_posteriors)
            results_cts[method_name] = estim_cts

        diff_cts = {}
        for method1, method2 in IT.product(results_cts.keys(), results_cts.keys()):
            if method1 == method2 or (method1, method2) in diff_cts or (method2, method1) in diff_cts:
                continue
            diff_cts[(method1, method2)] = get_cts_diff_mean(results_cts[method1], results_cts[method2])
            diff_cts[(method1, "true")] = get_cts_diff_mean(results_cts[method1], true_cts)
            diff_cts[(method2, "true")] = get_cts_diff_mean(results_cts[method2], true_cts)

        results[(cls_name, dataset_name)] = diff_cts

    print(results)


if __name__ == "__main__":
    ctdiff()
