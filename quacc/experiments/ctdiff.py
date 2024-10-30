import numpy as np
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression as LR

from quacc.error import vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import LEAP, PHD, CAPContingencyTable


def sld():
    return EMQ(LR(), val_split=5)


def kdey():
    return KDEyML(LR())


class PredictedSet:
    def __init__(self, set, posteriors):
        self.A = set
        self.post = posteriors


def gen_classifiers():
    pass


def gen_datasets():
    pass


def gen_methods(h, V_ps, V1_ps, V2_prot_ps):
    acc_fn = vanilla_acc
    yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h), V_ps
    yield "PHD(SLD)", PHD(acc_fn, sld(), reuse_h=h)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h)
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)


def gen_classifier_dataset():
    for classifier in gen_classifiers():
        for dataset in gen_datasets():
            yield classifier, dataset


def get_cts(method: CAPContingencyTable, test_prot, test_prot_posteriors):
    cts_list = []
    for sample, sample_post in zip(test_prot(), test_prot_posteriors):
        cts_list.appen(method.predict_ct(sample, sample_post))
    return np.asarray(cts_list)


def ctdiff():
    NUM_TEST = 1000

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

        test_prot_posteriors, test_prot_y_hat = [], []
        for sample in test_prot():
            P = h.predict_proba(sample.X)
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(np.argmax(P, axis=-1))

        out_map = {}
        for method_name, method, val_ps in gen_methods(h, V_ps, V1_ps, V2_prot_ps):
            val, val_posteriors = val_ps.A, val_ps.post
            method.fit(val, val_posteriors)
            estim_cts = get_cts(method, test_prot, test_prot_posteriors)
            out_map[method_name] = estim_cts

        print(out_map)


if __name__ == "__main__":
    ctdiff()
