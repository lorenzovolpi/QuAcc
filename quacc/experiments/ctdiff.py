import numpy as np
from quapy.protocol import UPP

from quacc.experiments.util import split_validation


class PredictedSet:
    def __init__(self, set, posteriors):
        self.A = set
        self.post = posteriors


def gen_classifiers():
    pass


def gen_datasets():
    pass


def gen_methods(h, V_ps, V1_ps, V2_prot_ps):
    pass


def gen_classifier_dataset():
    for classifier in gen_classifiers():
        for dataset in gen_datasets():
            yield classifier, dataset


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

        for method_name, method, val_ps in gen_methods(h, V_ps, V1_ps, V2_prot_ps):
            val, val_posteriors = val_ps.A, val_ps.post


if __name__ == "__main__":
    ctdiff()
