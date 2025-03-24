from typing import override

import matplotlib.pyplot as plt
import numpy as np
import quapy as qp
import seaborn as sns
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeCrispQuantifier, KDEyML
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from exp.leap.config import NUM_TEST, gen_acc_measure, kdey
from exp.util import gen_model_dataset
from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.models.cont_table import LEAP, OCE, PHD


class MockQuantifier(AggregativeCrispQuantifier):
    def __init__(self, classifier: BaseEstimator, mock_q: np.ndarray, val_split=5) -> None:
        self.classifier = qp._get_classifier(classifier)
        self.mock_q = mock_q
        self.val_split = val_split

    @override
    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        assert data.n_classes == self.mock_q.shape[0], "Invalid mock_q: shape is incompatible with number of classes"

    @override
    def aggregate(self, classif_predictions: np.ndarray):
        return self.mock_q


def gen_classifiers():
    yield "LR", LogisticRegression()


def gen_datasets():
    for d in ["pageblocks.5"]:
        yield d, fetch_UCIBinaryDataset(d)


def gen_methods(h):
    _, acc_fn = next(gen_acc_measure())
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
    yield "OCE(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP")


def clip(qs, epsilon=1e-5):
    qs = np.where(qs >= 1, np.full_like(qs, 1 - epsilon), qs)
    qs = np.where(qs <= 0, np.full_like(qs, epsilon), qs)
    return qs


if __name__ == "__main__":
    qp.environ["SAMPLE_SIZE"] = 100
    for (cls_name, h), (dataset_name, (L, V, U)) in gen_model_dataset(gen_classifiers, gen_datasets):
        h.fit(*L.Xy)

        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=qp.environ["_R_SEED"])

        q = KDEyML(h).fit(V, fit_classifier=False, val_split=V)

        # pred_priors = np.array([q.quantify(Ui.X) for Ui in test_prot()])
        # q_mean = pred_priors[:, -1].mean()
        # q_std = pred_priors[:, -1].std()
        # print(q_mean, q_std)

        q_mean, q_std = 0.5, 0.152

        q_1s = np.random.normal(q_mean, q_std, NUM_TEST**2)
        q_1s = clip(q_1s)
        mocks = np.vstack([1 - q_1s, q_1s]).T

        assert np.all(mocks <= 1) and np.all(mocks >= 0)
