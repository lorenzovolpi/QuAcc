from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from quacc.legacy.data import LabelledCollection


class ClassifierAccuracyPrediction(ABC):
    def __init__(self, h: BaseEstimator, acc: callable):
        self.h = h
        self.acc = acc

    @abstractmethod
    def fit(self, val: LabelledCollection): ...

    @abstractmethod
    def predict(self, X, oracle_prev=None):
        """
        Evaluates the accuracy function on the predicted contingency table

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        return ...

    def true_acc(self, sample: LabelledCollection):
        y_pred = self.h.predict(sample.X)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)


class SebastianiCAP(ClassifierAccuracyPrediction):
    def __init__(
        self, h, acc_fn, q_class, n_val_samples=500, alpha=0.3, predict_train_prev=True
    ):
        self.h = h
        self.acc = acc_fn
        self.q = q_class(h)
        self.n_val_samples = n_val_samples
        self.alpha = alpha
        self.sample_size = qp.environ["SAMPLE_SIZE"]
        self.predict_train_prev = predict_train_prev

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=0.5)
        self.q.fit(v1, fit_classifier=False, val_split=v1)

        # precompute classifier predictions on samples
        gen_samples = UPP(
            v2,
            repeats=self.n_val_samples,
            sample_size=self.sample_size,
            return_type="labelled_collection",
        )
        self.sigma_acc = [self.true_acc(sigma_i) for sigma_i in gen_samples()]

        # precompute prevalence predictions on samples
        if self.predict_train_prev:
            gen_samples.on_preclassified_instances(self.q.classify(v2.X), in_place=True)
            self.sigma_pred_prevs = [
                self.q.aggregate(sigma_i.X) for sigma_i in gen_samples()
            ]
        else:
            self.sigma_pred_prevs = [sigma_i.prevalence() for sigma_i in gen_samples()]

    def predict(self, X, oracle_prev=None):
        if oracle_prev is None:
            test_pred_prev = self.q.quantify(X)
        else:
            test_pred_prev = oracle_prev

        if self.alpha > 0:
            # select samples from V2 with predicted prevalence close to the predicted prevalence for U
            selected_accuracies = []
            for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                max_discrepancy = np.max(np.abs(pred_prev_i - test_pred_prev))
                if max_discrepancy < self.alpha:
                    selected_accuracies.append(acc_i)

            return np.median(selected_accuracies)
        else:
            # mean average, weights samples from V2 according to the closeness of predicted prevalence in U
            accum_weight = 0
            moving_mean = 0
            epsilon = 10e-4
            for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                max_discrepancy = np.max(np.abs(pred_prev_i - test_pred_prev))
                weight = -np.log(max_discrepancy + epsilon)
                accum_weight += weight
                moving_mean += weight * acc_i

            return moving_mean / accum_weight


class PabloCAP(ClassifierAccuracyPrediction):
    def __init__(self, h, acc_fn, q_class, n_val_samples=100, aggr="mean"):
        self.h = h
        self.acc = acc_fn
        self.q = q_class(deepcopy(h))
        self.n_val_samples = n_val_samples
        self.aggr = aggr
        assert aggr in [
            "mean",
            "median",
        ], "unknown aggregation function, use mean or median"

    def fit(self, val: LabelledCollection):
        self.q.fit(val)
        label_predictions = self.h.predict(val.X)
        self.pre_classified = LabelledCollection(
            instances=label_predictions, labels=val.labels
        )

    def predict(self, X, oracle_prev=None):
        if oracle_prev is None:
            pred_prev = F.smooth(self.q.quantify(X))
        else:
            pred_prev = oracle_prev
        X_size = X.shape[0]
        acc_estim = []
        for _ in range(self.n_val_samples):
            sigma_i = self.pre_classified.sampling(X_size, *pred_prev[:-1])
            y_pred, y_true = sigma_i.Xy
            conf_table = confusion_matrix(
                y_true, y_pred=y_pred, labels=sigma_i.classes_
            )
            acc_i = self.acc(conf_table)
            acc_estim.append(acc_i)
        if self.aggr == "mean":
            return np.mean(acc_estim)
        elif self.aggr == "median":
            return np.median(acc_estim)
        else:
            raise ValueError("unknown aggregation function")
