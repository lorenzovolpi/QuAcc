import itertools as IT
from typing import Callable

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import AbstractProtocol
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

import cap
import cap.models.utils as utils
from cap.models.base import ClassifierAccuracyPrediction
from cap.models.utils import max_conf, neg_entropy


class CAPDirect(ClassifierAccuracyPrediction):
    def __init__(self, acc: Callable):
        super().__init__()
        self.acc = acc

    def true_acc(self, sample: LabelledCollection, posteriors):
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)

    def switch_and_fit(self, acc_fn, data, posteriors):
        self.acc = acc_fn
        return self.fit(data, posteriors)


class PrediQuant(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        quantifier: AggregativeQuantifier,
        protocol: AbstractProtocol,
        prot_posteriors,
        alpha=0.3,
        error: str | Callable = cap.error.mae,
        predict_train_prev=True,
    ):
        super().__init__(acc_fn)
        self.q = quantifier
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.alpha = alpha
        self.sample_size = qp.environ["SAMPLE_SIZE"]
        self.__check_error(error)
        self.predict_train_prev = predict_train_prev

    def __check_error(self, error):
        if error in cap.error.ACCURACY_ERROR_SINGLE:
            self.error = error
        elif isinstance(error, str) and error in cap.error.ACCURACY_ERROR_SINGLE_NAMES:
            self.error = cap.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str\n"
                f"representing the name of an error function in {cap.error.ACCURACY_ERROR_NAMES}"
            )

    def fit(self, val: LabelledCollection, posteriors):
        v2, v1 = val.split_stratified(train_prop=0.5)
        self.q.fit(v1, fit_classifier=False, val_split=v1)

        # precompute classifier predictions on samples
        self.sigma_acc = [
            self.true_acc(sigma_i, P)
            for sigma_i, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]

        # precompute prevalence predictions on samples
        if self.predict_train_prev:
            self.sigma_pred_prevs = [self.q.aggregate(P) for P in self.prot_posteriors]
        else:
            self.sigma_pred_prevs = [
                sigma_i.prevalence() for sigma_i in self.protocol()
            ]

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        if oracle_prev is None:
            test_pred_prev = self.q.quantify(X)
        else:
            test_pred_prev = oracle_prev

        if self.alpha > 0:
            # select samples from V2 with predicted prevalence close to the predicted prevalence for U
            selected_accuracies = []
            for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                max_discrepancy = np.max(self.error(pred_prev_i, test_pred_prev))
                if max_discrepancy < self.alpha:
                    selected_accuracies.append(acc_i)

            return np.median(selected_accuracies)
        else:
            # mean average, weights samples from V2 according to the closeness of predicted prevalence in U
            accum_weight = 0
            moving_mean = 0
            epsilon = 10e-4
            for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                max_discrepancy = np.max(self.error(pred_prev_i, test_pred_prev))
                weight = -np.log(max_discrepancy + epsilon)
                accum_weight += weight
                moving_mean += weight * acc_i

            return moving_mean / accum_weight


class PabloCAP(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        quantifier: AggregativeQuantifier,
        n_val_samples=100,
        aggr="mean",
    ):
        super().__init__(acc_fn)
        self.q = quantifier
        self.n_val_samples = n_val_samples
        self.aggr = aggr
        assert aggr in [
            "mean",
            "median",
        ], "unknown aggregation function, use mean or median"

    def fit(self, val: LabelledCollection, posteriors):
        self.q.fit(val)
        label_predictions = np.argmax(posteriors, axis=-1)
        self.pre_classified = LabelledCollection(
            instances=label_predictions, labels=val.labels
        )
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        if oracle_prev is None:
            pred_prev = utils.smooth(self.q.quantify(X))
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


### baselines ###


class ATC(CAPDirect):
    VALID_FUNCTIONS = {"maxconf", "neg_entropy"}

    def __init__(self, acc_fn: Callable, scoring_fn="maxconf"):
        assert scoring_fn in ATC.VALID_FUNCTIONS, (
            f"unknown scoring function, use any from {ATC.VALID_FUNCTIONS}"
        )
        # assert acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        super().__init__(acc_fn)
        self.scoring_fn = scoring_fn

    def get_scores(self, P):
        if self.scoring_fn == "maxconf":
            scores = max_conf(P)
        else:
            scores = neg_entropy(P)
        return scores

    def fit(self, val: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=1)
        true_labels = val.y
        scores = self.get_scores(posteriors)
        _, self.threshold = self.__find_ATC_threshold(
            scores=scores, labels=(pred_labels == true_labels)
        )
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        scores = self.get_scores(posteriors)
        # assert self.acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        return self.__get_ATC_acc(self.threshold, scores)

    def __find_ATC_threshold(self, scores, labels):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        sorted_idx = np.argsort(scores)

        sorted_scores = scores[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_scores[i]

        return min_fp_fn, thres

    def __get_ATC_acc(self, thres, scores):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        return np.mean(scores >= thres)


class DoC(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        protocol: AbstractProtocol,
        prot_posteriors,
        clip_vals=(0, 1),
    ):
        super().__init__(acc_fn)
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.clip_vals = clip_vals

    def _get_post_stats(self, X, y, posteriors):
        P = posteriors
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        acc = self.acc(y, pred_labels)
        return mc, acc

    def _doc(self, mc1, mc2):
        return mc2.mean() - mc1.mean()

    def train_regression(self, prot_mcs, prot_accs):
        docs = [self._doc(self.val_mc, prot_mc_i) for prot_mc_i in prot_mcs]
        target = [self.val_acc - prot_acc_i for prot_acc_i in prot_accs]
        docs = np.asarray(docs).reshape(-1, 1)
        target = np.asarray(target)
        lin_reg = LinearRegression()
        return lin_reg.fit(docs, target)

    def predict_regression(self, test_mc):
        docs = np.asarray([self._doc(self.val_mc, test_mc)]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return self.val_acc - pred_acc

    def fit(self, val: LabelledCollection, posteriors):
        self.val_mc, self.val_acc = self._get_post_stats(*val.Xy, posteriors)

        prot_stats = [
            self._get_post_stats(*sample.Xy, P)
            for sample, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        prot_mcs, prot_accs = list(zip(*prot_stats))

        self.reg_model = self.train_regression(prot_mcs, prot_accs)

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        mc = max_conf(posteriors)
        acc_pred = self.predict_regression(mc)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred
