from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

import baselines.atc as atc

__confs = {}


def metric(name):
    def wrapper(cl):
        __confs[name] = cl
        return cl

    return wrapper


class ConfidenceMetric:
    def fit(self, X, y, probas):
        pass

    def conf(self, X, probas):
        return probas


@metric("max_conf")
class MaxConf(ConfidenceMetric):
    def conf(self, X, probas):
        _mc = np.max(probas, axis=1, keepdims=True)
        return _mc


@metric("entropy")
class Entropy(ConfidenceMetric):
    def conf(self, X, probas):
        _ent = np.sum(
            np.multiply(probas, np.log(probas + 1e-20)), axis=1, keepdims=True
        )
        return _ent


@metric("isoft")
class InverseSoftmax(ConfidenceMetric):
    def conf(self, X, probas):
        _probas = probas / np.sum(probas, axis=1, keepdims=True)
        _probas = np.log(_probas) - np.mean(np.log(_probas), axis=1, keepdims=True)
        return np.max(_probas, axis=1, keepdims=True)


@metric("threshold")
class Threshold(ConfidenceMetric):
    def get_scores(self, probas, keepdims=False):
        return np.max(probas, axis=1, keepdims=keepdims)

    def fit(self, X, y, probas):
        scores = self.get_scores(probas)
        _, self.threshold = atc.find_ATC_threshold(scores, y)

    def conf(self, X, probas):
        scores = self.get_scores(probas, keepdims=True)
        _exp = scores - self.threshold
        return _exp


@metric("linreg")
class LinReg(ConfidenceMetric):
    def extend(self, X, probas):
        if sp.issparse(X):
            return sp.hstack([X, probas])
        else:
            return np.concatenate([X, probas], axis=1)

    def fit(self, X, y, probas):
        reg_X = self.extend(X, probas)
        reg_y = probas[np.arange(probas.shape[0]), y]
        self.reg = LinearRegression()
        self.reg.fit(reg_X, reg_y)

    def conf(self, X, probas):
        reg_X = self.extend(X, probas)
        return self.reg.predict(reg_X)[:, np.newaxis]


def get_metrics(names: List[str]):
    if names is None:
        return None

    __fnames = [n for n in names if n in __confs]
    return [__confs[m]() for m in __fnames]
