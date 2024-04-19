from abc import ABC, abstractmethod
from typing import Self

from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator


class ClassifierAccuracyPrediction(ABC):
    def __init__(self, h: BaseEstimator):
        self.h = h

    @abstractmethod
    def fit(self, val: LabelledCollection) -> Self:
        """
        Trains a CAP method.

        :param val: training data
        :return: self
        """
        ...

    @abstractmethod
    def predict(self, X, oracle_prev=None) -> float:
        """
        Predicts directly the accuracy using the accuracy function

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        ...
