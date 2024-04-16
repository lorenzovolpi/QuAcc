from abc import ABC, abstractmethod

from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator


class ClassifierAccuracyPrediction(ABC):
    def __init__(self, h: BaseEstimator):
        self.h = h

    @abstractmethod
    def fit(self, val: LabelledCollection): ...
