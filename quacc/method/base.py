from abc import abstractmethod
from copy import deepcopy
from typing import List

import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.method.aggregative import BaseQuantifier
from sklearn.base import BaseEstimator

from quacc.data import ExtendedCollection, ExtendedData


class BaseAccuracyEstimator(BaseQuantifier):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
    ):
        self.__check_classifier(classifier)
        self.quantifier = quantifier

    def __check_classifier(self, classifier):
        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                f"Passed classifier {classifier.__class__.__name__} cannot predict probabilities."
            )
        self.classifier = classifier

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        return ExtendedCollection.from_lc(coll, pred_proba=pred_proba)

    def _extend_instances(self, instances: np.ndarray | sp.csr_matrix, pred_proba=None):
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(instances)

        return ExtendedData(instances, pred_proba=pred_proba)

    @abstractmethod
    def fit(self, train: LabelledCollection | ExtendedCollection):
        ...

    @abstractmethod
    def estimate(self, instances, ext=False) -> np.ndarray:
        ...


class ConfidenceBasedAccuracyEstimator(BaseAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
        confidence=None,
    ):
        super().__init__(classifier, quantifier)
        self.__check_confidence(confidence)

    def __check_confidence(self, confidence):
        if isinstance(confidence, str):
            self.confidence = [confidence]
        elif isinstance(confidence, list):
            self.confidence = confidence
        else:
            self.confidence = None

    def __get_confidence(self):
        def max_conf(probas):
            _mc = np.max(probas, axis=-1)
            _min = 1.0 / probas.shape[1]
            _norm_mc = (_mc - _min) / (1.0 - _min)
            return _norm_mc

        def entropy(probas):
            _ent = np.sum(np.multiply(probas, np.log(probas + 1e-20)), axis=1)
            return _ent

        if self.confidence is None:
            return []

        __confs = {
            "max_conf": max_conf,
            "entropy": entropy,
        }
        return [__confs.get(c, None) for c in self.confidence]

    def __get_ext(self, pred_proba: np.ndarray) -> np.ndarray:
        __confidence = self.__get_confidence()

        if __confidence is None or len(__confidence) == 0:
            return None

        return np.concatenate(
            [
                _f_conf(pred_proba).reshape((len(pred_proba), 1))
                for _f_conf in __confidence
                if _f_conf is not None
            ],
            axis=1,
        )

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        _ext = self.__get_ext(pred_proba)
        return ExtendedCollection.from_lc(coll, pred_proba=pred_proba, ext=_ext)

    def _extend_instances(
        self,
        instances: np.ndarray | sp.csr_matrix,
        pred_proba=None,
    ) -> ExtendedData:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(instances)

        _ext = self.__get_ext(pred_proba)
        return ExtendedData(instances, pred_proba=pred_proba, ext=_ext)


class MultiClassAccuracyEstimator(ConfidenceBasedAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
        confidence: str = None,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.e_train = None

    def fit(self, train: LabelledCollection):
        self.e_train = self.extend(train)

        self.quantifier.fit(self.e_train)

        return self

    def estimate(
        self, instances: ExtendedData | np.ndarray | sp.csr_matrix
    ) -> np.ndarray:
        e_inst = instances
        if not isinstance(e_inst, ExtendedData):
            e_inst = self._extend_instances(instances)

        estim_prev = self.quantifier.quantify(e_inst.X)
        return self._check_prevalence_classes(estim_prev, self.quantifier.classes_)

    def _check_prevalence_classes(self, estim_prev, estim_classes) -> np.ndarray:
        true_classes = self.e_train.classes_
        for _cls in true_classes:
            if _cls not in estim_classes:
                estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
        return estim_prev


class BinaryQuantifierAccuracyEstimator(ConfidenceBasedAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseAccuracyEstimator,
        confidence: str = None,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.quantifiers = []

    def fit(self, train: LabelledCollection | ExtendedCollection):
        self.e_train = self.extend(train)

        self.n_classes = self.e_train.n_classes
        e_trains = self.e_train.split_by_pred()

        self.quantifiers = []
        for train in e_trains:
            quant = deepcopy(self.quantifier)
            quant.fit(train)
            self.quantifiers.append(quant)

        return self

    def estimate(
        self, instances: ExtendedData | np.ndarray | sp.csr_matrix
    ) -> np.ndarray:
        e_inst = instances
        if not isinstance(e_inst, ExtendedData):
            e_inst = self._extend_instances(instances)

        s_inst = e_inst.split_by_pred()
        norms = [s_i.shape[0] / len(e_inst) for s_i in s_inst]
        estim_prevs = self._quantify_helper(s_inst, norms)

        estim_prev = np.array([prev_row for prev_row in zip(*estim_prevs)]).flatten()
        return estim_prev

    def _quantify_helper(
        self,
        s_inst: List[np.ndarray | sp.csr_matrix],
        norms: List[float],
    ):
        estim_prevs = []
        for quant, inst, norm in zip(self.quantifiers, s_inst, norms):
            if inst.shape[0] > 0:
                estim_prevs.append(quant.quantify(inst) * norm)
            else:
                estim_prevs.append(np.asarray([0.0, 0.0]))

        return estim_prevs


BAE = BaseAccuracyEstimator
MCAE = MultiClassAccuracyEstimator
BQAE = BinaryQuantifierAccuracyEstimator
