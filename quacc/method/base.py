from abc import abstractmethod
from copy import deepcopy
from typing import List

import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.method.aggregative import BaseQuantifier
from sklearn.base import BaseEstimator

import quacc.method.confidence as conf
from quacc.data import ExtendedCollection, ExtendedData, ExtensionPolicy


class BaseAccuracyEstimator(BaseQuantifier):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
    ):
        self.__check_classifier(classifier)
        self.quantifier = quantifier
        self.extpol = ExtensionPolicy()

    def __check_classifier(self, classifier):
        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                f"Passed classifier {classifier.__class__.__name__} cannot predict probabilities."
            )
        self.classifier = classifier

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        return ExtendedCollection.from_lc(
            coll, pred_proba=pred_proba, extpol=self.extpol
        )

    def _extend_instances(self, instances: np.ndarray | sp.csr_matrix):
        pred_proba = self.classifier.predict_proba(instances)
        return ExtendedData(instances, pred_proba=pred_proba, extpol=self.extpol)

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
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
        )
        self.__check_confidence(confidence)
        self.calibrator = None

    def __check_confidence(self, confidence):
        if isinstance(confidence, str):
            self.confidence = [confidence]
        elif isinstance(confidence, list):
            self.confidence = confidence
        else:
            self.confidence = None

    def _fit_confidence(self, X, y, probas):
        self.confidence_metrics = conf.get_metrics(self.confidence)
        if self.confidence_metrics is None:
            return

        for m in self.confidence_metrics:
            m.fit(X, y, probas)

    def _get_pred_ext(self, pred_proba: np.ndarray):
        return pred_proba

    def __get_ext(
        self, X: np.ndarray | sp.csr_matrix, pred_proba: np.ndarray
    ) -> np.ndarray:
        if self.confidence_metrics is None or len(self.confidence_metrics) == 0:
            return pred_proba

        _conf_ext = np.concatenate(
            [m.conf(X, pred_proba) for m in self.confidence_metrics],
            axis=1,
        )

        _pred_ext = self._get_pred_ext(pred_proba)

        return np.concatenate([_conf_ext, _pred_ext], axis=1)

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        _ext = self.__get_ext(coll.X, pred_proba)
        return ExtendedCollection.from_lc(
            coll, pred_proba=pred_proba, ext=_ext, extpol=self.extpol
        )

    def _extend_instances(
        self,
        instances: np.ndarray | sp.csr_matrix,
    ) -> ExtendedData:
        pred_proba = self.classifier.predict_proba(instances)
        _ext = self.__get_ext(instances, pred_proba)
        return ExtendedData(
            instances, pred_proba=pred_proba, ext=_ext, extpol=self.extpol
        )


class MultiClassAccuracyEstimator(ConfidenceBasedAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
        confidence: str = None,
        collapse_false=False,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.extpol = ExtensionPolicy(collapse_false=collapse_false)
        self.e_train = None

    def _get_pred_ext(self, pred_proba: np.ndarray):
        return np.argmax(pred_proba, axis=1, keepdims=True)

    def fit(self, train: LabelledCollection):
        pred_proba = self.classifier.predict_proba(train.X)
        self._fit_confidence(train.X, train.y, pred_proba)
        self.e_train = self.extend(train, pred_proba=pred_proba)

        self.quantifier.fit(self.e_train)

        return self

    def estimate(
        self, instances: ExtendedData | np.ndarray | sp.csr_matrix
    ) -> np.ndarray:
        e_inst = instances
        if not isinstance(e_inst, ExtendedData):
            e_inst = self._extend_instances(instances)

        estim_prev = self.quantifier.quantify(e_inst.X)
        estim_prev = self._check_prevalence_classes(
            estim_prev, self.quantifier.classes_
        )
        if self.extpol.collapse_false:
            estim_prev = np.insert(estim_prev, 2, 0.0)

        return estim_prev

    def _check_prevalence_classes(self, estim_prev, estim_classes) -> np.ndarray:
        true_classes = self.e_train.classes_
        for _cls in true_classes:
            if _cls not in estim_classes:
                estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
        return estim_prev

    @property
    def collapse_false(self):
        return self.extpol.collapse_false


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
        pred_proba = self.classifier.predict_proba(train.X)
        self._fit_confidence(train.X, train.y, pred_proba)
        self.e_train = self.extend(train, pred_proba=pred_proba)

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

        # estim_prev = np.array([prev_row for prev_row in zip(*estim_prevs)]).flatten()
        estim_prev = np.concatenate(estim_prevs.T)
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

        return np.array(estim_prevs)


BAE = BaseAccuracyEstimator
MCAE = MultiClassAccuracyEstimator
BQAE = BinaryQuantifierAccuracyEstimator
