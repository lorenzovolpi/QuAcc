from abc import abstractmethod
from copy import deepcopy
from typing import List

import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.method.aggregative import BaseQuantifier
from sklearn.base import BaseEstimator

import quacc.method.confidence as conf
from quacc.data import (
    ExtBinPrev,
    ExtendedCollection,
    ExtendedData,
    ExtendedPrev,
    ExtensionPolicy,
    ExtMulPrev,
)


class BaseAccuracyEstimator(BaseQuantifier):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
        dense=False,
    ):
        self.__check_classifier(classifier)
        self.quantifier = quantifier
        self.extpol = ExtensionPolicy(dense=dense)

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
            coll, pred_proba=pred_proba, ext=pred_proba, extpol=self.extpol
        )

    def _extend_instances(self, instances: np.ndarray | sp.csr_matrix):
        pred_proba = self.classifier.predict_proba(instances)
        return ExtendedData(instances, pred_proba=pred_proba, extpol=self.extpol)

    @abstractmethod
    def fit(self, train: LabelledCollection | ExtendedCollection):
        ...

    @abstractmethod
    def estimate(self, instances, ext=False) -> ExtendedPrev:
        ...

    @property
    def dense(self):
        return self.extpol.dense


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

    def extend(
        self, coll: LabelledCollection, pred_proba=None, prefit=False
    ) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        if prefit:
            self._fit_confidence(coll.X, coll.y, pred_proba)
        else:
            if not hasattr(self, "confidence_metrics"):
                raise AttributeError(
                    "Confidence metrics are not fit and cannot be computed."
                    "Consider setting prefit to True."
                )

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
        group_false=False,
        dense=False,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.extpol = ExtensionPolicy(
            collapse_false=collapse_false,
            group_false=group_false,
            dense=dense,
        )
        self.e_train = None

    # def _get_pred_ext(self, pred_proba: np.ndarray):
    #     return np.argmax(pred_proba, axis=1, keepdims=True)

    def _get_multi_quant(self, quant, train: LabelledCollection):
        _nz = np.nonzero(train.counts())[0]
        if _nz.shape[0] == 1:
            return TrivialQuantifier(train.n_classes, _nz[0])
        else:
            return quant

    def fit(self, train: LabelledCollection):
        pred_proba = self.classifier.predict_proba(train.X)
        self._fit_confidence(train.X, train.y, pred_proba)
        self.e_train = self.extend(train, pred_proba=pred_proba)

        self.quantifier = self._get_multi_quant(self.quantifier, self.e_train)
        self.quantifier.fit(self.e_train)

        return self

    def estimate(
        self, instances: ExtendedData | np.ndarray | sp.csr_matrix
    ) -> ExtendedPrev:
        e_inst = instances
        if not isinstance(e_inst, ExtendedData):
            e_inst = self._extend_instances(instances)

        estim_prev = self.quantifier.quantify(e_inst.X)
        return ExtMulPrev(
            estim_prev,
            e_inst.nbcl,
            q_classes=self.quantifier.classes_,
            extpol=self.extpol,
        )

    @property
    def collapse_false(self):
        return self.extpol.collapse_false

    @property
    def group_false(self):
        return self.extpol.group_false


class TrivialQuantifier:
    def __init__(self, n_classes, trivial_class):
        self.trivial_class = trivial_class

    def fit(self, train: LabelledCollection):
        pass

    def quantify(self, inst: LabelledCollection) -> np.ndarray:
        return np.array([1.0])

    @property
    def classes_(self):
        return np.array([self.trivial_class])


class QuantifierProxy:
    def __init__(self, train: LabelledCollection):
        self.o_nclasses = train.n_classes
        self.o_classes = train.classes_
        self.o_index = {c: i for i, c in enumerate(train.classes_)}

        self.mapping = {}
        self.r_mapping = {}
        _cnt = 0
        for cl, c in zip(train.classes_, train.counts()):
            if c > 0:
                self.mapping[cl] = _cnt
                self.r_mapping[_cnt] = cl
                _cnt += 1

        self.n_nclasses = len(self.mapping)

    def apply_mapping(self, coll: LabelledCollection) -> LabelledCollection:
        if not self.proxied:
            return coll

        n_labels = np.copy(coll.labels)
        for k in self.mapping:
            n_labels[coll.labels == k] = self.mapping[k]

        return LabelledCollection(coll.X, n_labels, classes=np.arange(self.n_nclasses))

    def apply_rmapping(self, prevs: np.ndarray, q_classes: np.ndarray) -> np.ndarray:
        if not self.proxied:
            return prevs, q_classes

        n_qclasses = np.array([self.r_mapping[qc] for qc in q_classes])

        return prevs, n_qclasses

    def get_trivial(self):
        return TrivialQuantifier(self.o_nclasses, self.n_nclasses)

    @property
    def proxied(self):
        return self.o_nclasses != self.n_nclasses


class BinaryQuantifierAccuracyEstimator(ConfidenceBasedAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseAccuracyEstimator,
        confidence: str = None,
        group_false: bool = False,
        dense: bool = False,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.quantifiers = []
        self.extpol = ExtensionPolicy(
            group_false=group_false,
            dense=dense,
        )

    def _get_binary_quant(self, quant, train: LabelledCollection):
        _nz = np.nonzero(train.counts())[0]
        if _nz.shape[0] == 1:
            return TrivialQuantifier(train.n_classes, _nz[0])
        else:
            return deepcopy(quant)

    def fit(self, train: LabelledCollection | ExtendedCollection):
        pred_proba = self.classifier.predict_proba(train.X)
        self._fit_confidence(train.X, train.y, pred_proba)
        self.e_train = self.extend(train, pred_proba=pred_proba)

        self.n_classes = self.e_train.n_classes
        e_trains = self.e_train.split_by_pred()

        self.quantifiers = []
        for train in e_trains:
            quant = self._get_binary_quant(self.quantifier, train)
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

        # estim_prev = np.concatenate(estim_prevs.T)
        # return ExtendedPrev(estim_prev, e_inst.nbcl, extpol=self.extpol)

        return ExtBinPrev(
            estim_prevs,
            e_inst.nbcl,
            q_classes=[quant.classes_ for quant in self.quantifiers],
            extpol=self.extpol,
        )

    def _quantify_helper(
        self,
        s_inst: List[np.ndarray | sp.csr_matrix],
        norms: List[float],
    ):
        estim_prevs = []
        for quant, inst, norm in zip(self.quantifiers, s_inst, norms):
            if inst.shape[0] > 0:
                estim_prev = quant.quantify(inst) * norm
                estim_prevs.append(estim_prev)
            else:
                estim_prevs.append(np.zeros((len(quant.classes_),)))

        # return np.array(estim_prevs)
        return estim_prevs

    @property
    def group_false(self):
        return self.extpol.group_false


BAE = BaseAccuracyEstimator
MCAE = MultiClassAccuracyEstimator
BQAE = BinaryQuantifierAccuracyEstimator
