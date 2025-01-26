import pdb
from abc import abstractmethod
from copy import deepcopy
from typing import Callable

import numpy as np
import quapy.functional as F
import scipy
from quapy.data.base import LabelledCollection as LC
from quapy.method.aggregative import AggregativeQuantifier
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.utils import max_conf, max_inverse_softmax, neg_entropy


class LabelledCollection(LC):
    def empty_classes(self):
        """
        Returns a np.ndarray of empty classes (classes present in self.classes_ but with
        no positive instance). In case there is none, then an empty np.ndarray is returned

        :return: np.ndarray
        """
        idx = np.argwhere(self.counts() == 0).flatten()
        return self.classes_[idx]

    def non_empty_classes(self):
        """
        Returns a np.ndarray of non-empty classes (classes present in self.classes_ but with
        at least one positive instance). In case there is none, then an empty np.ndarray is returned

        :return: np.ndarray
        """
        idx = np.argwhere(self.counts() > 0).flatten()
        return self.classes_[idx]

    def has_empty_classes(self):
        """
        Checks whether the collection has empty classes

        :return: boolean
        """
        return len(self.empty_classes()) > 0

    def compact_classes(self):
        """
        Generates a new LabelledCollection object with no empty classes. It also returns a np.ndarray of
        indexes that correspond to the old indexes of the new self.classes_.

        :return: (LabelledCollection, np.ndarray,)
        """
        non_empty = self.non_empty_classes()
        all_classes = self.classes_
        old_pos = np.searchsorted(all_classes, non_empty)
        compact_classes = np.arange(len(old_pos))
        compact_y = np.array(self.y, copy=True)
        for necls, ccls in zip(non_empty, compact_classes):
            compact_y[self.y == necls] = ccls
        non_empty_collection = LabelledCollection(self.X, compact_y, classes=compact_classes)
        return non_empty_collection, old_pos


class CAPContingencyTable(ClassifierAccuracyPrediction):
    def __init__(self, acc_fn: Callable):
        self.acc_fn = acc_fn

    @abstractmethod
    def predict_ct(self, X, posteriors, oracle_prev=None) -> np.ndarray:
        """
        Predicts the contingency table for the test data

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a contingency table
        """
        ...

    def switch(self, acc_fn):
        self.acc_fn = acc_fn
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        cont_table = self.predict_ct(X, posteriors, oracle_prev)
        return self.acc_fn(cont_table)


class NaiveCAP(CAPContingencyTable):
    """
    The Naive CAP is a method that relies on the IID assumption, and thus uses the estimation in the validation data
    as an estimate for the test data.
    """

    def __init__(self, acc_fn: Callable):
        super().__init__(acc_fn)

    def fit(self, val: LabelledCollection, posteriors):
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = val.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=val.classes_)
        return self

    def predict_ct(self, test, posteriors, oracle_prev=None):
        """
        This method disregards the test set, under the assumption that it is IID wrt the training. This meaning that
        the confusion matrix for the test data should coincide with the one computed for training (using any cross
        validation strategy).

        :param test: test collection (ignored)
        :param oracle_prev: ignored
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        return self.cont_table


class CAPContingencyTableQ(CAPContingencyTable, BaseEstimator):
    def __init__(
        self,
        acc_fn: Callable,
        q_class: AggregativeQuantifier,
        reuse_h: BaseEstimator | None = None,
    ):
        CAPContingencyTable.__init__(self, acc_fn)
        self.reuse_h = reuse_h
        self.q_class = q_class

    def preprocess_data(self, data: LabelledCollection, posteriors):
        return data

    def prepare_quantifier(self):
        if self.reuse_h is not None:
            assert isinstance(self.q_class, AggregativeQuantifier), (
                f"quantifier {self.q_class} is not of type aggregative"
            )
            self.q = deepcopy(self.q_class)
            self.q.set_params(classifier=self.reuse_h)
        else:
            self.q = self.q_class

    def quant_classifier_fit_predict(self, data: LabelledCollection):
        if self.reuse_h is not None:
            return self.q.classifier_fit_predict(data, fit_classifier=False, predict_on=data)
        else:
            return self.q.classifier_fit_predict(data)

    def quant_aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.q.aggregation_fit(classif_predictions, data)

    def fit(self, data: LabelledCollection, posteriors):
        data = self.preprocess_data(data, posteriors)
        self.prepare_quantifier()
        classif_predictions = self.quant_classifier_fit_predict(data)
        self.quant_aggregation_fit(classif_predictions, data)
        return self

    def true_acc(self, sample: LabelledCollection, posteriors, acc_fn=None):
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        acc_fn = self.acc if acc_fn is None else acc_fn
        return acc_fn(conf_table)


class ContTableTransferCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        reuse_h: BaseEstimator | None = None,
    ):
        super().__init__(acc_fn, q_class, reuse_h)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true=y_true, y_pred=y_hat, labels=data.classes_, normalize="all")
        self.train_prev = data.prevalence()
        return data

    def predict_ct(self, test, posteriors, oracle_prev=None):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        if oracle_prev is None:
            prev_hat = self.q.quantify(test)
        else:
            prev_hat = oracle_prev
        adjustment = prev_hat / self.train_prev
        return self.cont_table * adjustment[:, np.newaxis]


class NsquaredEquationsCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        always_optimize=False,
        reuse_h: BaseEstimator | None = None,
        verbose=False,
    ):
        super().__init__(acc_fn, q_class, reuse_h)
        self.verbose = verbose
        self.always_optimize = always_optimize

    def _sout(self, *msgs, **kwargs):
        if self.verbose:
            print(*msgs, **kwargs)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        self.classes_ = data.classes_
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]

        # I is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        I = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n * n, n * n))
        b = np.zeros(shape=(n * n))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the
        # PPS assumptions. Example in three classes, a ratio: a/(a+b+c) [test] = ar [a ratio in training]
        # a / (a + b + c) = ar
        # a = (a + b + c) * ar
        # a = a ar + b ar + c ar
        # a - a ar - b ar - c ar = 0
        # a (1-ar) + b (-ar)  + c (-ar) = 0
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(1, n):
            for j in range(1, n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, I[i, :]] = -ratio_ij
                A[eq_no, I[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(1, n):
            A[eq_no, I[:, i]] = 1
            # b[eq_no] = cc_prev_estim[i]
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(1, n):
            A[eq_no, I[i, :]] = 1
            # b[eq_no] = q_prev_estim[i]
            eq_no += 1

        return A, b

    def predict_ct(self, test, posteriors, oracle_prev=None, return_true_solve=False):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = F.prevalence_from_labels(h_label_preds, self.classes_)
        if oracle_prev is None:
            q_prev_estim = self.q.quantify(test)
        else:
            q_prev_estim = oracle_prev

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * (n - 1) : -(n - 1)] = cc_prev_estim[1:]
        b[-(n - 1) :] = q_prev_estim[1:]

        # try the fast solution (may not be valid)
        x = np.linalg.solve(A, b)

        _true_solve = True
        if any(x < 0) or not np.isclose(x.sum(), 1) or self.always_optimize:
            self._sout("L", end="")
            _true_solve = False

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            x = F.optim_minimize(loss, n_classes=n**2)

        else:
            self._sout(".", end="")

        cont_table_test = x.reshape(n, n)
        if return_true_solve:
            return cont_table_test, _true_solve
        else:
            return cont_table_test


class OverConstrainedEquationsCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        reuse_h: BaseEstimator | None = None,
        verbose=False,
    ):
        super().__init__(acc_fn, q_class, reuse_h)
        self.verbose = verbose

    def _sout(self, *msgs, **kwargs):
        if self.verbose:
            print(*msgs, **kwargs)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        self.classes_ = data.classes_
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]
        n_eqs = n * n + 2 * n + 1
        n_unknowns = n * n

        # I is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        I = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n_eqs, n_unknowns))
        b = np.zeros(shape=(n_eqs))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the
        # PPS assumptions. Example in three classes, a ratio: a/(a+b+c) [test] = ar [a ratio in training]
        # a / (a + b + c) = ar
        # a = (a + b + c) * ar
        # a = a ar + b ar + c ar
        # a - a ar - b ar - c ar = 0
        # a (1-ar) + b (-ar)  + c (-ar) = 0
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(n):
            for j in range(n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, I[i, :]] = -ratio_ij
                A[eq_no, I[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(n):
            A[eq_no, I[:, i]] = 1
            # b[eq_no] = cc_prev_estim[i]
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(n):
            A[eq_no, I[i, :]] = 1
            # b[eq_no] = q_prev_estim[i]
            eq_no += 1

        return A, b

    def predict_ct(self, test, posteriors, oracle_prev=None):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = F.prevalence_from_labels(h_label_preds, self.classes_)
        if oracle_prev is None:
            q_prev_estim = self.q.quantify(test)
        else:
            q_prev_estim = oracle_prev

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * n : -n] = cc_prev_estim
        b[-n:] = q_prev_estim

        def loss(x):
            return np.linalg.norm(A @ x - b, ord=2)

        x = F.optim_minimize(loss, n_classes=n**2)

        cont_table_test = x.reshape(n, n)
        return cont_table_test


class QuAcc(CAPContingencyTableQ):
    def __init__(
        self,
        acc_fn: Callable,
        q_class: AggregativeQuantifier,
        add_X=True,
        add_posteriors=True,
        add_y_hat=False,
        add_maxconf=False,
        add_negentropy=False,
        add_maxinfsoft=False,
    ):
        super().__init__(acc_fn, q_class)
        self.add_X = add_X
        self.add_posteriors = add_posteriors
        self.add_y_hat = add_y_hat
        self.add_maxconf = add_maxconf
        self.add_negentropy = add_negentropy
        self.add_maxinfsoft = add_maxinfsoft

    def _get_X_dot(self, X, posteriors):
        P = posteriors
        simplex = P[:, 1:]

        add_covs = []

        if self.add_X:
            add_covs.append(X)

        if self.add_posteriors:
            add_covs.append(simplex)

        if self.add_y_hat:
            y_hat = np.argmax(P, axis=-1, keepdims=True)
            add_covs.append(y_hat)

        if self.add_maxconf:
            mc = max_conf(P, keepdims=True)
            add_covs.append(mc)

        if self.add_negentropy:
            ne = neg_entropy(P, keepdims=True)
            add_covs.append(ne)

        if self.add_maxinfsoft:
            mis = max_inverse_softmax(P, keepdims=True)
            add_covs.append(mis)

        if len(add_covs) > 0:
            X_dot = safehstack(add_covs)
        else:
            X_dot = simplex

        return X_dot

    def _num_non_empty_classes(self):
        return len(self.q_old_class_idx)

    def quant_classifier_fit_predict(self, data: LabelledCollection):
        self.q_n_classes = data.n_classes
        class_compact_data, self.q_old_class_idx = data.compact_classes()
        if self._num_non_empty_classes() > 1:
            return self.q.classifier_fit_predict(class_compact_data)
        return None

    def quant_aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.q_n_classes = data.n_classes
        class_compact_data, _ = data.compact_classes()
        if self._num_non_empty_classes() > 1:
            self.q.aggregation_fit(classif_predictions, class_compact_data)

    def _safe_quantify(self, instances):
        num_instances = instances.shape[0]
        if self._num_non_empty_classes() == 0 or num_instances == 0:
            # returns the uniform prevalence vector
            uniform = np.full(fill_value=1.0 / self.q_n_classes, shape=self.q_n_classes, dtype=float)
            return uniform
        elif self._num_non_empty_classes() == 1:
            # returns a prevalence vector with 100% of the mass in the only non empty class
            prev_vector = np.full(fill_value=0.0, shape=self.q_n_classes, dtype=float)
            prev_vector[self.q_old_class_idx[0]] = 1
            return prev_vector
        else:
            class_compact_prev = self.q.quantify(instances)
            prev_vector = np.full(fill_value=0.0, shape=self.q_n_classes, dtype=float)
            prev_vector[self.q_old_class_idx] = class_compact_prev
            return prev_vector


class QuAcc1xN2(QuAcc):
    def preprocess_data(self, data: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=-1)
        true_labels = data.y

        self.ncl = data.n_classes
        classes_dot = np.arange(self.ncl**2)
        ct_class_idx = classes_dot.reshape(self.ncl, self.ncl)

        X_dot = self._get_X_dot(data.X, posteriors)
        y_dot = ct_class_idx[true_labels, pred_labels]
        return LabelledCollection(X_dot, y_dot, classes=classes_dot)

    def prepare_quantifier(self):
        self.q = deepcopy(self.q_class)

    def predict_ct(self, X: LabelledCollection, posteriors, oracle_prev=None):
        # pdb.set_trace()
        X_dot = self._get_X_dot(X, posteriors)
        flat_ct = self._safe_quantify(X_dot)
        return flat_ct.reshape(self.ncl, self.ncl)


class QuAcc1xNp1(QuAcc):
    def preprocess_data(self, data: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=-1)
        true_labels = data.y

        self.ncl = data.n_classes
        classes_dot = np.arange(self.ncl + 1)
        # ct_class_idx = classes_dot.reshape(n, n)
        ct_class_idx = np.full((self.ncl, self.ncl), self.ncl)
        ct_class_idx[np.diag_indices(self.ncl)] = np.arange(self.ncl)

        X_dot = self._get_X_dot(data.X, posteriors)
        y_dot = ct_class_idx[true_labels, pred_labels]
        return LabelledCollection(X_dot, y_dot, classes=classes_dot)

    def prepare_quantifier(self):
        self.q = deepcopy(self.q_class)

    def _get_ct_hat(self, n, ct_compressed):
        _diag_idx = np.diag_indices(n)
        ct_rev_idx = (np.append(_diag_idx[0], 0), np.append(_diag_idx[1], 1))
        ct_hat = np.zeros((n, n))
        ct_hat[ct_rev_idx] = ct_compressed
        return ct_hat

    def predict_ct(self, X: LabelledCollection, posteriors, oracle_prev=None):
        X_dot = self._get_X_dot(X, posteriors)
        ct_compressed = self._safe_quantify(X_dot)
        return self._get_ct_hat(self.ncl, ct_compressed)


class QuAcc1xNN(QuAcc):
    def preprocess_data(self, data: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=-1)
        true_labels = data.y

        self.ncl = data.n_classes
        classes_dot = np.arange(2 * self.ncl)
        # ct_class_idx = classes_dot.reshape(n, n)
        ct_class_idx = (np.full((self.ncl, self.ncl), self.ncl) + np.arange(self.ncl)).T
        ct_class_idx[np.diag_indices(self.ncl)] = np.arange(self.ncl)

        X_dot = self._get_X_dot(data.X, posteriors)
        y_dot = ct_class_idx[true_labels, pred_labels]
        return LabelledCollection(X_dot, y_dot, classes=classes_dot)

    def prepare_quantifier(self):
        self.q = deepcopy(self.q_class)

    def _get_ct_hat(self, n, ct_compressed):
        _diag_idx = np.diag_indices(n)
        ct_rev_idx = (np.append(_diag_idx[0], (_diag_idx[0] + 1) % n), np.append(_diag_idx[1], _diag_idx[1]))
        ct_hat = np.zeros((n, n))
        ct_hat[ct_rev_idx] = ct_compressed
        return ct_hat

    def predict_ct(self, X: LabelledCollection, posteriors, oracle_prev=None):
        X_dot = self._get_X_dot(X, posteriors)
        ct_compressed = self._safe_quantify(X_dot)
        return self._get_ct_hat(self.ncl, ct_compressed)


class QuAccNxN(QuAcc):
    def preprocess_data(self, data: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=-1)
        true_labels = data.y
        X_dot = self._get_X_dot(data.X, posteriors)

        datas = []
        self.classes_ = data.classes_
        for class_i in self.classes_:
            X_dot_i = X_dot[pred_labels == class_i]
            y_i = true_labels[pred_labels == class_i]
            data_i = LabelledCollection(X_dot_i, y_i, classes=data.classes_)
            datas.append(data_i)

        return datas

    def prepare_quantifier(self):
        self.q: list[AggregativeQuantifier] = []
        for class_i in self.classes_:
            q_i = deepcopy(self.q_class)
            self.q.append(q_i)

    def _num_non_empty_classes(self):
        return [len(old_class_idx_i) for old_class_idx_i in self.q_old_class_idx]

    def _safe_q_classifier_fit_predict(self, compact_data: LabelledCollection):
        classif_predictions = []
        for q_i, data_i, num_nec_i in zip(self.q, compact_data, self._num_non_empty_classes()):
            if num_nec_i <= 1:
                classif_predictions.append(None)
                continue

            predict_on = q_i.val_split
            if isinstance(predict_on, int):
                if predict_on <= 1:
                    raise ValueError(f"invalid value {predict_on} in fit. Specify a integer >1 for kFCV estimation.")

                skf = StratifiedKFold(n_splits=predict_on)

                _invalid_split = np.all(data_i.counts() < predict_on)
                if not _invalid_split:
                    _split_n_classes = [len(np.unique(data_i.y[train_idx])) for train_idx, _ in skf.split(*data_i.Xy)]
                    _invalid_split = np.any(np.array(_split_n_classes) <= 1)

                if _invalid_split:
                    q_i.classifier.fit(*data_i.Xy)
                    preds = q_i.classify(data_i.X)
                    preds = LabelledCollection(preds, data_i.y, classes=data_i.classes_)
                else:
                    preds = cross_val_predict(q_i.classifier, *data_i.Xy, cv=skf, method=q_i._classifier_method())
                    preds = LabelledCollection(preds, data_i.y, classes=data_i.classes_)
                    q_i.classifier.fit(*data_i.Xy)
            else:
                preds = q_i.classifier_fit_predict(data_i)

            classif_predictions.append(preds)

        return classif_predictions

    def quant_classifier_fit_predict(self, data: list[LabelledCollection]):
        self.q_n_classes = [data_i.n_classes for data_i in data]
        compact_data, self.q_old_class_idx = tuple(map(list, zip(*[data_i.compact_classes() for data_i in data])))
        classif_predictions = self._safe_q_classifier_fit_predict(compact_data)

        return classif_predictions

    def quant_aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        compact_data, _ = tuple(map(list, zip(*[data_i.compact_classes() for data_i in data])))
        for q_i, cp_i, compact_data_i, num_nec_i in zip(
            self.q, classif_predictions, compact_data, self._num_non_empty_classes()
        ):
            if num_nec_i > 1:
                q_i.aggregation_fit(cp_i, compact_data_i)

    def _safe_quantify(self, instances_list):
        prev_vectors = []
        for X, q_i, num_nec_i, n_classes_i, qoci_i in zip(
            instances_list, self.q, self._num_non_empty_classes(), self.q_n_classes, self.q_old_class_idx
        ):
            num_instances = X.shape[0]
            if num_nec_i == 0 or num_instances == 0:
                uniform = np.full(fill_value=1.0 / n_classes_i, shape=n_classes_i, dtype=float)
                prev_vectors.append(uniform)
            elif num_nec_i == 1:
                prev_vector = np.full(fill_value=0.0, shape=n_classes_i, dtype=float)
                prev_vector[qoci_i[0]] = 1
                prev_vectors.append(prev_vector)
            else:
                class_compact_prev = q_i.quantify(X)
                prev_vector = np.full(fill_value=0.0, shape=n_classes_i, dtype=float)
                prev_vector[qoci_i] = class_compact_prev
                prev_vectors.append(prev_vector)

        return prev_vectors

    def predict_ct(self, X: LabelledCollection, posteriors, oracle_prev=None):
        pred_labels = np.argmax(posteriors, axis=-1)
        X_dot = self._get_X_dot(X, posteriors)
        pred_prev = F.prevalence_from_labels(pred_labels, self.classes_)
        X_dot_list = [X_dot[pred_labels == class_i] for class_i in self.classes_]
        classcond_cond_table_prevs = self._safe_quantify(X_dot_list)
        cont_table = [p_i * cctp_i for p_i, cctp_i in zip(pred_prev, classcond_cond_table_prevs)]
        cont_table = np.vstack(cont_table)
        return cont_table


# def safehstack(X, P):
#     if issparse(X) or issparse(P):
#         XP = scipy.sparse.hstack([X, P])
#         XP = csr_matrix(XP)
#     else:
#         XP = np.hstack([X, P])
#     return XP


def safehstack(covs):
    if any(map(issparse, covs)):
        XP = scipy.sparse.hstack(covs)
        XP = csr_matrix(XP)
    else:
        XP = np.hstack(covs)
    return XP


LEAP = NsquaredEquationsCAP
PHD = ContTableTransferCAP
OCE = OverConstrainedEquationsCAP
