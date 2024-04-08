from abc import abstractmethod
from copy import deepcopy

import numpy as np
import quapy.functional as F
import scipy
from quapy.data.base import LabelledCollection as LC
from quapy.method.aggregative import AggregativeQuantifier
from quapy.method.base import BaseQuantifier
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.utils import get_posteriors_from_h, max_conf, neg_entropy


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
        non_empty_collection = LabelledCollection(*self.Xy, classes=non_empty)
        return non_empty_collection, old_pos


class CAPContingencyTable(ClassifierAccuracyPrediction):
    def __init__(self, h: BaseEstimator, acc: callable):
        self.h = h
        self.acc = acc

    def predict(self, X, oracle_prev=None):
        """
        Evaluates the accuracy function on the predicted contingency table

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        cont_table = self.predict_ct(X, oracle_prev)
        raw_acc = self.acc(cont_table)
        norm_acc = np.clip(raw_acc, 0, 1)
        return norm_acc

    @abstractmethod
    def predict_ct(self, X, oracle_prev=None):
        """
        Predicts the contingency table for the test data

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a contingency table
        """
        ...


class NaiveCAP(CAPContingencyTable):
    """
    The Naive CAP is a method that relies on the IID assumption, and thus uses the estimation in the validation data
    as an estimate for the test data.
    """

    def __init__(self, h: BaseEstimator, acc: callable):
        super().__init__(h, acc)

    def fit(self, val: LabelledCollection):
        y_hat = self.h.predict(val.X)
        y_true = val.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=val.classes_)
        return self

    def predict_ct(self, test, oracle_prev=None):
        """
        This method disregards the test set, under the assumption that it is IID wrt the training. This meaning that
        the confusion matrix for the test data should coincide with the one computed for training (using any cross
        validation strategy).

        :param test: test collection (ignored)
        :param oracle_prev: ignored
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        return self.cont_table


class CAPContingencyTableQ(CAPContingencyTable):
    def __init__(
        self,
        h: BaseEstimator,
        acc: callable,
        q_class: AggregativeQuantifier,
        reuse_h=False,
    ):
        super().__init__(h, acc)
        self.reuse_h = reuse_h
        if reuse_h:
            assert isinstance(
                q_class, AggregativeQuantifier
            ), f"quantifier {q_class} is not of type aggregative"
            self.q = deepcopy(q_class)
            self.q.set_params(classifier=h)
        else:
            self.q = q_class

    def quantifier_fit(self, val: LabelledCollection):
        if self.reuse_h:
            self.q.fit(val, fit_classifier=False, val_split=val)
        else:
            self.q.fit(val)


class ContTableTransferCAP(CAPContingencyTableQ):
    """ """

    def __init__(self, h: BaseEstimator, acc: callable, q_class, reuse_h=False):
        super().__init__(h, acc, q_class, reuse_h)

    def fit(self, val: LabelledCollection):
        y_hat = self.h.predict(val.X)
        y_true = val.y
        self.cont_table = confusion_matrix(
            y_true=y_true, y_pred=y_hat, labels=val.classes_, normalize="all"
        )
        self.train_prev = val.prevalence()
        self.quantifier_fit(val)
        return self

    def predict_ct(self, test, oracle_prev=None):
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

    def __init__(self, h: BaseEstimator, acc: callable, q_class, reuse_h=False):
        super().__init__(h, acc, q_class, reuse_h)

    def fit(self, val: LabelledCollection):
        y_hat = self.h.predict(val.X)
        y_true = val.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=val.classes_)
        self.quantifier_fit(val)
        self.A, self.partial_b = self._construct_equations()
        return self

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

        # (n-1)*(n-1) equations: the class cond rations should be the same in training and in test due to the
        # PPS assumptions. Example in three classes, a ratio: a/(a+b+c) [test] = ar [a ratio in training]
        # a / (a + b + c) = ar
        # a = (a + b + c) * ar
        # a = a ar + b ar + c ar
        # a - a ar - b ar - c ar = 0
        # a (1-ar) + b (-ar)  + c (-ar) = 0
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(
            axis=1, keepdims=True
        )
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

    def predict_ct(self, test, oracle_prev):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = self.h.predict(test)
        cc_prev_estim = F.prevalence_from_labels(h_label_preds, self.h.classes_)
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

        if any(x < 0) or any(x > 0) or not np.isclose(x.sum(), 1):
            print("L", end="")

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            x = F.optim_minimize(loss, n_classes=n**2)

        else:
            print(".", end="")

        cont_table_test = x.reshape(n, n)
        return cont_table_test


class QuAcc:
    def _get_X_dot(self, X):
        h = self.h

        P = get_posteriors_from_h(h, X)

        add_covs = []

        if self.add_posteriors:
            add_covs.append(P[:, 1:])

        if self.add_maxconf:
            mc = max_conf(P, keepdims=True)
            add_covs.append(mc)

        if self.add_negentropy:
            ne = neg_entropy(P, keepdims=True)
            add_covs.append(ne)

        if self.add_maxinfsoft:
            lgP = np.log(P)
            mis = np.max(lgP - lgP.mean(axis=1, keepdims=True), axis=1, keepdims=True)
            add_covs.append(mis)

        if len(add_covs) > 0:
            X_dot = np.hstack(add_covs)

        if self.add_X:
            X_dot = safehstack(X, X_dot)

        return X_dot


class QuAcc1xN2(CAPContingencyTableQ, QuAcc):
    def __init__(
        self,
        h: BaseEstimator,
        acc: callable,
        q_class: AggregativeQuantifier,
        add_X=True,
        add_posteriors=True,
        add_maxconf=False,
        add_negentropy=False,
        add_maxinfsoft=False,
    ):
        self.h = h
        self.acc = acc
        self.q = EmptySafeQuantifier(q_class)
        self.add_X = add_X
        self.add_posteriors = add_posteriors
        self.add_maxconf = add_maxconf
        self.add_negentropy = add_negentropy
        self.add_maxinfsoft = add_maxinfsoft

    def fit(self, val: LabelledCollection):
        pred_labels = self.h.predict(val.X)
        true_labels = val.y

        self.ncl = val.n_classes
        classes_dot = np.arange(self.ncl**2)
        ct_class_idx = classes_dot.reshape(self.ncl, self.ncl)

        X_dot = self._get_X_dot(val.X)
        y_dot = ct_class_idx[true_labels, pred_labels]
        val_dot = LabelledCollection(X_dot, y_dot, classes=classes_dot)
        self.q.fit(val_dot)

    def predict_ct(self, X, oracle_prev=None):
        X_dot = self._get_X_dot(X)
        flat_ct = self.q.quantify(X_dot)
        return flat_ct.reshape(self.ncl, self.ncl)


class QuAcc1xNp1(CAPContingencyTableQ, QuAcc):
    def __init__(
        self,
        h: BaseEstimator,
        acc: callable,
        q_class: AggregativeQuantifier,
        add_X=True,
        add_posteriors=True,
        add_maxconf=False,
        add_negentropy=False,
        add_maxinfsoft=False,
    ):
        self.h = h
        self.acc = acc
        self.q = EmptySafeQuantifier(q_class)
        self.add_X = add_X
        self.add_posteriors = add_posteriors
        self.add_maxconf = add_maxconf
        self.add_negentropy = add_negentropy
        self.add_maxinfsoft = add_maxinfsoft

    def fit(self, val: LabelledCollection):
        pred_labels = self.h.predict(val.X)
        true_labels = val.y

        self.ncl = val.n_classes
        classes_dot = np.arange(self.ncl + 1)
        # ct_class_idx = classes_dot.reshape(n, n)
        ct_class_idx = np.full((self.ncl, self.ncl), self.ncl)
        ct_class_idx[np.diag_indices(self.ncl)] = np.arange(self.ncl)

        X_dot = self._get_X_dot(val.X)
        y_dot = ct_class_idx[true_labels, pred_labels]
        val_dot = LabelledCollection(X_dot, y_dot, classes=classes_dot)
        self.q.fit(val_dot)

    def _get_ct_hat(self, n, ct_compressed):
        _diag_idx = np.diag_indices(n)
        ct_rev_idx = (np.append(_diag_idx[0], 0), np.append(_diag_idx[1], 1))
        ct_hat = np.zeros((n, n))
        ct_hat[ct_rev_idx] = ct_compressed
        return ct_hat

    def predict_ct(self, X: LabelledCollection, oracle_prev=None):
        X_dot = self._get_X_dot(X)
        ct_compressed = self.q.quantify(X_dot)
        return self._get_ct_hat(self.ncl, ct_compressed)


class QuAccNxN(CAPContingencyTableQ, QuAcc):
    def __init__(
        self,
        h: BaseEstimator,
        acc: callable,
        q_class: AggregativeQuantifier,
        add_X=True,
        add_posteriors=True,
        add_maxconf=False,
        add_negentropy=False,
        add_maxinfsoft=False,
    ):
        self.h = h
        self.acc = acc
        self.q_class = q_class
        self.add_X = add_X
        self.add_posteriors = add_posteriors
        self.add_maxconf = add_maxconf
        self.add_negentropy = add_negentropy
        self.add_maxinfsoft = add_maxinfsoft

    def fit(self, val: LabelledCollection):
        pred_labels = self.h.predict(val.X)
        true_labels = val.y
        X_dot = self._get_X_dot(val.X)

        self.q = []
        for class_i in self.h.classes_:
            X_dot_i = X_dot[pred_labels == class_i]
            y_i = true_labels[pred_labels == class_i]
            data_i = LabelledCollection(X_dot_i, y_i, classes=val.classes_)

            q_i = EmptySafeQuantifier(deepcopy(self.q_class))
            q_i.fit(data_i)
            self.q.append(q_i)

    def predict_ct(self, X, oracle_prev=None):
        classes = self.h.classes_
        pred_labels = self.h.predict(X)
        X_dot = self._get_X_dot(X)
        pred_prev = F.prevalence_from_labels(pred_labels, classes)
        cont_table = []
        for class_i, q_i, p_i in zip(classes, self.q, pred_prev):
            X_dot_i = X_dot[pred_labels == class_i]
            classcond_cond_table_prevs = q_i.quantify(X_dot_i)
            cond_table_prevs = p_i * classcond_cond_table_prevs
            cont_table.append(cond_table_prevs)
        cont_table = np.vstack(cont_table)
        return cont_table


def safehstack(X, P):
    if issparse(X) or issparse(P):
        XP = scipy.sparse.hstack([X, P])
        XP = csr_matrix(XP)
    else:
        XP = np.hstack([X, P])
    return XP


class EmptySafeQuantifier(BaseQuantifier):
    def __init__(self, surrogate_quantifier: BaseQuantifier):
        self.surrogate = surrogate_quantifier

    def fit(self, data: LabelledCollection):
        self.n_classes = data.n_classes
        class_compact_data, self.old_class_idx = data.compact_classes()
        if self.num_non_empty_classes() > 1:
            self.surrogate.fit(class_compact_data)
        return self

    def quantify(self, instances):
        num_instances = instances.shape[0]
        if self.num_non_empty_classes() == 0 or num_instances == 0:
            # returns the uniform prevalence vector
            uniform = np.full(
                fill_value=1.0 / self.n_classes, shape=self.n_classes, dtype=float
            )
            return uniform
        elif self.num_non_empty_classes() == 1:
            # returns a prevalence vector with 100% of the mass in the only non empty class
            prev_vector = np.full(fill_value=0.0, shape=self.n_classes, dtype=float)
            prev_vector[self.old_class_idx[0]] = 1
            return prev_vector
        else:
            class_compact_prev = self.surrogate.quantify(instances)
            prev_vector = np.full(fill_value=0.0, shape=self.n_classes, dtype=float)
            prev_vector[self.old_class_idx] = class_compact_prev
            return prev_vector

    def num_non_empty_classes(self):
        return len(self.old_class_idx)
