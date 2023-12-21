from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection

# Extended classes
#
# 0 ~ True 0
# 1 ~ False 1
# 2 ~ False 0
# 3 ~ True 1
#      _____________________
#     |          |          |
#     |  True 0  |  False 1 |
#     |__________|__________|
#     |          |          |
#     |  False 0 |  True 1  |
#     |__________|__________|
#


def _split_index_by_pred(pred_proba: np.ndarray) -> List[np.ndarray]:
    _pred_label = np.argmax(pred_proba, axis=1)
    return [(_pred_label == cl).nonzero()[0] for cl in np.arange(pred_proba.shape[1])]


class ExtensionPolicy:
    def __init__(self, collapse_false=False, group_false=False, dense=False):
        self.collapse_false = collapse_false
        self.group_false = group_false
        self.dense = dense

    def qclasses(self, nbcl):
        if self.collapse_false:
            return np.arange(nbcl + 1)
        elif self.group_false:
            return np.arange(nbcl * 2)

        return np.arange(nbcl**2)

    def eclasses(self, nbcl):
        return np.arange(nbcl**2)

    def tfp_classes(self, nbcl):
        if self.group_false:
            return np.arange(2)
        else:
            return np.arange(nbcl)

    def matrix_idx(self, nbcl):
        if self.collapse_false:
            _idxs = np.array([[i, i] for i in range(nbcl)] + [[0, 1]]).T
            return tuple(_idxs)
        elif self.group_false:
            diag_idxs = np.diag_indices(nbcl)
            sub_diag_idxs = tuple(
                np.array([((i + 1) % nbcl, i) for i in range(nbcl)]).T
            )
            return tuple(np.concatenate(axis) for axis in zip(diag_idxs, sub_diag_idxs))
            # def mask_fn(m, k):
            #     n = m.shape[0]
            #     d = np.diag(np.tile(1, n))
            #     d[tuple(zip(*[(i, (i + 1) % n) for i in range(n)]))] = 1
            #     return d

            # _mi = np.mask_indices(nbcl, mask_func=mask_fn)
            # print(_mi)
            # return _mi
        else:
            _idxs = np.indices((nbcl, nbcl))
            return _idxs[0].flatten(), _idxs[1].flatten()

    def ext_lbl(self, nbcl):
        if self.collapse_false:

            def cf_fun(t, p):
                return t if t == p else nbcl

            return np.vectorize(cf_fun, signature="(),()->()")

        elif self.group_false:

            def gf_fun(t, p):
                # if t < nbcl - 1:
                #     return t * 2 if t == p else (t * 2) + 1
                # else:
                #     return t * 2 if t != p else (t * 2) + 1
                return p if t == p else nbcl + p

            return np.vectorize(gf_fun, signature="(),()->()")

        else:

            def default_fn(t, p):
                return t * nbcl + p

            return np.vectorize(default_fn, signature="(),()->()")

    def true_lbl_from_pred(self, nbcl):
        if self.group_false:
            return np.vectorize(lambda t, p: 0 if t == p else 1, signature="(),()->()")
        else:
            return np.vectorize(lambda t, p: t, signature="(),()->()")

    def can_f1(self, nbcl):
        return nbcl == 2 or (not self.collapse_false and not self.group_false)


class ExtendedData:
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
        extpol=None,
    ):
        self.extpol = ExtensionPolicy() if extpol is None else extpol
        self.b_instances_ = instances
        self.pred_proba_ = pred_proba
        self.ext_ = ext
        self.instances = self.__extend_instances(instances, pred_proba, ext=ext)

    def __extend_instances(
        self,
        instances: np.ndarray | sp.csr_matrix,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
    ) -> np.ndarray | sp.csr_matrix:
        to_append = ext
        if ext is None:
            to_append = pred_proba

        if isinstance(instances, sp.csr_matrix):
            if self.extpol.dense:
                n_x = to_append
            else:
                n_x = sp.hstack([instances, sp.csr_matrix(to_append)], format="csr")
        elif isinstance(instances, np.ndarray):
            _concat = [instances, to_append] if not self.extpol.dense else [to_append]
            n_x = np.concatenate(_concat, axis=1)
        else:
            raise ValueError("Unsupported matrix format")

        return n_x

    @property
    def X(self):
        return self.instances

    @property
    def nbcl(self):
        return self.pred_proba_.shape[1]

    def split_by_pred(self, _indexes: List[np.ndarray] | None = None):
        def _empty_matrix():
            if isinstance(self.instances, np.ndarray):
                return np.asarray([], dtype=int)
            elif isinstance(self.instances, sp.csr_matrix):
                return sp.csr_matrix(np.empty((0, 0), dtype=int))

        if _indexes is None:
            _indexes = _split_index_by_pred(self.pred_proba_)

        _instances = [
            self.instances[ind] if ind.shape[0] > 0 else _empty_matrix()
            for ind in _indexes
        ]

        return _instances

    def __len__(self):
        return self.instances.shape[0]


class ExtendedLabels:
    def __init__(
        self,
        true: np.ndarray,
        pred: np.ndarray,
        nbcl: np.ndarray,
        extpol: ExtensionPolicy = None,
    ):
        self.extpol = ExtensionPolicy() if extpol is None else extpol
        self.true = true
        self.pred = pred
        self.nbcl = nbcl

    @property
    def y(self):
        return self.extpol.ext_lbl(self.nbcl)(self.true, self.pred)

    @property
    def classes(self):
        return self.extpol.qclasses(self.nbcl)

    def __getitem__(self, idx):
        return ExtendedLabels(self.true[idx], self.pred[idx], self.nbcl)

    def split_by_pred(self, _indexes: List[np.ndarray]):
        _labels = []
        for cl, ind in enumerate(_indexes):
            _true, _pred = self.true[ind], self.pred[ind]
            assert (
                _pred.shape[0] == 0 or (_pred == _pred[0]).all()
            ), "index is selecting non uniform class"
            _tfp = self.extpol.true_lbl_from_pred(self.nbcl)(_true, _pred)
            _labels.append(_tfp)

        return _labels, self.extpol.tfp_classes(self.nbcl)


class ExtendedPrev:
    def __init__(
        self,
        flat: np.ndarray,
        nbcl: int,
        extpol: ExtensionPolicy = None,
    ):
        self.flat = flat
        self.nbcl = nbcl
        self.extpol = ExtensionPolicy() if extpol is None else extpol
        # self._matrix = self.__build_matrix()

    def __build_matrix(self):
        _matrix = np.zeros((self.nbcl, self.nbcl))
        _matrix[self.extpol.matrix_idx(self.nbcl)] = self.flat
        return _matrix

    def can_f1(self):
        return self.extpol.can_f1(self.nbcl)

    @property
    def A(self):
        # return self._matrix
        return self.__build_matrix()

    @property
    def classes(self):
        return self.extpol.qclasses(self.nbcl)


class ExtMulPrev(ExtendedPrev):
    def __init__(
        self,
        flat: np.ndarray,
        nbcl: int,
        q_classes: list = None,
        extpol: ExtensionPolicy = None,
    ):
        super().__init__(flat, nbcl, extpol=extpol)
        self.flat = self.__check_q_classes(q_classes, flat)

    def __check_q_classes(self, q_classes, flat):
        if q_classes is None:
            return flat
        q_classes = np.array(q_classes)
        _flat = np.zeros(self.extpol.qclasses(self.nbcl).shape)
        _flat[q_classes] = flat
        return _flat


class ExtBinPrev(ExtendedPrev):
    def __init__(
        self,
        flat: List[np.ndarray],
        nbcl: int,
        q_classes: List[List[int]] = None,
        extpol: ExtensionPolicy = None,
    ):
        super().__init__(flat, nbcl, extpol=extpol)
        flat = self.__check_q_classes(q_classes, flat)
        self.flat = self.__build_flat(flat)

    def __check_q_classes(self, q_classes, flat):
        if q_classes is None:
            return flat
        _flat = []
        for fl, qc in zip(flat, q_classes):
            qc = np.array(qc)
            _fl = np.zeros(self.extpol.tfp_classes(self.nbcl).shape)
            _fl[qc] = fl
            _flat.append(_fl)
        return np.array(_flat)

    def __build_flat(self, flat):
        return np.concatenate(flat.T)


class ExtendedCollection(LabelledCollection):
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        labels: np.ndarray,
        pred_proba: np.ndarray = None,
        ext: np.ndarray = None,
        extpol=None,
    ):
        self.extpol = ExtensionPolicy() if extpol is None else extpol
        e_data, e_labels = self.__extend_collection(
            instances=instances,
            labels=labels,
            pred_proba=pred_proba,
            ext=ext,
        )
        self.e_data_ = e_data
        self.e_labels_ = e_labels
        super().__init__(e_data.X, e_labels.y, classes=e_labels.classes)

    @classmethod
    def from_lc(
        cls,
        lc: LabelledCollection,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
        extpol=None,
    ):
        return ExtendedCollection(
            lc.X, lc.y, pred_proba=pred_proba, ext=ext, extpol=extpol
        )

    @property
    def pred_proba(self):
        return self.e_data_.pred_proba_

    @property
    def ext(self):
        return self.e_data_.ext_

    @property
    def eX(self):
        return self.e_data_

    @property
    def ey(self):
        return self.e_labels_

    @property
    def n_base_classes(self):
        return self.e_labels_.nbcl

    @property
    def n_classes(self):
        return len(self.e_labels_.classes)

    def e_prevalence(self) -> ExtendedPrev:
        _prev = self.prevalence()
        return ExtendedPrev(_prev, self.n_base_classes, extpol=self.extpol)

    def split_by_pred(self):
        _indexes = _split_index_by_pred(self.pred_proba)
        _instances = self.e_data_.split_by_pred(_indexes)
        # _labels = [self.ey[ind] for ind in _indexes]
        _labels, _cls = self.e_labels_.split_by_pred(_indexes)
        return [
            LabelledCollection(inst, lbl, classes=_cls)
            for inst, lbl in zip(_instances, _labels)
        ]

    def __extend_collection(
        self,
        instances: sp.csr_matrix | np.ndarray,
        labels: np.ndarray,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
        extpol=None,
    ) -> Tuple[ExtendedData, ExtendedLabels]:
        n_classes = pred_proba.shape[1]
        # n_X = [ X | predicted probs. ]
        e_instances = ExtendedData(instances, pred_proba, ext=ext, extpol=self.extpol)

        # n_y = (exptected y, predicted y)
        preds = np.argmax(pred_proba, axis=-1)
        e_labels = ExtendedLabels(labels, preds, n_classes, extpol=self.extpol)

        return e_instances, e_labels
