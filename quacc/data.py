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


class ExtendedData:
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
    ):
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
        to_append = pred_proba
        if ext is not None:
            to_append = np.concatenate([ext, pred_proba], axis=1)

        if isinstance(instances, sp.csr_matrix):
            _to_append = sp.csr_matrix(to_append)
            n_x = sp.hstack([instances, _to_append])
        elif isinstance(instances, np.ndarray):
            n_x = np.concatenate((instances, to_append), axis=1)
        else:
            raise ValueError("Unsupported matrix format")

        return n_x

    @property
    def X(self):
        return self.instances

    def __split_index_by_pred(self) -> List[np.ndarray]:
        _pred_label = np.argmax(self.pred_proba_, axis=0)

        return [
            (_pred_label == cl).nonzero()[0]
            for cl in np.arange(self.pred_proba_.shape[0])
        ]

    def split_by_pred(self, return_indexes=False):
        _indexes = self.__split_index_by_pred()
        if isinstance(self.instances, np.ndarray):
            _instances = [
                self.instances[ind] if ind.shape[0] > 0 else np.asarray([], dtype=int)
                for ind in _indexes
            ]
        elif isinstance(self.instances, sp.csr_matrix):
            _instances = [
                self.instances[ind]
                if ind.shape[0] > 0
                else sp.csr_matrix(np.empty((0, 0), dtype=int))
                for ind in _indexes
            ]

        if return_indexes:
            return _instances, _indexes

        return _instances

    def __len__(self):
        return self.instances.shape[0]


class ExtendedLabels:
    def __init__(self, true: np.ndarray, pred: np.ndarray, ncl: np.ndarray):
        self.true = true
        self.pred = pred
        self.ncl = ncl

    @property
    def y(self):
        return self.true * self.ncl + self.pred

    def __getitem__(self, idx):
        return ExtendedLabels(self.true[idx], self.pred[idx], self.ncl)


class ExtendedCollection(LabelledCollection):
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        labels: np.ndarray,
        pred_proba: np.ndarray = None,
        ext: np.ndarray = None,
    ):
        e_data, e_labels, _classes = self.__extend_collection(
            instances=instances,
            labels=labels,
            pred_proba=pred_proba,
            ext=ext,
        )
        self.e_data_ = e_data
        self.e_labels_ = e_labels
        super().__init__(e_data.X, e_labels.y, classes=_classes)

    @classmethod
    def from_lc(
        cls,
        lc: LabelledCollection,
        predict_proba: np.ndarray,
        ext: np.ndarray = None,
    ):
        return ExtendedCollection(lc.X, lc.y, pred_proba=predict_proba, ext=ext)

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

    def split_by_pred(self):
        _ncl = len(self.pred_proba)
        _instances, _indexes = self.e_data_.split_by_pred(return_indexes=True)
        _labels = [self.ey[ind] for ind in _indexes]
        return [
            LabelledCollection(inst, lbl.true, classes=range(0, _ncl))
            for inst, lbl in zip(_instances, _labels)
        ]

    def __extend_collection(
        self,
        instances: sp.csr_matrix | np.ndarray,
        labels: np.ndarray,
        pred_proba: np.ndarray,
        ext: np.ndarray = None,
    ) -> Tuple[ExtendedData, ExtendedLabels, np.ndarray]:
        n_classes = np.unique(labels).shape[0]
        # n_X = [ X | predicted probs. ]
        e_instances = ExtendedData(instances, pred_proba, ext=ext)

        # n_y = (exptected y, predicted y)
        preds = np.argmax(pred_proba, axis=-1)
        e_labels = ExtendedLabels(labels, preds, n_classes)

        return e_instances, e_labels, np.arange(n_classes**2)
