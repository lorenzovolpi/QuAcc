from copy import deepcopy
from pprint import pp
from time import time

import numpy as np
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import UPP
from sklearn.linear_model import LinearRegression, LogisticRegression

from quacc.models.cont_table import QuAcc
from quacc.models.direct import CAPDirect
from quacc.models.utils import get_posteriors_from_h


class ReQuAcc(CAPDirect):
    def __init__(
        self,
        h,
        acc_fn,
        quacc_classes: QuAcc | list[QuAcc],
        sample_size,
        n_val_samples=500,
        val_prop=0.5,
        clip_vals=(0, 1),
        verbose=False,
    ):
        super().__init__(h, acc_fn)
        self._build_models(quacc_classes)
        self.sample_size = sample_size
        self.n_val_samples = n_val_samples
        self.val_prop = val_prop
        self.clip_vals = clip_vals
        self.verbose = verbose

    def _sout(self, msg):
        if self.verbose:
            print(msg)

    def _build_models(self, quacc_classes: QuAcc | list[QuAcc]):
        if isinstance(quacc_classes, list):
            self.models = [deepcopy(qcc) for qcc in quacc_classes]
        else:
            self.models = [deepcopy(quacc_classes)]

    def _get_post_stats(self, X, y):
        cts = self._get_models_cts(X)
        P = get_posteriors_from_h(self.h, X)
        pred_labels = np.argmax(P, axis=-1)
        acc = self.acc(y, pred_labels)
        return cts, acc

    def _get_models_cts(self, X):
        return np.hstack([m.predict_ct(X).flatten() for m in self.models])

    def predict_regression(self, test_cts: np.ndarray):
        test_cts = test_cts.reshape(1, -1)
        pred_acc = self.reg_model.predict(test_cts)
        return pred_acc

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=self.val_prop)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.n_val_samples,
            return_type="labelled_collection",
        )

        # training models
        for m in self.models:
            t_init = time()
            m.fit(v1)
            self._sout(f"training {m.__class__.__name__}({m.q_class.__class__.__name__}) took {time() - t_init:.3f}s")

        # predicting v2 sample cont. tables for each model
        v2_ctss = []
        for m in self.models:
            t_init = time()
            cts = np.vstack([m.predict_ct(v2_i.X).flatten() for v2_i in v2_prot()])
            self._sout(
                f"predicting {m.__class__.__name__}({m.q_class.__class__.__name__}) on val prot took {time() - t_init:.3f}s"
            )
            v2_ctss.append(cts)
        v2_ctss = np.hstack(v2_ctss)

        # computing v2 samples true accs
        v2_accs = np.asarray([self.true_acc(v2_i) for v2_i in v2_prot()])

        # fitting linear regression model
        t_init = time()
        self.reg_model = LinearRegression()
        self.reg_model.fit(v2_ctss, v2_accs)
        self._sout(f"training reg_model took {time() - t_init:.3f}s")

        return self

    def predict(self, X, oracle_prev=None):
        test_cts = self._get_models_cts(X)
        acc_pred = self.predict_regression(test_cts)
        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred[0]
