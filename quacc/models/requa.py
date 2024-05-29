import itertools as IT
from copy import deepcopy
from time import time

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import UPP
from sklearn.linear_model import LinearRegression, LogisticRegression

import quacc as qc
from quacc.models.cont_table import QuAcc
from quacc.models.direct import CAPDirect
from quacc.models.utils import get_posteriors_from_h
from quacc.utils.commons import parallel as qc_parallel


class ReQua(CAPDirect):
    def __init__(
        self,
        h,
        acc_fn,
        quacc_classes: QuAcc | list[QuAcc],
        param_grid: dict,
        sample_size,
        n_val_samples=500,
        val_prop=0.5,
        clip_vals=(0, 1),
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(h, acc_fn)
        self.param_grid = param_grid
        self._build_models(quacc_classes)
        self.sample_size = sample_size
        self.n_val_samples = n_val_samples
        self.val_prop = val_prop
        self.clip_vals = clip_vals
        self.n_jobs = qc.commons.get_njobs(n_jobs)
        self.verbose = verbose
        self.joblib_verbose = 60 if verbose else 0

    def _sout(self, msg):
        if self.verbose:
            print(msg)

    def _build_models(self, quacc_classes: QuAcc | list[QuAcc]):
        if isinstance(quacc_classes, list):
            _base_models = [deepcopy(qcc) for qcc in quacc_classes]
        else:
            _base_models = [deepcopy(quacc_classes)]

        grid_product = list(IT.product(*list(self.param_grid.values())))
        quacc_params = [dict(zip(list(self.param_grid.keys()), vs)) for vs in grid_product]

        self.models = []
        for _base_m, grid in IT.product(_base_models, quacc_params):
            m = deepcopy(_base_m)
            m.set_params(**grid)
            self.models.append(m)

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

    def _fit_model(self, args):
        m, train = args
        t_init = time()
        m.fit(train)
        self._sout(f"training {m.__class__.__name__}({m.q_class.__class__.__name__}) took {time() - t_init:.3f}s")

    def _predict_model_ct(self, args):
        m, val = args
        return m.predict_ct(val.X).flatten()

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=self.val_prop)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.n_val_samples,
            return_type="labelled_collection",
        )

        # training models
        models_fit_args = [(m, v1) for m in self.models]
        qc_parallel(
            self._fit_model,
            models_fit_args,
            n_jobs=self.n_jobs,
            seed=qp.environ.get("_R_SEED", None),
            verbose=self.joblib_verbose,
        )

        # predicting v2 sample cont. tables for each model
        models_cts_args = IT.product(self.models, list(v2_prot()))
        v2_ctss = qc_parallel(
            self._predict_model_cts,
            models_cts_args,
            n_jobs=self.n_jobs,
            seed=qp.environ.get("_R_SEED", None),
            asarray=False,
            verbose=self.joblib_verbose,
        )
        v2_ctss = np.vstack(v2_ctss)
        v2_ctss = np.hstack(np.vsplit(v2_ctss, len(self.models)))

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
