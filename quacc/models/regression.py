import itertools as IT
from copy import deepcopy
from time import time

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import SLD, AggregativeQuantifier, BaseQuantifier
from quapy.protocol import UPP, AbstractProtocol
from sklearn.linear_model import LinearRegression, LogisticRegression

import quacc as qc
from quacc.models.base import CAP
from quacc.models.cont_table import N2E, QuAcc
from quacc.models.direct import ATC, CAPDirect, DoC
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.models.utils import get_posteriors_from_h, max_conf, max_inverse_softmax, neg_entropy
from quacc.utils.commons import parallel as qc_parallel


class ReQua(CAPDirect):
    def __init__(
        self,
        h,
        acc_fn,
        reg_model,
        quacc_classes: QuAcc | list[QuAcc],
        param_grid: dict,
        sample_size,
        n_val_samples=500,
        val_prop=0.5,
        clip_vals=(0, 1),
        add_conf=False,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(h, acc_fn)
        self.param_grid = param_grid
        self.reg_model = reg_model
        self._build_models(quacc_classes)
        self.sample_size = sample_size
        self.n_val_samples = n_val_samples
        self.val_prop = val_prop
        self.clip_vals = clip_vals
        self.add_conf = add_conf
        self.n_jobs = qc.commons.get_njobs(n_jobs)
        self.verbose = verbose
        self.joblib_verbose = 10 if verbose else 0

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

    def fit_regression(self, feats_list, accs):
        reg_feats = np.hstack(feats_list)
        self.reg_model.fit(reg_feats, accs)

    def predict_regression(self, feats_list: list[np.ndarray]):
        test_feats = np.hstack(feats_list)
        if test_feats.ndim == 1:
            test_feats = test_feats.reshape(1, -1)
        pred_acc = self.reg_model.predict(test_feats)
        return pred_acc

    def _fit_quacc_models(self, val, parallel=True):
        def _fit_model(args):
            m, train = args
            return m.fit(train)

        # training models
        models_fit_args = [(m, val) for m in self.models]
        if parallel:
            self.models = qc_parallel(
                _fit_model,
                models_fit_args,
                n_jobs=self.n_jobs,
                seed=qp.environ.get("_R_SEED", None),
                verbose=self.joblib_verbose,
            )
        else:
            self.models = [_fit_model(arg) for arg in models_fit_args]

    def _get_quacc_feats(self, X):
        def predict_cts(m, _X):
            return m.predict_ct(_X).flatten()

        cts = np.hstack([predict_cts(m, X) for m in self.models])
        return cts

    def _get_batch_quacc_feats(self, prot, parallel=True):
        def _predict_model_cts(args):
            m, _prot = args
            cts = np.vstack([m.predict_ct(sigma_i.X).flatten() for sigma_i in _prot()])

            return cts

        # predicting v2 sample cont. tables for each model
        models_cts_args = [(m, prot) for m in self.models]
        if parallel:
            v2_ctss = qc_parallel(
                _predict_model_cts,
                models_cts_args,
                n_jobs=self.n_jobs,
                seed=qp.environ.get("_R_SEED", None),
                asarray=False,
                verbose=self.joblib_verbose,
                batch_size=round(len(self.models) / (self.n_jobs * 2)),
            )
        else:
            v2_ctss = [_predict_model_cts(arg) for arg in models_cts_args]

        v2_ctss = np.hstack(v2_ctss)

        return v2_ctss

    def _get_linear_feats(self, X):
        P = get_posteriors_from_h(self.h, X)
        conf_fns = [
            max_conf,
            neg_entropy,
            max_inverse_softmax,
        ]
        lin_feats = np.hstack([fn(P, keepdims=True) for fn in conf_fns]).mean(axis=0)
        return lin_feats

    def _get_batch_linear_feats(self, prot):
        lin_feats = np.vstack([self._get_linear_feats(sigma_i.X) for sigma_i in prot()])
        return lin_feats

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=self.val_prop)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.n_val_samples,
            return_type="labelled_collection",
        )

        # train models used to generate features

        t_fit_init = time()
        self._fit_quacc_models(v1)
        self._sout(f"training quacc models took {time() - t_fit_init:.3f}s")

        # compute features to train the regressor

        features = []

        t_ct_init = time()
        quacc_feats = self._get_batch_quacc_feats(v2_prot)
        features.append(quacc_feats)
        self._sout(f"generating quacc features took {time() - t_ct_init:.3f}s")

        if self.add_conf:
            t_lin_init = time()
            lin_feats = self._get_batch_linear_feats(v2_prot)
            features.append(lin_feats)
            self._sout(f"generating linear features took {time() - t_lin_init:.3f}s")

        # compute true accs as targets for the regressor

        v2_accs = np.asarray([self.true_acc(v2_i) for v2_i in v2_prot()])

        # train regression model

        t_init = time()
        self.fit_regression(features, v2_accs)
        self._sout(f"training reg_model took {time() - t_init:.3f}s")

        return self

    def predict(self, X, oracle_prev=None) -> float:
        features = []
        features.append(self._get_quacc_feats(X))
        if self.add_conf:
            features.append(self._get_linear_feats(X))

        acc_pred = self.predict_regression(features)

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred[0]

    def batch_predict(self, prot: AbstractProtocol, oracle_prevs=None) -> list[float]:
        t_bpred_init = time()

        features = []

        quacc_feats = self._get_batch_quacc_feats(prot)
        features.append(quacc_feats)

        if self.add_conf:
            lin_feats = self._get_batch_linear_feats(prot)
            features.append(lin_feats)

        acc_pred = self.predict_regression(features)
        self._sout(f"batch prediction took {time() - t_bpred_init:.3f}s")

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred.tolist()


class reDAN(CAPDirect):
    def __init__(
        self,
        h,
        acc_fn,
        reg_model,
        q_class: BaseQuantifier,
        sample_size,
        n_val_samples=500,
        add_n2e_opt=False,
        val_prop=0.5,
        clip_vals=(0, 1),
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(h, acc_fn)
        self.reg_model = reg_model
        self.q_class = q_class
        self.sample_size = sample_size
        self.n_val_samples = n_val_samples
        self.add_n2e_opt = add_n2e_opt
        self.val_prop = val_prop
        self.clip_vals = clip_vals
        self.n_jobs = qc.commons.get_njobs(n_jobs)
        self.verbose = verbose
        self.joblib_verbose = 10 if verbose else 0

    def _fit_models(self, val: LabelledCollection):
        n2e = N2E(self.h, self.acc, self.q_class, reuse_h=True).fit(val)
        doc = DoC(self.h, self.acc, self.sample_size).fit(val)
        atc = ATC(self.h, self.acc, scoring_fn="maxconf").fit(val)
        self.models = [n2e, doc, atc]

        if self.add_n2e_opt:
            _params = {
                "q_class__classifier__C": np.logspace(-3, 3, 7),
                "q_class__classifier__class_weight": [None, "balanced"],
            }
            v11, v12 = val.split_stratified(self.val_prop, random_state=qp.environ["_R_SEED"])
            v_prot = UPP(v12, self.sample_size, repeats=100, random_state=qp.environ["_R_SEED"])
            n2e_opt = GSCAP(deepcopy(self.n2e), _params, v_prot, self.acc).fit(v11)
            self.models.append(n2e_opt)

    def _get_models_feats(self, X):
        preds = np.hstack([m.predict(X) for m in self.models])
        return preds

    def fit_regression(self, feats, accs):
        self.reg_model.fit(feats, accs)

    def predict_regression(self, feats):
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        pred_acc = self.reg_model.predict(feats)
        return pred_acc

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=self.val_prop)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.n_val_samples,
            return_type="labelled_collection",
        )

        self._fit_models(v1)
        feats = np.vstack([self._get_models_feats(Ui.X) for Ui in v2_prot()])

        v2_accs = np.asarray([self.true_acc(v2_i) for v2_i in v2_prot()])

        self.reg_model.fit(feats, v2_accs)

        return self

    def predict(self, X, oracle_prev=None) -> float:
        feats = self._get_models_feats(X)
        acc_pred = self.predict_regression(feats)

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred[0]
