from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np
from matplotlib.pylab import rand
from quapy.method.aggregative import CC, PACC, SLD, BaseQuantifier
from quapy.protocol import UPP, AbstractProtocol, OnLabelledCollectionProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

import quacc as qc
from quacc.environment import env
from quacc.evaluation.report import EvaluationReport
from quacc.method.base import BQAE, MCAE, BaseAccuracyEstimator
from quacc.method.model_selection import (
    GridSearchAE,
    SpiderSearchAE,
)
from quacc.quantification import KDEy


def _param_grid(method, X_fit: np.ndarray):
    match method:
        case "sld_lr":
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "q__recalib": [None, "bcts"],
                "confidence": [
                    None,
                    ["isoft"],
                    ["max_conf", "entropy"],
                    ["max_conf", "entropy", "isoft"],
                ],
            }
        case "sld_rbf":
            _scale = 1.0 / (X_fit.shape[1] * X_fit.var())
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "q__classifier__gamma": _scale * np.logspace(-2, 2, 5),
                "q__recalib": [None, "bcts"],
                "confidence": [
                    None,
                    ["isoft"],
                    ["max_conf", "entropy"],
                    ["max_conf", "entropy", "isoft"],
                ],
            }
        case "pacc":
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "confidence": [None, ["isoft"], ["max_conf", "entropy"]],
            }
        case "cc_lr":
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "confidence": [
                    None,
                    ["isoft"],
                    ["max_conf", "entropy"],
                    ["max_conf", "entropy", "isoft"],
                ],
            }
        case "kde_lr":
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "q__bandwidth": np.linspace(0.01, 0.2, 20),
                "confidence": [None, ["isoft"], ["max_conf", "entropy", "isoft"]],
            }
        case "kde_rbf":
            _scale = 1.0 / (X_fit.shape[1] * X_fit.var())
            return {
                "q__classifier__C": np.logspace(-3, 3, 7),
                "q__classifier__class_weight": [None, "balanced"],
                "q__classifier__gamma": _scale * np.logspace(-2, 2, 5),
                "q__bandwidth": np.linspace(0.01, 0.2, 20),
                "confidence": [None, ["isoft"], ["max_conf", "entropy", "isoft"]],
            }


def evaluation_report(
    estimator: BaseAccuracyEstimator, protocol: AbstractProtocol, method_name=None
) -> EvaluationReport:
    # method_name = inspect.stack()[1].function
    report = EvaluationReport(name=method_name)
    for sample in protocol():
        try:
            e_sample = estimator.extend(sample)
            estim_prev = estimator.estimate(e_sample.eX)
            true_prev = e_sample.e_prevalence()
            acc_score = qc.error.acc(estim_prev)
            row = dict(
                acc_score=acc_score,
                acc=abs(qc.error.acc(true_prev) - acc_score),
            )
            if estim_prev.can_f1():
                f1_score = qc.error.f1(estim_prev)
                row = row | dict(
                    f1_score=f1_score,
                    f1=abs(qc.error.f1(true_prev) - f1_score),
                )
            report.append_row(sample.prevalence(), **row)
        except Exception as e:
            print(f"sample prediction failed for method {method_name}: {e}")
            report.append_row(
                sample.prevalence(),
                acc_score=np.nan,
                acc=np.nan,
                f1_score=np.nan,
                f1=np.nan,
            )

    return report


@dataclass(frozen=True)
class EmptyMethod:
    name: str
    nocall: bool = True

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        pass


@dataclass(frozen=True)
class EvaluationMethod:
    name: str
    q: BaseQuantifier
    est_n: str
    conf: List[str] | str = None
    cf: bool = False  # collapse_false
    gf: bool = False  # group_false
    d: bool = False  # dense

    def get_est(self, c_model):
        match self.est_n:
            case "mul":
                return MCAE(
                    c_model,
                    self.q,
                    confidence=self.conf,
                    collapse_false=self.cf,
                    group_false=self.gf,
                    dense=self.d,
                )
            case "bin":
                return BQAE(
                    c_model,
                    self.q,
                    confidence=self.conf,
                    group_false=self.gf,
                    dense=self.d,
                )

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        est = self.get_est(c_model).fit(validation)
        return evaluation_report(
            estimator=est, protocol=protocol, method_name=self.name
        )


@dataclass(frozen=True)
class EvaluationMethodGridSearch(EvaluationMethod):
    pg: str = "sld"
    search: str = "grid"

    def get_search(self):
        match self.search:
            case "grid":
                return (GridSearchAE, {})
            case "spider" | "spider2":
                return (SpiderSearchAE, dict(best_width=2))
            case "spider3":
                return (SpiderSearchAE, dict(best_width=3))
            case _:
                return GridSearchAE

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        v_train, v_val = validation.split_stratified(0.6, random_state=env._R_SEED)
        _model = self.get_est(c_model)
        _grid = _param_grid(self.pg, X_fit=_model.extend(v_train, prefit=True).X)
        _search_class, _search_params = self.get_search()
        est = _search_class(
            model=_model,
            param_grid=_grid,
            refit=False,
            protocol=UPP(v_val, repeats=100),
            verbose=False,
            **_search_params,
        ).fit(v_train)
        er = evaluation_report(
            estimator=est,
            protocol=protocol,
            method_name=self.name,
        )
        er.fit_score = est.best_score()
        return er


E = EmptyMethod
M = EvaluationMethod
G = EvaluationMethodGridSearch


def __sld_lr():
    return SLD(LogisticRegression())


def __sld_rbf():
    return SLD(SVC(kernel="rbf", probability=True))


def __kde_lr():
    return KDEy(LogisticRegression(), random_state=env._R_SEED)


def __kde_rbf():
    return KDEy(SVC(kernel="rbf", probability=True), random_state=env._R_SEED)


def __sld_lsvc():
    return SLD(LinearSVC())


def __pacc_lr():
    return PACC(LogisticRegression())


def __cc_lr():
    return CC(LogisticRegression())


# fmt: off

__sld_lr_set = [
    M("bin_sld_lr",      __sld_lr(),  "bin"                                       ),
    M("bgf_sld_lr",      __sld_lr(),  "bin",                               gf=True),
    M("mul_sld_lr",      __sld_lr(),  "mul"                                       ),
    M("m3w_sld_lr",      __sld_lr(),  "mul",                               cf=True),
    M("mgf_sld_lr",      __sld_lr(),  "mul",                               gf=True),
    # max_conf sld
    M("bin_sld_lr_mc",   __sld_lr(),  "bin", conf="max_conf",                     ),
    M("bgf_sld_lr_mc",   __sld_lr(),  "bin", conf="max_conf",              gf=True),
    M("mul_sld_lr_mc",   __sld_lr(),  "mul", conf="max_conf",                     ),
    M("m3w_sld_lr_mc",   __sld_lr(),  "mul", conf="max_conf",              cf=True),
    M("mgf_sld_lr_mc",   __sld_lr(),  "mul", conf="max_conf",              gf=True),
    # entropy sld
    M("bin_sld_lr_ne",   __sld_lr(),  "bin", conf="entropy",                      ),
    M("bgf_sld_lr_ne",   __sld_lr(),  "bin", conf="entropy",               gf=True),
    M("mul_sld_lr_ne",   __sld_lr(),  "mul", conf="entropy",                      ),
    M("m3w_sld_lr_ne",   __sld_lr(),  "mul", conf="entropy",               cf=True),
    M("mgf_sld_lr_ne",   __sld_lr(),  "mul", conf="entropy",               gf=True),
    # inverse softmax sld
    M("bin_sld_lr_is",   __sld_lr(),  "bin", conf="isoft",                        ),
    M("bgf_sld_lr_is",   __sld_lr(),  "bin", conf="isoft",                 gf=True),
    M("mul_sld_lr_is",   __sld_lr(),  "mul", conf="isoft",                        ),
    M("m3w_sld_lr_is",   __sld_lr(),  "mul", conf="isoft",                 cf=True),
    M("mgf_sld_lr_is",   __sld_lr(),  "mul", conf="isoft",                 gf=True),
    # max_conf + entropy sld
    M("bin_sld_lr_c",    __sld_lr(),  "bin", conf=["max_conf", "entropy"]         ),
    M("bgf_sld_lr_c",    __sld_lr(),  "bin", conf=["max_conf", "entropy"], gf=True),
    M("mul_sld_lr_c",    __sld_lr(),  "mul", conf=["max_conf", "entropy"]         ),
    M("m3w_sld_lr_c",    __sld_lr(),  "mul", conf=["max_conf", "entropy"], cf=True),
    M("mgf_sld_lr_c",    __sld_lr(),  "mul", conf=["max_conf", "entropy"], gf=True),
    # sld all
    M("bin_sld_lr_a",   __sld_lr(),  "bin", conf=["max_conf", "entropy", "isoft"],         ),
    M("bgf_sld_lr_a",   __sld_lr(),  "bin", conf=["max_conf", "entropy", "isoft"],  gf=True),
    M("mul_sld_lr_a",   __sld_lr(),  "mul", conf=["max_conf", "entropy", "isoft"],         ),
    M("m3w_sld_lr_a",   __sld_lr(),  "mul", conf=["max_conf", "entropy", "isoft"],  cf=True),
    M("mgf_sld_lr_a",   __sld_lr(),  "mul", conf=["max_conf", "entropy", "isoft"],  gf=True),
    # gs sld
    G("bin_sld_lr_gs",   __sld_lr(),  "bin", pg="sld_lr"                          ),
    G("bgf_sld_lr_gs",   __sld_lr(),  "bin", pg="sld_lr",                  gf=True),
    G("mul_sld_lr_gs",   __sld_lr(),  "mul", pg="sld_lr"                          ),
    G("m3w_sld_lr_gs",   __sld_lr(),  "mul", pg="sld_lr",                  cf=True),
    G("mgf_sld_lr_gs",   __sld_lr(),  "mul", pg="sld_lr",                  gf=True),
]

__dense_sld_lr_set = [
    M("d_bin_sld_lr",      __sld_lr(),  "bin", d=True,                                      ),
    M("d_bgf_sld_lr",      __sld_lr(),  "bin", d=True,                               gf=True),
    M("d_mul_sld_lr",      __sld_lr(),  "mul", d=True,                                      ),
    M("d_m3w_sld_lr",      __sld_lr(),  "mul", d=True,                               cf=True),
    M("d_mgf_sld_lr",      __sld_lr(),  "mul", d=True,                               gf=True),
    # max_conf sld
    M("d_bin_sld_lr_mc",   __sld_lr(),  "bin", d=True, conf="max_conf",                     ),
    M("d_bgf_sld_lr_mc",   __sld_lr(),  "bin", d=True, conf="max_conf",              gf=True),
    M("d_mul_sld_lr_mc",   __sld_lr(),  "mul", d=True, conf="max_conf",                     ),
    M("d_m3w_sld_lr_mc",   __sld_lr(),  "mul", d=True, conf="max_conf",              cf=True),
    M("d_mgf_sld_lr_mc",   __sld_lr(),  "mul", d=True, conf="max_conf",              gf=True),
    # entropy sld
    M("d_bin_sld_lr_ne",   __sld_lr(),  "bin", d=True, conf="entropy",                      ),
    M("d_bgf_sld_lr_ne",   __sld_lr(),  "bin", d=True, conf="entropy",               gf=True),
    M("d_mul_sld_lr_ne",   __sld_lr(),  "mul", d=True, conf="entropy",                      ),
    M("d_m3w_sld_lr_ne",   __sld_lr(),  "mul", d=True, conf="entropy",               cf=True),
    M("d_mgf_sld_lr_ne",   __sld_lr(),  "mul", d=True, conf="entropy",               gf=True),
    # inverse softmax sld
    M("d_bin_sld_lr_is",   __sld_lr(),  "bin", d=True, conf="isoft",                        ),
    M("d_bgf_sld_lr_is",   __sld_lr(),  "bin", d=True, conf="isoft",                 gf=True),
    M("d_mul_sld_lr_is",   __sld_lr(),  "mul", d=True, conf="isoft",                        ),
    M("d_m3w_sld_lr_is",   __sld_lr(),  "mul", d=True, conf="isoft",                 cf=True),
    M("d_mgf_sld_lr_is",   __sld_lr(),  "mul", d=True, conf="isoft",                 gf=True),
    # max_conf + entropy sld
    M("d_bin_sld_lr_c",    __sld_lr(),  "bin", d=True, conf=["max_conf", "entropy"]         ),
    M("d_bgf_sld_lr_c",    __sld_lr(),  "bin", d=True, conf=["max_conf", "entropy"], gf=True),
    M("d_mul_sld_lr_c",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy"]         ),
    M("d_m3w_sld_lr_c",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy"], cf=True),
    M("d_mgf_sld_lr_c",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy"], gf=True),
    # sld all
    M("d_bin_sld_lr_a",    __sld_lr(),  "bin", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_bgf_sld_lr_a",    __sld_lr(),  "bin", d=True, conf=["max_conf", "entropy", "isoft"],  gf=True),
    M("d_mul_sld_lr_a",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_m3w_sld_lr_a",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy", "isoft"],  cf=True),
    M("d_mgf_sld_lr_a",    __sld_lr(),  "mul", d=True, conf=["max_conf", "entropy", "isoft"],  gf=True),
    # gs sld
    G("d_bin_sld_lr_gs",   __sld_lr(),  "bin", d=True, pg="sld_lr"                          ),
    G("d_bgf_sld_lr_gs",   __sld_lr(),  "bin", d=True, pg="sld_lr",                  gf=True),
    G("d_mul_sld_lr_gs",   __sld_lr(),  "mul", d=True, pg="sld_lr"                          ),
    G("d_m3w_sld_lr_gs",   __sld_lr(),  "mul", d=True, pg="sld_lr",                  cf=True),
    G("d_mgf_sld_lr_gs",   __sld_lr(),  "mul", d=True, pg="sld_lr",                  gf=True),
]

__dense_sld_rbf_set = [
    M("d_bin_sld_rbf",    __sld_rbf(), "bin", d=True,                                       ),
    M("d_bgf_sld_rbf",    __sld_rbf(), "bin", d=True,                                 gf=True),
    M("d_mul_sld_rbf",    __sld_rbf(), "mul", d=True,                                       ),
    M("d_m3w_sld_rbf",    __sld_rbf(), "mul", d=True,                                 cf=True),
    M("d_mgf_sld_rbf",    __sld_rbf(), "mul", d=True,                                 gf=True),
    # max_conf sld
    M("d_bin_sld_rbf_mc", __sld_rbf(), "bin", d=True, conf="max_conf",                       ),
    M("d_bgf_sld_rbf_mc", __sld_rbf(), "bin", d=True, conf="max_conf",                gf=True),
    M("d_mul_sld_rbf_mc", __sld_rbf(), "mul", d=True, conf="max_conf",                       ),
    M("d_m3w_sld_rbf_mc", __sld_rbf(), "mul", d=True, conf="max_conf",                cf=True),
    M("d_mgf_sld_rbf_mc", __sld_rbf(), "mul", d=True, conf="max_conf",                gf=True),
    # entropy sld
    M("d_bin_sld_rbf_ne", __sld_rbf(), "bin", d=True, conf="entropy",                        ),
    M("d_bgf_sld_rbf_ne", __sld_rbf(), "bin", d=True, conf="entropy",                 gf=True),
    M("d_mul_sld_rbf_ne", __sld_rbf(), "mul", d=True, conf="entropy",                        ),
    M("d_m3w_sld_rbf_ne", __sld_rbf(), "mul", d=True, conf="entropy",                 cf=True),
    M("d_mgf_sld_rbf_ne", __sld_rbf(), "mul", d=True, conf="entropy",                 gf=True),
    # inverse softmax sld
    M("d_bin_sld_rbf_is", __sld_rbf(), "bin", d=True, conf="isoft",                          ),
    M("d_bgf_sld_rbf_is", __sld_rbf(), "bin", d=True, conf="isoft",                   gf=True),
    M("d_mul_sld_rbf_is", __sld_rbf(), "mul", d=True, conf="isoft",                          ),
    M("d_m3w_sld_rbf_is", __sld_rbf(), "mul", d=True, conf="isoft",                   cf=True),
    M("d_mgf_sld_rbf_is", __sld_rbf(), "mul", d=True, conf="isoft",                   gf=True),
    # max_conf + entropy sld
    M("d_bin_sld_rbf_c",  __sld_rbf(), "bin", d=True, conf=["max_conf", "entropy"]           ),
    M("d_bgf_sld_rbf_c",  __sld_rbf(), "bin", d=True, conf=["max_conf", "entropy"],   gf=True),
    M("d_mul_sld_rbf_c",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy"]           ),
    M("d_m3w_sld_rbf_c",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy"],   cf=True),
    M("d_mgf_sld_rbf_c",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy"],   gf=True),
    # sld all
    M("d_bin_sld_rbf_a",  __sld_rbf(), "bin", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_bgf_sld_rbf_a",  __sld_rbf(), "bin", d=True, conf=["max_conf", "entropy", "isoft"],  gf=True),
    M("d_mul_sld_rbf_a",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_m3w_sld_rbf_a",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],  cf=True),
    M("d_mgf_sld_rbf_a",  __sld_rbf(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],  gf=True),
    # gs sld
    G("d_bin_sld_rbf_gs", __sld_rbf(), "bin", d=True, pg="sld_rbf", search="grid",        ),
    G("d_bgf_sld_rbf_gs", __sld_rbf(), "bin", d=True, pg="sld_rbf", search="grid", gf=True),
    G("d_mul_sld_rbf_gs", __sld_rbf(), "mul", d=True, pg="sld_rbf", search="grid",        ),
    G("d_m3w_sld_rbf_gs", __sld_rbf(), "mul", d=True, pg="sld_rbf", search="grid", cf=True),
    G("d_mgf_sld_rbf_gs", __sld_rbf(), "mul", d=True, pg="sld_rbf", search="grid", gf=True),
]

__kde_lr_set = [
    # base kde
    M("bin_kde_lr",    __kde_lr(), "bin"                                       ),
    M("mul_kde_lr",    __kde_lr(), "mul"                                       ),
    M("m3w_kde_lr",    __kde_lr(), "mul",                               cf=True),
    # max_conf kde
    M("bin_kde_lr_mc", __kde_lr(), "bin", conf="max_conf",                     ),
    M("mul_kde_lr_mc", __kde_lr(), "mul", conf="max_conf",                     ),
    M("m3w_kde_lr_mc", __kde_lr(), "mul", conf="max_conf",              cf=True),
    # entropy kde
    M("bin_kde_lr_ne", __kde_lr(), "bin", conf="entropy",                      ),
    M("mul_kde_lr_ne", __kde_lr(), "mul", conf="entropy",                      ),
    M("m3w_kde_lr_ne", __kde_lr(), "mul", conf="entropy",               cf=True),
    # inverse softmax kde
    M("bin_kde_lr_is", __kde_lr(), "bin", conf="isoft",                        ),
    M("mul_kde_lr_is", __kde_lr(), "mul", conf="isoft",                        ),
    M("m3w_kde_lr_is", __kde_lr(), "mul", conf="isoft",                 cf=True),
    # max_conf + entropy kde
    M("bin_kde_lr_c",  __kde_lr(), "bin", conf=["max_conf", "entropy"]         ),
    M("mul_kde_lr_c",  __kde_lr(), "mul", conf=["max_conf", "entropy"]         ),
    M("m3w_kde_lr_c",  __kde_lr(), "mul", conf=["max_conf", "entropy"], cf=True),
    # kde all
    M("bin_kde_lr_a",  __kde_lr(), "bin", conf=["max_conf", "entropy", "isoft"],         ),
    M("mul_kde_lr_a",  __kde_lr(), "mul", conf=["max_conf", "entropy", "isoft"],         ),
    M("m3w_kde_lr_a",  __kde_lr(), "mul", conf=["max_conf", "entropy", "isoft"],  cf=True),
    # gs kde
    G("bin_kde_lr_gs", __kde_lr(), "bin", pg="kde_lr", search="grid"         ),
    G("mul_kde_lr_gs", __kde_lr(), "mul", pg="kde_lr", search="grid"         ),
    G("m3w_kde_lr_gs", __kde_lr(), "mul", pg="kde_lr", search="grid", cf=True),
]

__dense_kde_lr_set = [
    # base kde
    M("d_bin_kde_lr",    __kde_lr(), "bin", d=True,                                      ),
    M("d_mul_kde_lr",    __kde_lr(), "mul", d=True,                                      ),
    M("d_m3w_kde_lr",    __kde_lr(), "mul", d=True,                               cf=True),
    # max_conf kde                       
    M("d_bin_kde_lr_mc", __kde_lr(), "bin", d=True, conf="max_conf",                     ),
    M("d_mul_kde_lr_mc", __kde_lr(), "mul", d=True, conf="max_conf",                     ),
    M("d_m3w_kde_lr_mc", __kde_lr(), "mul", d=True, conf="max_conf",              cf=True),
    # entropy kde                        
    M("d_bin_kde_lr_ne", __kde_lr(), "bin", d=True, conf="entropy",                      ),
    M("d_mul_kde_lr_ne", __kde_lr(), "mul", d=True, conf="entropy",                      ),
    M("d_m3w_kde_lr_ne", __kde_lr(), "mul", d=True, conf="entropy",               cf=True),
    # inverse softmax kde                  d=True,
    M("d_bin_kde_lr_is", __kde_lr(), "bin", d=True, conf="isoft",                        ),
    M("d_mul_kde_lr_is", __kde_lr(), "mul", d=True, conf="isoft",                        ),
    M("d_m3w_kde_lr_is", __kde_lr(), "mul", d=True, conf="isoft",                 cf=True),
    # max_conf + entropy kde               
    M("d_bin_kde_lr_c",  __kde_lr(), "bin", d=True, conf=["max_conf", "entropy"]         ),
    M("d_mul_kde_lr_c",  __kde_lr(), "mul", d=True, conf=["max_conf", "entropy"]         ),
    M("d_m3w_kde_lr_c",  __kde_lr(), "mul", d=True, conf=["max_conf", "entropy"], cf=True),
    # kde all
    M("d_bin_kde_lr_a",  __kde_lr(), "bin", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_mul_kde_lr_a",  __kde_lr(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_m3w_kde_lr_a",  __kde_lr(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],  cf=True),
    # gs kde                             
    G("d_bin_kde_lr_gs", __kde_lr(), "bin", d=True, pg="kde_lr", search="grid"            ),
    G("d_mul_kde_lr_gs", __kde_lr(), "mul", d=True, pg="kde_lr", search="grid"            ),
    G("d_m3w_kde_lr_gs", __kde_lr(), "mul", d=True, pg="kde_lr", search="grid",    cf=True),
]

__dense_kde_rbf_set = [
    # base kde
    M("d_bin_kde_rbf",    __kde_rbf(), "bin", d=True,                                       ),
    M("d_mul_kde_rbf",    __kde_rbf(), "mul", d=True,                                       ),
    M("d_m3w_kde_rbf",    __kde_rbf(), "mul", d=True,                                cf=True),
    # max_conf kde
    M("d_bin_kde_rbf_mc", __kde_rbf(), "bin", d=True, conf="max_conf",                      ),
    M("d_mul_kde_rbf_mc", __kde_rbf(), "mul", d=True, conf="max_conf",                      ),
    M("d_m3w_kde_rbf_mc", __kde_rbf(), "mul", d=True, conf="max_conf",               cf=True),
    # entropy kde
    M("d_bin_kde_rbf_ne", __kde_rbf(), "bin", d=True, conf="entropy",                       ),
    M("d_mul_kde_rbf_ne", __kde_rbf(), "mul", d=True, conf="entropy",                       ),
    M("d_m3w_kde_rbf_ne", __kde_rbf(), "mul", d=True, conf="entropy",                cf=True),
    # inverse softmax kde
    M("d_bin_kde_rbf_is", __kde_rbf(), "bin", d=True, conf="isoft",                         ),
    M("d_mul_kde_rbf_is", __kde_rbf(), "mul", d=True, conf="isoft",                         ),
    M("d_m3w_kde_rbf_is", __kde_rbf(), "mul", d=True, conf="isoft",                  cf=True),
    # max_conf + entropy kde
    M("d_bin_kde_rbf_c",  __kde_rbf(), "bin", d=True, conf=["max_conf", "entropy"]          ),
    M("d_mul_kde_rbf_c",  __kde_rbf(), "mul", d=True, conf=["max_conf", "entropy"]          ),
    M("d_m3w_kde_rbf_c",  __kde_rbf(), "mul", d=True, conf=["max_conf", "entropy"],  cf=True),
    # kde all
    M("d_bin_kde_rbf_a",  __kde_rbf(), "bin", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_mul_kde_rbf_a",  __kde_rbf(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],         ),
    M("d_m3w_kde_rbf_a",  __kde_rbf(), "mul", d=True, conf=["max_conf", "entropy", "isoft"],  cf=True),
    # gs kde
    G("d_bin_kde_rbf_gs", __kde_rbf(), "bin", d=True, pg="kde_rbf", search="spider"          ),
    G("d_mul_kde_rbf_gs", __kde_rbf(), "mul", d=True, pg="kde_rbf", search="spider"          ),
    G("d_m3w_kde_rbf_gs", __kde_rbf(), "mul", d=True, pg="kde_rbf", search="spider", cf=True),
]

__cc_lr_set = [
    # base cc
    M("bin_cc_lr",    __cc_lr(), "bin"                                       ),
    M("mul_cc_lr",    __cc_lr(), "mul"                                       ),
    M("m3w_cc_lr",    __cc_lr(), "mul",                               cf=True),
    # max_conf cc
    M("bin_cc_lr_mc", __cc_lr(), "bin", conf="max_conf",                     ),
    M("mul_cc_lr_mc", __cc_lr(), "mul", conf="max_conf",                     ),
    M("m3w_cc_lr_mc", __cc_lr(), "mul", conf="max_conf",              cf=True),
    # entropy cc
    M("bin_cc_lr_ne", __cc_lr(), "bin", conf="entropy",                      ),
    M("mul_cc_lr_ne", __cc_lr(), "mul", conf="entropy",                      ),
    M("m3w_cc_lr_ne", __cc_lr(), "mul", conf="entropy",               cf=True),
    # inverse softmax cc
    M("bin_cc_lr_is", __cc_lr(), "bin", conf="isoft",                        ),
    M("mul_cc_lr_is", __cc_lr(), "mul", conf="isoft",                        ),
    M("m3w_cc_lr_is", __cc_lr(), "mul", conf="isoft",                 cf=True),
    # max_conf + entropy cc
    M("bin_cc_lr_c",  __cc_lr(), "bin", conf=["max_conf", "entropy"]         ),
    M("mul_cc_lr_c",  __cc_lr(), "mul", conf=["max_conf", "entropy"]         ),
    M("m3w_cc_lr_c",  __cc_lr(), "mul", conf=["max_conf", "entropy"], cf=True),
    # cc all
    M("bin_cc_lr_a",  __cc_lr(), "bin", conf=["max_conf", "entropy", "isoft"],         ),
    M("mul_cc_lr_a",  __cc_lr(), "mul", conf=["max_conf", "entropy", "isoft"],         ),
    M("m3w_cc_lr_a",  __cc_lr(), "mul", conf=["max_conf", "entropy", "isoft"],  cf=True),
    # gs cc
    G("bin_cc_lr_gs", __cc_lr(), "bin", pg="cc_lr", search="grid"         ),
    G("mul_cc_lr_gs", __cc_lr(), "mul", pg="cc_lr", search="grid"         ),
    G("m3w_cc_lr_gs", __cc_lr(), "mul", pg="cc_lr", search="grid", cf=True),
]

__ms_set = [
    E("sld_lr_gs"),
    E("kde_lr_gs"),
    E("cc_lr_gs"),
    E("QuAcc"),
]

# fmt: on

__methods_set = (
    __sld_lr_set
    + __dense_sld_lr_set
    + __dense_sld_rbf_set
    + __kde_lr_set
    + __dense_kde_lr_set
    + __dense_kde_rbf_set
    + __cc_lr_set
    + __ms_set
)

_methods = {m.name: m for m in __methods_set}
