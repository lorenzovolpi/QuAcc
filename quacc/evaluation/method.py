from dataclasses import dataclass
from typing import List

import numpy as np
from quapy.method.aggregative import PACC, SLD, BaseQuantifier
from quapy.protocol import UPP, AbstractProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import quacc as qc
from quacc.environment import env
from quacc.evaluation.report import EvaluationReport
from quacc.method.base import BQAE, MCAE, BaseAccuracyEstimator
from quacc.method.model_selection import (
    GridSearchAE,
    HalvingSearchAE,
    RandomizedSearchAE,
    SpiderSearchAE,
)
from quacc.quantification import KDEy

_param_grid = {
    "sld": {
        "q__classifier__C": np.logspace(-3, 3, 7),
        "q__classifier__class_weight": [None, "balanced"],
        "q__recalib": [None, "bcts"],
        # "q__recalib": [None],
        "confidence": [None, ["isoft"], ["max_conf", "entropy"]],
    },
    "pacc": {
        "q__classifier__C": np.logspace(-3, 3, 7),
        "q__classifier__class_weight": [None, "balanced"],
        "confidence": [None, ["isoft"], ["max_conf", "entropy"]],
    },
    "kde": {
        "q__classifier__C": np.logspace(-3, 3, 7),
        "q__classifier__class_weight": [None, "balanced"],
        # "q__classifier__class_weight": [None],
        "q__bandwidth": np.linspace(0.01, 0.2, 20),
        "confidence": [None, ["isoft"]],
        # "confidence": [None],
    },
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
            acc_score = qc.error.acc(estim_prev)
            f1_score = qc.error.f1(estim_prev)
            report.append_row(
                sample.prevalence(),
                acc_score=acc_score,
                acc=abs(qc.error.acc(e_sample.prevalence()) - acc_score),
                f1_score=f1_score,
                f1=abs(qc.error.f1(e_sample.prevalence()) - f1_score),
            )
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
class EvaluationMethod:
    name: str
    q: BaseQuantifier
    est_n: str
    conf: List[str] | str = None
    cf: bool = False

    def get_est(self, c_model):
        match self.est_n:
            case "mul":
                return MCAE(
                    c_model,
                    self.q,
                    confidence=self.conf,
                    collapse_false=self.cf,
                )
            case "bin":
                return BQAE(c_model, self.q, confidence=self.conf)

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
                return GridSearchAE
            case "spider":
                return SpiderSearchAE
            case _:
                return GridSearchAE

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        v_train, v_val = validation.split_stratified(0.6, random_state=env._R_SEED)
        __grid = _param_grid.get(self.pg, {})
        _search_class = self.get_search()
        est = _search_class(
            model=self.get_est(c_model),
            param_grid=__grid,
            refit=False,
            protocol=UPP(v_val, repeats=100),
            verbose=False,
        ).fit(v_train)
        return evaluation_report(
            estimator=est,
            protocol=protocol,
            method_name=self.name,
        )


M = EvaluationMethod
G = EvaluationMethodGridSearch


def __sld_lr():
    return SLD(LogisticRegression())


def __kde_lr():
    return KDEy(LogisticRegression(), random_state=env._R_SEED)


def __sld_lsvc():
    return SLD(LinearSVC())


def __pacc_lr():
    return PACC(LogisticRegression())


# fmt: off
__methods_set = [
    # base sld
    M("bin_sld",     __sld_lr(),  "bin"                                       ),
    M("mul_sld",     __sld_lr(),  "mul"                                       ),
    M("m3w_sld",     __sld_lr(),  "mul",                               cf=True),
    # max_conf + entropy sld
    M("binc_sld",    __sld_lr(),  "bin", conf=["max_conf", "entropy"]         ),
    M("mulc_sld",    __sld_lr(),  "mul", conf=["max_conf", "entropy"]         ),
    M("m3wc_sld",    __sld_lr(),  "mul", conf=["max_conf", "entropy"], cf=True),
    # max_conf sld
    M("binmc_sld",   __sld_lr(),  "bin", conf="max_conf",                     ),
    M("mulmc_sld",   __sld_lr(),  "mul", conf="max_conf",                     ),
    M("m3wmc_sld",   __sld_lr(),  "mul", conf="max_conf",              cf=True),
    # entropy sld
    M("binne_sld",   __sld_lr(),  "bin", conf="entropy",                      ),
    M("mulne_sld",   __sld_lr(),  "mul", conf="entropy",                      ),
    M("m3wne_sld",   __sld_lr(),  "mul", conf="entropy",               cf=True),
    # inverse softmax sld
    M("binis_sld",   __sld_lr(),  "bin", conf="isoft",                        ),
    M("mulis_sld",   __sld_lr(),  "mul", conf="isoft",                        ),
    M("m3wis_sld",   __sld_lr(),  "mul", conf="isoft",                 cf=True),
    # gs sld
    G("bin_sld_gs",  __sld_lr(),  "bin", pg="sld"                             ),
    G("mul_sld_gs",  __sld_lr(),  "mul", pg="sld"                             ),
    G("m3w_sld_gs",  __sld_lr(),  "mul", pg="sld",                     cf=True),

    # base kde
    M("bin_kde",     __kde_lr(),  "bin"                                       ),
    M("mul_kde",     __kde_lr(),  "mul"                                       ),
    M("m3w_kde",     __kde_lr(),  "mul",                               cf=True),
    # max_conf + entropy kde
    M("binc_kde",    __kde_lr(),  "bin", conf=["max_conf", "entropy"]         ),
    M("mulc_kde",    __kde_lr(),  "mul", conf=["max_conf", "entropy"]         ),
    M("m3wc_kde",    __kde_lr(),  "mul", conf=["max_conf", "entropy"], cf=True),
    # max_conf kde
    M("binmc_kde",   __kde_lr(),  "bin", conf="max_conf",                     ),
    M("mulmc_kde",   __kde_lr(),  "mul", conf="max_conf",                     ),
    M("m3wmc_kde",   __kde_lr(),  "mul", conf="max_conf",              cf=True),
    # entropy kde
    M("binne_kde",   __kde_lr(),  "bin", conf="entropy",                      ),
    M("mulne_kde",   __kde_lr(),  "mul", conf="entropy",                      ),
    M("m3wne_kde",   __kde_lr(),  "mul", conf="entropy",               cf=True),
    # inverse softmax kde
    M("binis_kde",   __kde_lr(),  "bin", conf="isoft",                        ),
    M("mulis_kde",   __kde_lr(),  "mul", conf="isoft",                        ),
    M("m3wis_kde",   __kde_lr(),  "mul", conf="isoft",                 cf=True),
    # gs kde
    G("bin_kde_gs",  __kde_lr(),  "bin", pg="kde", search="spider"            ),
    G("mul_kde_gs",  __kde_lr(),  "mul", pg="kde", search="spider"            ),
    G("m3w_kde_gs",  __kde_lr(),  "mul", pg="kde", search="spider",    cf=True),
]
# fmt: on

_methods = {m.name: m for m in __methods_set}
