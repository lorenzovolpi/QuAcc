import numpy as np
from quapy.method.aggregative import PACC, SLD
from quapy.protocol import UPP, AbstractProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import quacc as qc
from quacc.evaluation.report import EvaluationReport
from quacc.method.model_selection import GridSearchAE

from ..method.base import BQAE, MCAE, BaseAccuracyEstimator

_param_grid = {
    "sld": {
        "q__classifier__C": np.logspace(-3, 3, 7),
        "q__classifier__class_weight": [None, "balanced"],
        "q__recalib": [None, "bcts"],
        "confidence": [["isoft"], ["max_conf", "entropy"]],
    },
    "pacc": {
        "q__classifier__C": np.logspace(-3, 3, 7),
        "q__classifier__class_weight": [None, "balanced"],
        "confidence": [["isoft"], ["max_conf", "entropy"]],
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


class EvaluationMethod:
    def __init__(self, name, q, est_c, conf=None, cf=False):
        self.name = name
        self.__name__ = name
        self.q = q
        self.est_c = est_c
        self.conf = conf
        self.cf = cf

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        est = self.est_c(
            c_model,
            self.q,
            confidence=self.conf,
            collapse_false=self.cf,
        ).fit(validation)
        return evaluation_report(
            estimator=est, protocol=protocol, method_name=self.name
        )


class EvaluationMethodGridSearch(EvaluationMethod):
    def __init__(self, name, q, est_c, cf=False, pg="sld"):
        super().__init__(name, q, est_c, cf=cf)
        self.pg = pg

    def __call__(self, c_model, validation, protocol) -> EvaluationReport:
        v_train, v_val = validation.split_stratified(0.6, random_state=0)
        model = self.est_c(c_model, self.q, collapse_false=self.cf)
        __grid = _param_grid.get(self.pg, {})
        est = GridSearchAE(
            model=model,
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


def __sld_lsvc():
    return SLD(LinearSVC())


def __pacc_lr():
    return PACC(LogisticRegression())


# fmt: off
__methods_set = [
    # base sld
    M("bin_sld",     __sld_lr(),  BQAE                                       ),
    M("mul_sld",     __sld_lr(),  MCAE                                       ),
    M("m3w_sld",     __sld_lr(),  MCAE,                               cf=True),
    # max_conf + entropy sld
    M("binc_sld",    __sld_lr(),  BQAE, conf=["max_conf", "entropy"]         ),
    M("mulc_sld",    __sld_lr(),  MCAE, conf=["max_conf", "entropy"]         ),
    M("m3wc_sld",    __sld_lr(),  MCAE, conf=["max_conf", "entropy"], cf=True),
    # max_conf sld
    M("binmc_sld",   __sld_lr(),  BQAE, conf="max_conf",                     ),
    M("mulmc_sld",   __sld_lr(),  MCAE, conf="max_conf",                     ),
    M("m3wmc_sld",   __sld_lr(),  MCAE, conf="max_conf",              cf=True),
    # entropy sld
    M("binne_sld",   __sld_lr(),  BQAE, conf="entropy",                      ),
    M("mulne_sld",   __sld_lr(),  MCAE, conf="entropy",                      ),
    M("m3wne_sld",   __sld_lr(),  MCAE, conf="entropy",               cf=True),
    # inverse softmax sld
    M("binis_sld",   __sld_lr(),  BQAE, conf="isoft",                        ),
    M("mulis_sld",   __sld_lr(),  MCAE, conf="isoft",                        ),
    M("m3wis_sld",   __sld_lr(),  MCAE, conf="isoft",                 cf=True),
    # inverse softmax sld
    M("binis_pacc",  __pacc_lr(), BQAE, conf="isoft",                        ),
    M("mulis_pacc",  __pacc_lr(), MCAE, conf="isoft",                        ),
    M("m3wis_pacc",  __pacc_lr(), MCAE, conf="isoft",                 cf=True),
    # gs sld
    G("bin_sld_gs",  __sld_lr(),  BQAE, pg="sld"                             ),
    G("mul_sld_gs",  __sld_lr(),  MCAE, pg="sld"                             ),
    G("m3w_sld_gs",  __sld_lr(),  MCAE, pg="sld",                     cf=True),
    # gs pacc
    G("bin_pacc_gs", __pacc_lr(), BQAE, pg="pacc"                            ),
    G("mul_pacc_gs", __pacc_lr(), MCAE, pg="pacc"                            ),
    G("m3w_pacc_gs", __pacc_lr(), MCAE, pg="pacc",                    cf=True),
]
# fmt: on

_methods = {m.name: m for m in __methods_set}
