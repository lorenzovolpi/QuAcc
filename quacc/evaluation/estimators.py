from typing import List

import numpy as np

from quacc.evaluation import baseline, method, alt


class CompEstimatorFunc_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return list(self.ce._CompEstimator__get(e).values())[0]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).values())


class CompEstimatorName_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return list(self.ce._CompEstimator__get(e).keys())[0]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).keys())

    def sort(self, e: List[str]):
        return list(self.ce._CompEstimator__get(e, get_ref=False).keys())

    @property
    def all(self):
        return list(self.ce._CompEstimator__get("__all").keys())

    @property
    def baselines(self):
        return list(self.ce._CompEstimator__get("__baselines").keys())


class CompEstimator:
    def __get(cls, e: str | List[str], get_ref=True):
        _dict = alt._alts | baseline._baselines | method._methods

        if isinstance(e, str) and e == "__all":
            e = list(_dict.keys())
        if isinstance(e, str) and e == "__baselines":
            e = list(baseline._baselines.keys())

        if isinstance(e, str):
            try:
                return {e: _dict[e]}
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list) or isinstance(e, np.ndarray):
            _subtr = np.setdiff1d(e, list(_dict.keys()))
            if len(_subtr) > 0:
                raise KeyError(
                    f"Invalid estimator: estimator {_subtr[0]} does not exist"
                )

            e_fun = {k: fun for k, fun in _dict.items() if k in e}
            if get_ref and "ref" not in e:
                e_fun["ref"] = _dict["ref"]
            elif not get_ref and "ref" in e:
                del e_fun["ref"]

            return e_fun

    @property
    def name(self):
        return CompEstimatorName_(self)

    @property
    def func(self):
        return CompEstimatorFunc_(self)


CE = CompEstimator()

_renames = {
    "bin_sld_lr": "(2x2)_SLD_LR",
    "mul_sld_lr": "(1x4)_SLD_LR",
    "m3w_sld_lr": "(1x3)_SLD_LR",
    "d_bin_sld_lr": "d_(2x2)_SLD_LR",
    "d_mul_sld_lr": "d_(1x4)_SLD_LR",
    "d_m3w_sld_lr": "d_(1x3)_SLD_LR",
    "d_bin_sld_rbf": "(2x2)_SLD_RBF",
    "d_mul_sld_rbf": "(1x4)_SLD_RBF",
    "d_m3w_sld_rbf": "(1x3)_SLD_RBF",
    "sld_lr_gs": "MS_SLD_LR",
    "bin_kde_lr": "(2x2)_KDEy_LR",
    "mul_kde_lr": "(1x4)_KDEy_LR",
    "m3w_kde_lr": "(1x3)_KDEy_LR",
    "d_bin_kde_lr": "d_(2x2)_KDEy_LR",
    "d_mul_kde_lr": "d_(1x4)_KDEy_LR",
    "d_m3w_kde_lr": "d_(1x3)_KDEy_LR",
    "bin_cc_lr": "(2x2)_CC_LR",
    "mul_cc_lr": "(1x4)_CC_LR",
    "m3w_cc_lr": "(1x3)_CC_LR",
    "kde_lr_gs": "MS_KDEy_LR",
    "cc_lr_gs": "MS_CC_LR",
    "atc_mc": "ATC",
    "doc": "DoC",
    "mandoline": "Mandoline",
    "rca": "RCA",
    "rca_star": "RCA*",
    "naive": "Naive",
}
