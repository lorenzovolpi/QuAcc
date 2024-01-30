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
        _dict = alt._alts | method._methods | baseline._baselines

        match e:
            case "__all":
                e = list(_dict.keys())
            case "__baselines":
                e = list(baseline._baselines.keys())

        if isinstance(e, str):
            try:
                return {e: _dict[e]}
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list):
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
