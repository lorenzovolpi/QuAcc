import itertools
import math
import os
from copy import deepcopy
from time import time
from typing import Callable, Union

import numpy as np
from joblib import Parallel
from quapy.data import LabelledCollection
from quapy.protocol import (
    AbstractProtocol,
    OnLabelledCollectionProtocol,
)

import quacc as qc
import quacc.error
from quacc.legacy.data import ExtendedCollection
from quacc.legacy.evaluation.evaluate import evaluate
from quacc.legacy.method.base import (
    BaseAccuracyEstimator,
)
from quacc.logger import logger


class GridSearchAE(BaseAccuracyEstimator):
    def __init__(
        self,
        model: BaseAccuracyEstimator,
        param_grid: dict,
        protocol: AbstractProtocol,
        error: Union[Callable, str] = qc.error.maccd,
        refit=True,
        # timeout=-1,
        n_jobs=None,
        verbose=False,
    ):
        self.model = model
        self.param_grid = self.__normalize_params(param_grid)
        self.protocol = protocol
        self.refit = refit
        # self.timeout = timeout
        self.n_jobs = qc._get_njobs(n_jobs)
        self.verbose = verbose
        self.__check_error(error)
        assert isinstance(protocol, AbstractProtocol), "unknown protocol"

    def _sout(self, msg, level=0):
        if level > 0 or self.verbose:
            print(f"[{self.__class__.__name__}@{self.model.__class__.__name__}]: {msg}")

    def __normalize_params(self, params):
        __remap = {}
        for key in params.keys():
            k, delim, sub_key = key.partition("__")
            if delim and k == "q":
                __remap[key] = f"quantifier__{sub_key}"

        return {(__remap[k] if k in __remap else k): v for k, v in params.items()}

    def __check_error(self, error):
        if error in qc.error.ACCURACY_ERROR:
            self.error = error
        elif isinstance(error, str):
            self.error = qc.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str representing\n"
                f"the name of an error function in {qc.error.ACCURACY_ERROR_NAMES}"
            )

    def fit(self, training: LabelledCollection):
        """Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :return: self
        """
        params_keys = list(self.param_grid.keys())
        params_values = list(self.param_grid.values())

        protocol = self.protocol

        self.param_scores_ = {}
        self.best_score_ = None

        tinit = time()

        hyper = [
            dict(zip(params_keys, val)) for val in itertools.product(*params_values)
        ]

        self._sout(f"starting model selection with {self.n_jobs =}")
        # self._sout("starting model selection")

        # scores = [self.__params_eval((params, training)) for params in hyper]
        scores = self._select_scores(hyper, training)

        for params, score, model in scores:
            if score is not None:
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = "timeout"

        tend = time() - tinit

        if self.best_score_ is None:
            raise TimeoutError("no combination of hyperparameters seem to work")

        self._sout(
            f"optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) "
            f"[took {tend:.4f}s]",
            level=1,
        )

        # log = Logger.logger()
        log = logger()
        log.debug(
            f"[{self.model.__class__.__name__}] "
            f"optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) "
            f"[took {tend:.4f}s]"
        )

        if self.refit:
            if isinstance(protocol, OnLabelledCollectionProtocol):
                self._sout("refitting on the whole development set")
                self.best_model_.fit(training + protocol.get_labelled_collection())
            else:
                raise RuntimeWarning(
                    f'"refit" was requested, but the protocol does not '
                    f"implement the {OnLabelledCollectionProtocol.__name__} interface"
                )

        return self

    def _select_scores(self, hyper, training):
        return qc.commons.parallel(
            self._params_eval,
            [(params, training) for params in hyper],
            n_jobs=self.n_jobs,
            verbose=1,
        )

    def _params_eval(self, params, training, protocol=None):
        protocol = self.protocol if protocol is None else protocol
        error = self.error

        # if self.timeout > 0:

        #     def handler(signum, frame):
        #         raise TimeoutError()

        #     signal.signal(signal.SIGALRM, handler)

        tinit = time()

        # if self.timeout > 0:
        #     signal.alarm(self.timeout)

        try:
            model = deepcopy(self.model)
            # overrides default parameters with the parameters being explored at this iteration
            model.set_params(**params)
            # print({k: v for k, v in model.get_params().items() if k in params})
            model.fit(training)
            score = evaluate(model, protocol=protocol, error_metric=error)

            ttime = time() - tinit
            self._sout(
                f"hyperparams={params}\t got score {score:.5f} [took {ttime:.4f}s]",
            )

            # if self.timeout > 0:
            #     signal.alarm(0)
        # except TimeoutError:
        #     self._sout(f"timeout ({self.timeout}s) reached for config {params}")
        #     score = None
        except ValueError as e:
            self._sout(
                f"the combination of hyperparameters {params} is invalid. Exception: {e}",
                level=1,
            )
            score = None
            # raise e
        except Exception as e:
            self._sout(
                f"something went wrong for config {params}; skipping:"
                f"\tException: {e}",
                level=1,
            )
            # raise e
            score = None

        return params, score, model

    def extend(
        self, coll: LabelledCollection, pred_proba=None, prefit=False
    ) -> ExtendedCollection:
        assert hasattr(self, "best_model_"), "quantify called before fit"
        return self.best_model().extend(coll, pred_proba=pred_proba, prefit=prefit)

    def estimate(self, instances):
        """Estimate class prevalence values using the best model found after calling the :meth:`fit` method.

        :param instances: sample contanining the instances
        :return: a ndarray of shape `(n_classes)` with class prevalence estimates as according to the best model found
            by the model selection process.
        """

        assert hasattr(self, "best_model_"), "estimate called before fit"
        return self.best_model().estimate(instances)

    def set_params(self, **parameters):
        """Sets the hyper-parameters to explore.

        :param parameters: a dictionary with keys the parameter names and values the list of values to explore
        """
        self.param_grid = parameters

    def get_params(self, deep=True):
        """Returns the dictionary of hyper-parameters to explore (`param_grid`)

        :param deep: Unused
        :return: the dictionary `param_grid`
        """
        return self.param_grid

    def best_model(self):
        """
        Returns the best model found after calling the :meth:`fit` method, i.e., the one trained on the combination
        of hyper-parameters that minimized the error function.

        :return: a trained quantifier
        """
        if hasattr(self, "best_model_"):
            return self.best_model_
        raise ValueError("best_model called before fit")

    def best_score(self):
        if hasattr(self, "best_score_"):
            return self.best_score_
        raise ValueError("best_score called before fit")


class RandomizedSearchAE(GridSearchAE):
    ERR_THRESHOLD = 1e-4
    MAX_ITER_IMPROV = 3

    def _select_scores(self, hyper, training: LabelledCollection):
        log = logger()
        hyper = np.array(hyper)
        rand_index = np.random.choice(
            np.arange(len(hyper)), size=len(hyper), replace=False
        )
        _n_jobs = os.cpu_count() + 1 + self.n_jobs if self.n_jobs < 0 else self.n_jobs
        batch_size = _n_jobs

        log.debug(f"{batch_size = }")
        rand_index = list(
            rand_index[: (len(hyper) // batch_size) * batch_size].reshape(
                (len(hyper) // batch_size, batch_size)
            )
        ) + [rand_index[(len(hyper) // batch_size) * batch_size :]]
        scores = []
        best_score, iter_from_improv = np.inf, 0
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for i, ri in enumerate(rand_index):
                tstart = time()
                _iter_scores = qc.commons.parallel(
                    self._params_eval,
                    [(params, training) for params in hyper[ri]],
                    parallel=parallel,
                )
                _best_iter_score = np.min(
                    [s for _, s, _ in _iter_scores if s is not None]
                )

                log.debug(
                    f"[iter {i}] best score = {_best_iter_score:.8f} [took {time() - tstart:.3f}s]"
                )
                scores += _iter_scores

                _check, best_score, iter_from_improv = self.__stop_condition(
                    _best_iter_score, best_score, iter_from_improv
                )
                if _check:
                    break

        return scores

    def __stop_condition(self, best_iter_score, best_score, iter_from_improv):
        if best_iter_score < best_score:
            _improv = best_score - best_iter_score
            best_score = best_iter_score
        else:
            _improv = 0

        if _improv > self.ERR_THRESHOLD:
            iter_from_improv = 0
        else:
            iter_from_improv += 1

        return iter_from_improv > self.MAX_ITER_IMPROV, best_score, iter_from_improv


class HalvingSearchAE(GridSearchAE):
    def _select_scores(self, hyper, training: LabelledCollection):
        log = logger()
        hyper = np.array(hyper)

        threshold = 22
        factor = 3
        n_steps = math.ceil(math.log(len(hyper) / threshold, factor))
        steps = np.logspace(n_steps, 0, base=1.0 / factor, num=n_steps + 1)
        with Parallel(n_jobs=self.n_jobs, verbose=1) as parallel:
            for _step in steps:
                tstart = time()
                _training, _ = (
                    training.split_stratified(train_prop=_step)
                    if _step < 1.0
                    else (training, None)
                )

                results = qc.commons.parallel(
                    self._params_eval,
                    [(params, _training) for params in hyper],
                    parallel=parallel,
                )
                scores = [(1.0 if s is None else s) for _, s, _ in results]
                res_hyper = np.array([h for h, _, _ in results], dtype="object")
                sorted_scores_idx = np.argsort(scores)
                best_score = scores[sorted_scores_idx[0]]
                hyper = res_hyper[
                    sorted_scores_idx[: round(len(res_hyper) * (1.0 / factor))]
                ]

                log.debug(
                    f"[step {_step}] best score = {best_score:.8f} [took {time() - tstart:.3f}s]"
                )

        return results


class SpiderSearchAE(GridSearchAE):
    def __init__(
        self,
        model: BaseAccuracyEstimator,
        param_grid: dict,
        protocol: AbstractProtocol,
        error: Union[Callable, str] = qc.error.maccd,
        refit=True,
        n_jobs=None,
        verbose=False,
        err_threshold=1e-4,
        max_iter_improv=0,
        pd_th_min=1,
        best_width=2,
    ):
        super().__init__(
            model=model,
            param_grid=param_grid,
            protocol=protocol,
            error=error,
            refit=refit,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.err_threshold = err_threshold
        self.max_iter_improv = max_iter_improv
        self.pd_th_min = pd_th_min
        self.best_width = best_width

    def _select_scores(self, hyper, training: LabelledCollection):
        log = logger()
        hyper = np.array(hyper)
        _n_jobs = os.cpu_count() + 1 + self.n_jobs if self.n_jobs < 0 else self.n_jobs
        batch_size = _n_jobs

        rand_index = np.arange(len(hyper))
        np.random.shuffle(rand_index)
        rand_index = rand_index[:batch_size]
        remaining_index = np.setdiff1d(np.arange(len(hyper)), rand_index)
        _hyper, _hyper_remaining = hyper[rand_index], hyper[remaining_index]

        scores = []
        best_score, last_best, iter_from_improv = np.inf, np.inf, 0
        with Parallel(n_jobs=self.n_jobs, verbose=1) as parallel:
            while len(_hyper) > 0:
                # log.debug(f"{len(_hyper_remaining)=}")
                tstart = time()
                _iter_scores = qc.commons.parallel(
                    self._params_eval,
                    [(params, training) for params in _hyper],
                    parallel=parallel,
                )

                # if all scores are None, select a new random batch
                if all([s[1] is None for s in _iter_scores]):
                    rand_index = np.arange(len(_hyper_remaining))
                    np.random.shuffle(rand_index)
                    rand_index = rand_index[:batch_size]
                    remaining_index = np.setdiff1d(
                        np.arange(len(_hyper_remaining)), rand_index
                    )
                    _hyper = _hyper_remaining[rand_index]
                    _hyper_remaining = _hyper_remaining[remaining_index]
                    continue

                _sorted_idx = np.argsort(
                    [1.0 if s is None else s for _, s, _ in _iter_scores]
                )
                _sorted_scores = np.array(_iter_scores, dtype="object")[_sorted_idx]
                _best_iter_params = np.array(
                    [p for p, _, _ in _sorted_scores], dtype="object"
                )
                _best_iter_scores = np.array(
                    [s for _, s, _ in _sorted_scores], dtype="object"
                )

                for i, (_score, _param) in enumerate(
                    zip(
                        _best_iter_scores[: self.best_width],
                        _best_iter_params[: self.best_width],
                    )
                ):
                    log.debug(
                        f"[size={len(_hyper)},place={i+1}] best score = {_score:.8f}; "
                        f"best param = {_param} [took {time() - tstart:.3f}s]"
                    )
                scores += _iter_scores

                _improv = best_score - _best_iter_scores[0]
                _improv_last = last_best - _best_iter_scores[0]
                if _improv > self.err_threshold:
                    iter_from_improv = 0
                    best_score = _best_iter_scores[0]
                elif _improv_last < 0:
                    iter_from_improv += 1

                last_best = _best_iter_scores[0]

                if iter_from_improv > self.max_iter_improv:
                    break

                _new_hyper = np.array([], dtype="object")
                for _base_param in _best_iter_params[: self.best_width]:
                    _rem_pds = np.array(
                        [
                            self.__param_distance(_base_param, h)
                            for h in _hyper_remaining
                        ]
                    )
                    _rem_pd_sort_idx = np.argsort(_rem_pds)
                    # _min_pd = np.min(_rem_pds)
                    _min_pd_len = (_rem_pds <= self.pd_th_min).nonzero()[0].shape[0]
                    _new_hyper_idx = _rem_pd_sort_idx[:_min_pd_len]
                    _hyper_rem_idx = np.setdiff1d(
                        np.arange(len(_hyper_remaining)), _new_hyper_idx
                    )
                    _new_hyper = np.concatenate(
                        [_new_hyper, _hyper_remaining[_new_hyper_idx]]
                    )
                    _hyper_remaining = _hyper_remaining[_hyper_rem_idx]
                _hyper = _new_hyper

        return scores

    def __param_distance(self, param1, param2):
        score = 0
        for k, v in param1.items():
            if param2[k] != v:
                score += 1

        return score
