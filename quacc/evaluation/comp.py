import multiprocessing
import os
import time
from traceback import print_exception as traceback
from typing import List

import numpy as np
import pandas as pd
import quapy as qp

from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation import baseline, method
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.evaluation.worker import WorkerArgs, estimate_worker
from quacc.logger import Logger

pd.set_option("display.float_format", "{:.4f}".format)
qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE


class CompEstimatorName_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return self.ce._CompEstimator__get(e)[0]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).keys())

    @property
    def all(self):
        all_keys = list(CompEstimator._CompEstimator__dict.keys())
        return self[all_keys]


class CompEstimatorFunc_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return self.ce._CompEstimator__get(e)[1]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).values())


class CompEstimator:
    __dict = method._methods | baseline._baselines

    def __get(cls, e: str | List[str]):
        if isinstance(e, str):
            try:
                return (e, cls.__dict[e])
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list):
            _subtr = np.setdiff1d(e, list(cls.__dict.keys()))
            if len(_subtr) > 0:
                raise KeyError(
                    f"Invalid estimator: estimator {_subtr[0]} does not exist"
                )

            e_fun = {k: fun for k, fun in cls.__dict.items() if k in e}
            if "ref" not in e:
                e_fun["ref"] = cls.__dict["ref"]

            return e_fun

    @property
    def name(self):
        return CompEstimatorName_(self)

    @property
    def func(self):
        return CompEstimatorFunc_(self)


CE = CompEstimator()


def evaluate_comparison(dataset: Dataset, estimators=None) -> DatasetReport:
    log = Logger.logger()
    # with multiprocessing.Pool(1) as pool:
    __pool_size = round(os.cpu_count() * 0.8)
    with multiprocessing.Pool(__pool_size) as pool:
        dr = DatasetReport(dataset.name)
        log.info(f"dataset {dataset.name} [pool size: {__pool_size}]")
        for d in dataset():
            log.info(
                f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} started"
            )
            tasks = [
                WorkerArgs(
                    _estimate=estim,
                    train=d.train,
                    validation=d.validation,
                    test=d.test,
                    _env=env,
                    q=Logger.queue(),
                )
                for estim in CE.func[estimators]
            ]
            try:
                tstart = time.time()
                results = [
                    r for r in pool.imap(estimate_worker, tasks) if r is not None
                ]

                g_time = time.time() - tstart
                log.info(
                    f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} finished "
                    f"[took {g_time:.4f}s]"
                )

                cr = CompReport(
                    results,
                    name=dataset.name,
                    train_prev=d.train_prev,
                    valid_prev=d.validation_prev,
                    g_time=g_time,
                )
                dr += cr

            except Exception as e:
                log.warning(
                    f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} failed. "
                    f"Exception: {e}"
                )
                traceback(e)
    return dr
