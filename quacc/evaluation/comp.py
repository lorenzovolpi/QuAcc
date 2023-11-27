import multiprocessing
import os
import time
from traceback import print_exception as traceback

import pandas as pd
import quapy as qp

from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.evaluation.worker import WorkerArgs, estimate_worker
from quacc.logger import Logger

pd.set_option("display.float_format", "{:.4f}".format)
qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE


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
