import os
import time
from traceback import print_exception as traceback

import numpy as np
import pandas as pd
import quapy as qp
from joblib import Parallel, delayed
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

from quacc import logger
from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.utils import parallel

# from quacc.logger import logger, logger_manager

# from quacc.evaluation.worker import WorkerArgs, estimate_worker

pd.set_option("display.float_format", "{:.4f}".format)
# qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE


def estimate_worker(_estimate, train, validation, test, q=None):
    # qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE
    log = logger.setup_worker_logger(q)

    model = LogisticRegression()

    model.fit(*train.Xy)
    protocol = APP(
        test,
        n_prevalences=env.PROTOCOL_N_PREVS,
        repeats=env.PROTOCOL_REPEATS,
        return_type="labelled_collection",
        random_state=env._R_SEED,
    )
    start = time.time()
    try:
        result = _estimate(model, validation, protocol)
    except Exception as e:
        log.warning(f"Method {_estimate.name} failed. Exception: {e}")
        traceback(e)
        return None

    result.time = time.time() - start
    log.info(f"{_estimate.name} finished [took {result.time:.4f}s]")

    logger.logger_manager().rm_worker()

    return result


def split_tasks(estimators, train, validation, test, q):
    _par, _seq = [], []
    for estim in estimators:
        if hasattr(estim, "nocall"):
            continue
        _task = [estim, train, validation, test]
        match estim.name:
            case n if n.endswith("_gs"):
                _seq.append(_task)
            case _:
                _par.append(_task + [q])

    return _par, _seq


def evaluate_comparison(dataset: Dataset, estimators=None) -> DatasetReport:
    # log = Logger.logger()
    log = logger.logger()
    # with multiprocessing.Pool(1) as pool:
    __pool_size = round(os.cpu_count() * 0.8)
    # with multiprocessing.Pool(__pool_size) as pool:
    dr = DatasetReport(dataset.name)
    log.info(f"dataset {dataset.name} [pool size: {__pool_size}]")
    for d in dataset():
        log.info(
            f"Dataset sample {np.around(d.train_prev, decimals=2)} "
            f"of dataset {dataset.name} started"
        )
        par_tasks, seq_tasks = split_tasks(
            CE.func[estimators],
            d.train,
            d.validation,
            d.test,
            logger.logger_manager().q,
        )
        try:
            tstart = time.time()
            results = parallel(estimate_worker, par_tasks, n_jobs=env.N_JOBS, _env=env)
            results += parallel(estimate_worker, seq_tasks, n_jobs=1, _env=env)
            results = [r for r in results if r is not None]

            g_time = time.time() - tstart
            log.info(
                f"Dataset sample {np.around(d.train_prev, decimals=2)} "
                f"of dataset {dataset.name} finished "
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
                f"Dataset sample {np.around(d.train_prev, decimals=2)} "
                f"of dataset {dataset.name} failed. "
                f"Exception: {e}"
            )
            traceback(e)
    return dr
