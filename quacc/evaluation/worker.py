import time
from dataclasses import dataclass
from multiprocessing import Queue
from traceback import print_exception as traceback

import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

from quacc.environment import env, environ
from quacc.logger import SubLogger


@dataclass(frozen=True)
class WorkerArgs:
    _estimate: callable
    train: LabelledCollection
    validation: LabelledCollection
    test: LabelledCollection
    _env: environ
    q: Queue


def estimate_worker(args: WorkerArgs):
    with env.load(args._env):
        qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE
        SubLogger.setup(args.q)
        log = SubLogger.logger()

        model = LogisticRegression()

        model.fit(*args.train.Xy)
        protocol = APP(
            args.test,
            n_prevalences=env.PROTOCOL_N_PREVS,
            repeats=env.PROTOCOL_REPEATS,
            return_type="labelled_collection",
            random_state=env._R_SEED,
        )
        start = time.time()
        try:
            result = args._estimate(model, args.validation, protocol)
        except Exception as e:
            log.warning(f"Method {args._estimate.name} failed. Exception: {e}")
            traceback(e)
            return None

        result.time = time.time() - start
        log.info(f"{args._estimate.name} finished [took {result.time:.4f}s]")

        return result
