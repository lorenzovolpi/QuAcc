from traceback import print_exception as traceback

import quacc.evaluation.comp as comp
from quacc.dataset import Dataset
from quacc.environment import env
from quacc.logger import Logger
from quacc.utils import create_dataser_dir

CE = comp.CompEstimator()


def estimate_comparison():
    log = Logger.logger()
    for conf in env.load_confs():
        dataset = Dataset(
            env.DATASET_NAME,
            target=env.DATASET_TARGET,
            n_prevalences=env.DATASET_N_PREVS,
            prevs=env.DATASET_PREVS,
        )
        create_dataser_dir(
            dataset.name,
            update=env.DATASET_DIR_UPDATE,
        )
        Logger.add_handler(env.OUT_DIR / f"{dataset.name}.log")
        try:
            dr = comp.evaluate_comparison(
                dataset,
                estimators=CE.name[env.COMP_ESTIMATORS],
            )
            dr.pickle(env.OUT_DIR / f"{dataset.name}.pickle")
        except Exception as e:
            log.error(f"Evaluation over {dataset.name} failed. Exception: {e}")
            traceback(e)

        Logger.clear_handlers()


def main():
    log = Logger.logger()
    try:
        estimate_comparison()
    except Exception as e:
        log.error(f"estimate comparison failed. Exception: {e}")
        traceback(e)

    Logger.close()


if __name__ == "__main__":
    main()
