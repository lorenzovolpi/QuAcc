from traceback import print_exception as traceback

import quacc.legacy.evaluation.comp as comp

# from quacc.logger import Logger
from quacc import logger
from quacc.dataset import Dataset
from quacc.legacy.environment import env
from quacc.legacy.evaluation.estimators import CE
from quacc.utils.commons import create_dataser_dir


def estimate_comparison():
    # log = Logger.logger()
    log = logger.logger()
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
        # Logger.add_handler(env.OUT_DIR / f"{dataset.name}.log")
        logger.add_handler(env.OUT_DIR / f"{dataset.name}.log")
        try:
            dr = comp.evaluate_comparison(
                dataset,
                estimators=CE.name[env.COMP_ESTIMATORS],
            )
            dr.pickle(env.OUT_DIR / f"{dataset.name}.pickle")
        except Exception as e:
            log.error(f"Evaluation over {dataset.name} failed. Exception: {e}")
            traceback(e)

        # Logger.clear_handlers()
        logger.clear_handlers()


def main():
    # log = Logger.logger()
    log = logger.setup_logger()

    try:
        estimate_comparison()
    except Exception as e:
        log.error(f"estimate comparison failed. Exception: {e}")
        traceback(e)

    # Logger.close()
    logger.logger_manager().close()


if __name__ == "__main__":
    main()
