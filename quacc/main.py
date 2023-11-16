from traceback import print_exception as traceback

import quacc.evaluation.comp as comp
from quacc.dataset import Dataset
from quacc.environment import env, environ
from quacc.evaluation.report import DatasetReport
from quacc.logger import Logger
from quacc.utils import create_dataser_dir

CE = comp.CompEstimator()

CREATE_MD = False


def create_md(_env: environ, dr: DatasetReport, dataset: Dataset, log: Logger):
    for plot_conf in _env.get_plot_confs():
        for m in _env.METRICS:
            output_path = _env.OUT_DIR / f"{plot_conf}_{m}.md"
            try:
                _repr = dr.to_md(
                    conf=plot_conf,
                    metric=m,
                    estimators=CE.name[_env.PLOT_ESTIMATORS],
                    plot_path=_env.PLOT_OUT_DIR,
                )
                with open(output_path, "w") as f:
                    f.write(_repr)
            except Exception as e:
                log.error(
                    f"Failed while saving configuration {plot_conf} of {dataset.name}. Exception: {e}"
                )
                traceback(e)


def estimate_comparison():
    log = Logger.logger()
    for conf in env.get_confs():
        dataset = Dataset(
            env.DATASET_NAME,
            target=env.DATASET_TARGET,
            n_prevalences=env.DATASET_N_PREVS,
            prevs=env.DATASET_PREVS,
        )
        create_dataser_dir(dataset.name, update=env.DATASET_DIR_UPDATE)
        Logger.add_handler(env.OUT_DIR / f"{dataset.name}.log")
        try:
            dr = comp.evaluate_comparison(
                dataset,
                estimators=CE.name[env.COMP_ESTIMATORS],
            ).pickle(env.OUT_DIR / f"{dataset.name}.pickle")
        except Exception as e:
            log.error(f"Evaluation over {dataset.name} failed. Exception: {e}")
            traceback(e)

        if CREATE_MD:
            create_md(env, dr, dataset, log)

        Logger.clear_handlers()


def main():
    log = Logger.logger()
    try:
        estimate_comparison()
    except Exception as e:
        log.error(f"estimate comparison failed. Exceprion: {e}")
        traceback(e)

    Logger.close()


if __name__ == "__main__":
    main()
