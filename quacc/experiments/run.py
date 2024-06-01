import itertools as IT
import os
from time import time
from traceback import print_exception

import quapy as qp
from quapy.protocol import UPP

import quacc as qc
from quacc.experiments.generators import (
    gen_acc_measure,
    gen_bin_datasets,
    gen_classifiers,
    gen_methods,
    gen_multi_datasets,
    gen_product,
    gen_tweet_datasets,
    get_method_names,
)
from quacc.experiments.report import Report, TestReport
from quacc.experiments.util import (
    fit_or_switch,
    get_logger,
    get_plain_prev,
    get_predictions,
    prevs_from_prot,
)
from quacc.utils.commons import save_dataset_stats, true_acc

PROBLEM = "multiclass"
ORACLE = False
basedir = PROBLEM + ("-oracle" if ORACLE else "")

PLOTS = "predi_quant"
plots_basedir = basedir if PLOTS is None else basedir + "_" + PLOTS

NUM_TEST = 1000

if PROBLEM == "binary":
    qp.environ["SAMPLE_SIZE"] = 1000
    gen_datasets = gen_bin_datasets
elif PROBLEM == "multiclass":
    qp.environ["SAMPLE_SIZE"] = 250
    gen_datasets = gen_multi_datasets
elif PROBLEM == "tweet":
    qp.environ["SAMPLE_SIZE"] = 100
    gen_datasets = gen_tweet_datasets

log = get_logger()


def all_exist_pre_check(basedir, cls_name, dataset_name):
    method_names = get_method_names(PROBLEM)
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = TestReport(basedir, cls_name, acc, dataset_name, None, None, method).get_path()
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def experiments():
    log.info("-" * 31 + "  start  " + "-" * 31)

    for (dataset_name, (L, V, U)), (cls_name, h) in gen_product(gen_datasets, gen_classifiers):
        if all_exist_pre_check(basedir, cls_name, dataset_name):
            log.info(f"{cls_name} on dataset={dataset_name}: all results already exist, skipping")
            continue

        log.info(f"{cls_name} training on dataset={dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # compute some stats of the dataset
        save_dataset_stats(dataset_name, test_prot, L, V)

        # precompute the actual accuracy values

        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure(multiclass=True):
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        L_prev = get_plain_prev(L.prevalence())
        for method_name, method, V in gen_methods(h, V, PROBLEM, ORACLE):
            V_prev = get_plain_prev(V.prevalence())

            t_train = None
            for acc_name, acc_fn in gen_acc_measure(multiclass=True):
                report = TestReport(basedir, cls_name, acc_name, dataset_name, L_prev, V_prev, method_name)
                if os.path.exists(report.get_path()):
                    log.info(f"{method_name}: {acc_name} exists, skipping")
                    continue

                try:
                    method, _t_train = fit_or_switch(method, V, acc_fn, t_train is not None)
                    t_train = t_train if _t_train is None else _t_train

                    test_prevs = prevs_from_prot(test_prot)
                    estim_accs, t_test_ave = get_predictions(method, test_prot, ORACLE)
                    report.add_result(test_prevs, true_accs[acc_name], estim_accs, t_train, t_test_ave)
                except Exception as e:
                    print_exception(e)
                    log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
                    continue

                report.save_json(basedir, acc_name)

                log.info(f"{method_name}: {acc_name} done [t_train:{t_train:.3f}s; t_test_ave:{t_test_ave:.3f}s]")

    log.info("-" * 32 + "  end  " + "-" * 32)


# generate plots
def plotting():
    for (cls_name, _), (acc_name, _) in IT.product(gen_classifiers(), gen_acc_measure()):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        for dataset_name, _ in gen_datasets(only_names=True):
            methods = get_method_names(PROBLEM)
            rep = Report.load_results(basedir, cls_name, acc_name, dataset_name=dataset_name, method_name=methods)
            qc.plot.seaborn.plot_diagonal(
                rep.diagonal_plot_data(), cls_name, acc_name, dataset_name, basedir=plots_basedir
            )
            qc.plot.seaborn.plot_shift(rep.shift_plot_data(), cls_name, acc_name, dataset_name, basedir=plots_basedir)
            for _method in methods:
                m_rep = rep.filter_by_method(_method)
                qc.plot.seaborn.plot_diagonal(
                    m_rep.diagonal_plot_data(),
                    cls_name,
                    acc_name,
                    dataset_name,
                    basedir=plots_basedir,
                    file_name=f"diagonal_{_method}",
                )
        print(f"{cls_name}-{acc_name} plots generated")


# print("generating tables")
# gen_tables(basedir, datasets=[d for d, _ in gen_datasets(only_names=True)])

if __name__ == "__main__":
    try:
        experiments()
        # plotting()
    except Exception as e:
        log.error(e)
        print_exception(e)
