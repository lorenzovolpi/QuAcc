import itertools
import os
from time import time

import quapy as qp
from quapy.protocol import UPP

import quacc as qc
from quacc.dataset import save_dataset_stats
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
from quacc.utils.commons import true_acc

PROBLEM = "binary"
ORACLE = False
basedir = PROBLEM + ("-oracle" if ORACLE else "")


if PROBLEM == "binary":
    qp.environ["SAMPLE_SIZE"] = 1000
    NUM_TEST = 1000
    gen_datasets = gen_bin_datasets
elif PROBLEM == "multiclass":
    qp.environ["SAMPLE_SIZE"] = 250
    NUM_TEST = 1000
    gen_datasets = gen_multi_datasets
elif PROBLEM == "tweet":
    qp.environ["SAMPLE_SIZE"] = 100
    NUM_TEST = 1000
    gen_datasets = gen_tweet_datasets

log = get_logger()


def experiments():
    for (dataset_name, (L, V, U)), (cls_name, h) in gen_product(gen_datasets, gen_classifiers):
        log.info(f"training {cls_name} in {dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # compute some stats of the dataset
        save_dataset_stats(f"dataset_stats/{dataset_name}.json", test_prot, L, V)

        # precompute the actual accuracy values
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        L_prev = get_plain_prev(L.prevalence())
        for method_name, method, V in gen_methods(h, V, ORACLE):
            V_prev = get_plain_prev(V.prevalence())

            log.info(f"  {method_name} computing...")
            t_train = None
            for acc_name, acc_fn in gen_acc_measure():
                report = TestReport(basedir, cls_name, acc_name, dataset_name, L_prev, V_prev, method_name)
                if os.path.exists(report.get_path()):
                    log.info(f"    {acc_name} exists, skipping")
                    continue

                log.info(f"    {acc_name}...")

                method, _t_train = fit_or_switch(method, V, acc_fn, t_train is not None)
                t_train = t_train if _t_train is None else _t_train

                test_prevs = prevs_from_prot(test_prot)
                estim_accs, t_test_ave = get_predictions(method, test_prot, ORACLE)
                report.add_result(test_prevs, true_accs[acc_name], estim_accs, t_train, t_test_ave)

                report.save_json(basedir, acc_name)

        log.info("-" * 70)
    log.info("-" * 70)


# generate plots
def plotting():
    for (cls_name, _), (acc_name, _) in itertools.product(gen_classifiers(), gen_acc_measure()):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        for dataset_name, _ in gen_datasets(only_names=True):
            methods = get_method_names()
            rep = Report.load_results(basedir, cls_name, acc_name, dataset_name=dataset_name, method_name=methods)
            qc.plot.seaborn.plot_diagonal(rep.diagonal_plot_data(), cls_name, acc_name, dataset_name, basedir=basedir)
            qc.plot.seaborn.plot_shift(rep.shift_plot_data(), cls_name, acc_name, dataset_name, basedir=basedir)
            for _method in methods:
                m_rep = rep.filter_by_method(_method)
                qc.plot.seaborn.plot_diagonal(
                    m_rep.diagonal_plot_data(),
                    cls_name,
                    acc_name,
                    dataset_name,
                    basedir=basedir,
                    file_name=f"diagonal_{_method}",
                )


# print("generating tables")
# gen_tables(basedir, datasets=[d for d, _ in gen_datasets(only_names=True)])

if __name__ == "__main__":
    experiments()
    plotting()
