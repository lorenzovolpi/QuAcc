import itertools
import os
from re import A

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
    cache_method,
    fit_method,
    get_intermediate_res,
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


def experiments():
    for (dataset_name, (L, V, U)), (cls_name, h) in gen_product(gen_datasets, gen_classifiers):
        print(f"training {cls_name} in {dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # compute some stats of the dataset
        save_dataset_stats(f"dataset_stats/{dataset_name}.json", test_prot, L, V)

        # precompute the actual accuracy values
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        _cache = {}
        for acc_name, acc_fn in gen_acc_measure():
            print(f"\tfor measure {acc_name}")
            L_prev, V_prev = get_plain_prev(L.prevalence()), get_plain_prev(V.prevalence())
            for method_name, method in gen_methods(h, acc_fn, ORACLE):
                report = _cache.get(method_name, None)
                if report is None:
                    report = TestReport(basedir, cls_name, dataset_name, L_prev, V_prev, method_name)

                if os.path.exists(report.get_path(acc_name)):
                    print(f"\t\t{method_name}-{acc_name} exists, skipping")
                    continue

                print(f"\t\t{method_name} computing...")

                if not report.has_intermediate_res():
                    method, t_train = fit_method(method, V)
                    test_prevs = prevs_from_prot(test_prot)
                    estim_inter, t_inter = get_intermediate_res(method, test_prot, ORACLE)
                    report.add_intermediate_res(method, test_prevs, estim_inter, t_train, t_inter)

                cache_method(report, _cache)

                estim_accs, t_test_ave = get_predictions(report.method, report.estim_inter, test_prot, acc_fn, ORACLE)
                report.add_final_res(acc_name, true_accs[acc_name], estim_accs, report.t_inter + t_test_ave)

                report.save_json(basedir, acc_name)

        print()


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
