import itertools
import os

import quapy as qp
from quapy.protocol import UPP

from quacc.dataset import save_dataset_stats
from quacc.experiments.generators import (
    any_missing,
    gen_acc_measure,
    gen_bin_datasets,
    gen_CAP,
    gen_CAP_cont_table,
    gen_classifiers,
    gen_multi_datasets,
    gen_tweet_datasets,
    get_method_names,
)
from quacc.experiments.plotting import (
    save_plot_diagonal,
    save_plot_shift,
)
from quacc.experiments.report import Report, TestReport
from quacc.experiments.util import (
    fit_method,
    get_plain_prev,
    predictionsCAP,
    predictionsCAPcont_table,
    prevs_from_prot,
    true_acc,
)

PROBLEM = "binary"
ORACLE = False
basedir = PROBLEM + ("-oracle" if ORACLE else "")
EXPERIMENT = False
PLOTTING = True


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


if EXPERIMENT:
    for (cls_name, h), (dataset_name, (L, V, U)) in itertools.product(
        gen_classifiers(), gen_datasets()
    ):
        print(f"training {cls_name} in {dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(
            U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0
        )

        # compute some stats of the dataset
        save_dataset_stats(f"dataset_stats/{dataset_name}.json", test_prot, L, V)

        # precompute the actual accuracy values
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        print("CAP methods")
        # instances of ClassifierAccuracyPrediction are bound to the evaluation measure, so they
        # must be nested in the acc-for
        for acc_name, acc_fn in gen_acc_measure():
            print(f"\tfor measure {acc_name}")
            for method_name, method in gen_CAP(h, acc_fn, with_oracle=ORACLE):
                report = TestReport(
                    basedir=basedir,
                    cls_name=cls_name,
                    acc_name=acc_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    train_prev=get_plain_prev(L.prevalence()),
                    val_prev=get_plain_prev(V.prevalence()),
                )
                if os.path.exists(report.path):
                    print(f"\t\t{method_name}-{acc_name} exists, skipping")
                    continue

                print(f"\t\t{method_name} computing...")
                method, t_train = fit_method(method, V)
                estim_accs, t_test_ave = predictionsCAP(method, test_prot, ORACLE)
                test_prevs = prevs_from_prot(test_prot)
                report.add_result(
                    test_prevs=test_prevs,
                    true_accs=true_accs[acc_name],
                    estim_accs=estim_accs,
                    t_train=t_train,
                    t_test_ave=t_test_ave,
                ).save_json(basedir)

        print("\nCAP_cont_table methods")
        # instances of CAPContingencyTable instead are generic, and the evaluation measure can
        # be nested to the predictions to speed up things
        for method_name, method in gen_CAP_cont_table(h):
            if not any_missing(basedir, cls_name, dataset_name, method_name):
                print(
                    f"\t\tmethod {method_name} has all results already computed. Skipping."
                )
                continue

            print(f"\t\tmethod {method_name} computing...")

            method, t_train = fit_method(method, V)
            estim_accs_dict, t_test_ave = predictionsCAPcont_table(
                method, test_prot, gen_acc_measure, ORACLE
            )
            for acc_name, estim_accs in estim_accs_dict.items():
                report = TestReport(
                    basedir=basedir,
                    cls_name=cls_name,
                    acc_name=acc_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    train_prev=get_plain_prev(L.prevalence()),
                    val_prev=get_plain_prev(V.prevalence()),
                )
                test_prevs = prevs_from_prot(test_prot)
                report.add_result(
                    test_prevs=test_prevs,
                    true_accs=true_accs[acc_name],
                    estim_accs=estim_accs,
                    t_train=t_train,
                    t_test_ave=t_test_ave,
                ).save_json(basedir)

        print()

# generate plots
if PLOTTING:
    for (cls_name, _), (acc_name, _) in itertools.product(
        gen_classifiers(), gen_acc_measure()
    ):
        # save_plot_diagonal(basedir, cls_name, acc_name)
        for dataset_name, _ in gen_datasets(only_names=True):
            methods = get_method_names()
            report = Report.load_results(
                basedir,
                cls_name,
                acc_name,
                dataset_name=dataset_name,
                method_name=methods,
            )
            save_plot_diagonal(
                basedir, cls_name, acc_name, dataset_name=dataset_name, report=report
            )
            # save_plot_delta(basedir, cls_name, acc_name, dataset_name=dataset_name, report=report)
            # save_plot_delta(basedir,cls_name,acc_name,dataset_name=dataset_name,stdev=True,report=report)
            save_plot_shift(
                basedir, cls_name, acc_name, dataset_name=dataset_name, report=report
            )

# print("generating tables")
# gen_tables(basedir, datasets=[d for d, _ in gen_datasets(only_names=True)])
