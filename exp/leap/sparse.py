import os
from argparse import ArgumentParser
from traceback import print_exception
from typing import Callable

import numpy as np
import pandas as pd
import quapy as qp
from scipy import optimize
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone
from sklearn.linear_model import LogisticRegression

import exp.leap.config as cfg
import exp.leap.env as env
import quacc as qc
from exp.leap.bootstrap import RepDatasetBundle
from exp.leap.config import (
    EXP,
    DatasetBundle,
    gen_acc_measure,
    gen_datasets,
    kdey,
)
from exp.leap.util import (
    all_exist_pre_check,
    gen_method_df,
    get_extra_from_method,
    is_excluded,
    load_results,
    local_path,
)
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from quacc.models._oleap import OCE_cuda, OCE_sparse
from quacc.models.direct import DoC
from quacc.utils.commons import get_shift, parallel, true_acc

SUBPROJECT = "sparse"
qp.environ["SAMPLE_SIZE"] = 100

log = get_logger(id=f"{env.PROJECT}.{SUBPROJECT}")


def _optim_minimize(loss: Callable, n_classes: int, method="SLSQP", bounds=None, constraints=None):
    # the initial point is set as the uniform distribution
    uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))
    # uniform_distribution = csr_array(uniform_distribution)

    # solutions are bounded to those contained in the unit-simplex
    r = optimize.minimize(loss, x0=uniform_distribution, method=method, bounds=bounds, constraints=constraints)
    return r.x


def _optimize_lsq_linear(A, b, bounds=(0, np.inf)):
    r = optimize.lsq_linear(A, b, bounds=bounds)

    x = r.x
    if not np.isclose(x.sum(), 1):
        x = softmax(x)

    return x


def gen_methods(h: BaseEstimator, D: RepDatasetBundle):
    _, acc_fn = next(gen_acc_measure())
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors), D.V1, D.V1_posteriors
    yield "OCE-CUDA", OCE_cuda(acc_fn, kdey()), D.V, D.V_posteriors
    yield "OCE-SPARSE", OCE_sparse(acc_fn, kdey(), optim_method="cvxpy"), D.V, D.V_posteriors


def gen_classifiers():
    yield "LR", LogisticRegression()


def get_classifier_names():
    return [c for c, _ in gen_classifiers()]


def get_method_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_h = LogisticRegression()
    mock_D = RepDatasetBundle.mock()
    return [m for m, _, _, _ in gen_methods(mock_h, mock_D)]


def exp_protocol(args):
    cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, cls_name, method_name, acc_name, subproject=SUBPROJECT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(cls_name, dataset_name, acc_name, method_name))
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
            estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, D.test_prot, D.test_prot_posteriors)
            if estim_cts is None:
                estim_cts = [None] * len(estim_accs)
            else:
                estim_cts = [ct.tolist() for ct in estim_cts]
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, cls_name, dataset_name, acc_name, method_name))
            continue

        ae = qc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs)).tolist()

        df_len = len(estim_accs)
        method_df = gen_method_df(
            df_len,
            shifts=test_shift,
            true_accs=true_accs[acc_name],
            estim_accs=estim_accs,
            acc_err=ae,
            estim_cts=estim_cts,
            true_cts=D.test_prot_true_cts,
            classifier=cls_name,
            method=method_name,
            dataset=dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        get_extra_from_method(method_df, method)

        results.append(
            EXP.SUCCESS(
                cls_name, dataset_name, acc_name, method_name, df=method_df, t_train=t_train, t_test_ave=t_test_ave
            )
        )

    return results


def train_cls(args):
    (cls_name, orig_h), (dataset_name, (L, V, U)) = args
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(dataset_name, cls_name, subproject=SUBPROJECT):
        return (cls_name, dataset_name, None, None, None)
    else:
        # clone model from the original one
        h = skl_clone(orig_h)
        # fit model
        h.fit(*L.Xy)
        # create dataset bundle
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(h)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (cls_name, dataset_name, h, D, true_accs)


def experiments():
    cls_train_args = list(gen_model_dataset(gen_classifiers, gen_datasets))
    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )
    cls_dataset = []
    for cls_name, dataset_name, h, D, true_accs in cls_dataset_gen:
        if h is None and D is None:
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
        else:
            log.info(f"Trained {cls_name} over {dataset_name}")
            cls_dataset.append((cls_name, dataset_name, h, D, true_accs))

    exp_prot_args_list = []
    for cls_name, dataset_name, h, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(h, D):
            exp_prot_args_list.append(
                (cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors)
            )

    results_gen = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )

    # results_gen = (exp_protocol(args_list) for args_list in exp_prot_args_list)

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(r.dataset_name, r.cls_name, r.method_name, r.acc_name, subproject=SUBPROJECT)
                r.df.to_json(path)
                log.info(
                    f"[{r.cls_name}@{r.dataset_name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.cls_name}@{r.dataset_name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.cls_name}@{r.dataset_name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


def show_results():
    old_df = load_results(base_dir=env.root_dir, filter_methods=["OCE(KDEy-MLP)-SLSQP"])
    new_df = load_results(
        base_dir=os.path.join(env.root_dir, "sparse"), filter_methods=["DoC", "OCE-CUDA", "OCE-SPARSE"]
    )
    df = pd.concat([old_df, new_df])

    print(pd.pivot_table(df, index=["dataset"], columns=["method"], values="acc_err"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        action="store",
        default="multiclass",
        help="Select the problem you want to generate plots for",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.problem not in env._valid_problems:
        raise ValueError(f"Invalid problem {args.problem}: valid problems are {env._valid_problems}")
    env.PROBLEM = args.problem

    if args.show:
        show_results()
    else:
        try:
            log.info("-" * 31 + "  start  " + "-" * 31)
            experiments()
            log.info("-" * 32 + "  end  " + "-" * 32)
        except Exception as e:
            log.error(e)
            print_exception(e)
