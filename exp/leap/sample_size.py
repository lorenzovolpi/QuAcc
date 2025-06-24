import itertools as IT
import os
from dataclasses import dataclass
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from sklearn.base import clone as skl_clone

import quacc as qc
from exp.leap.config import (
    PROBLEM,
    PROJECT,
    DatasetBundle,
    gen_acc_measure,
    gen_classifiers,
    gen_methods,
    get_method_names,
    root_dir,
)
from exp.leap.util import gen_method_df, get_extra_from_method, is_excluded
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size
from quacc.utils.commons import get_shift, parallel, true_acc

log = get_logger(id=PROJECT)


def local_path(dataset_name, sample_size, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, "sample_size", PROBLEM, cls_name, acc_name, f"{dataset_name}_{sample_size}")
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.json")


def all_exist_pre_check(dataset_name, sample_size, cls_name):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(dataset_name, sample_size, cls_name, method, acc)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def gen_sample_sizes():
    if PROBLEM == "binary":
        return np.unique(np.around(np.hstack([np.logspace(1, 4, 20), np.logspace(1, 4, 4)]), decimals=0).astype("int"))
    elif PROBLEM == "multiclass":
        return np.unique(np.around(np.hstack([np.logspace(1, 5, 20), np.logspace(1, 5, 5)]), decimals=0).astype("int"))


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        dn = _sorted_uci_names[0]
        dval = None if only_names else fetch_UCIBinaryDataset(dn)
        yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        dn = _sorted_uci_names[0]
        dval = None if only_names else fetch_UCIMulticlassDataset(dn)
        yield dn, dval


@dataclass
class EXP:
    code: int
    cls_name: str
    dataset_name: str
    sample_size: int
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(args):
    cls_name, dataset_name, sample_size, h, D, true_accs, method_name, method, val, val_posteriors = args
    qp.environ["SAMPLE_SIZE"] = sample_size
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, sample_size, cls_name, method_name, acc_name)
        if os.path.exists(path):
            results.append(EXP.EXISTS(cls_name, dataset_name, sample_size, acc_name, method_name))
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
            results.append(EXP.ERROR(e, cls_name, dataset_name, sample_size, acc_name, method_name))
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
            sample_size=sample_size,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        get_extra_from_method(method_df, method)

        results.append(
            EXP.SUCCESS(
                cls_name,
                dataset_name,
                sample_size,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def train_cls(args):
    (cls_name, orig_h), (dataset_name, (L, V, U)), sample_size = args
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(dataset_name, sample_size, cls_name):
        return (cls_name, dataset_name, sample_size, None, None, None)
    else:
        # clone model from the original one
        h = skl_clone(orig_h)
        # fit model
        h.fit(*L.Xy)
        # create dataset bundle
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(h, sample_size=sample_size)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (cls_name, dataset_name, sample_size, h, D, true_accs)


def experiments():
    cls_train_args = []
    for _model in gen_classifiers():
        for _dataset in gen_datasets():
            for sample_size in gen_sample_sizes():
                cls_train_args.append((_model, _dataset, sample_size))

    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )
    cls_dataset = []
    for cls_name, dataset_name, sample_size, h, D, true_accs in cls_dataset_gen:
        if h is None and D is None:
            log.info(f"All results for {cls_name} over {dataset_name}_{sample_size} exist, skipping")
        else:
            log.info(f"Trained {cls_name} over {dataset_name}_{sample_size}")
            cls_dataset.append((cls_name, dataset_name, sample_size, h, D, true_accs))

    exp_prot_args_list = []
    for cls_name, dataset_name, sample_size, h, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(h, D):
            exp_prot_args_list.append(
                (cls_name, dataset_name, sample_size, h, D, true_accs, method_name, method, val, val_posteriors)
            )

    results_gen = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(r.dataset_name, r.sample_size, r.cls_name, r.method_name, r.acc_name)
                r.df.to_json(path)
                log.info(
                    f"[{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(
                    f"[{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name} on {r.acc_name} exists, skipping"
                )
            elif r.error:
                log.warning(
                    f"[{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
