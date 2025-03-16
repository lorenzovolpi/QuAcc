import itertools as IT
import os
import pdb
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.protocol import UPP

import quacc as qc
from exp.leap.config import (
    CSV_SEP,
    NUM_TEST,
    PROBLEM,
    PROJECT,
    gen_acc_measure,
    gen_classifiers,
    gen_datasets,
    gen_methods,
    gen_transformer_model_dataset,
    get_method_names,
    root_dir,
    sample_size,
)
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    get_predictions,
    split_validation,
    timestamp,
)
from quacc.models.cont_table import LEAP
from quacc.utils.commons import contingency_table, get_shift, true_acc

log = get_logger(id=PROJECT)


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, cls_name, acc_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.json")


def is_excluded(classifier, dataset, method, acc):
    return False


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


def all_exist_pre_check(dataset_name, cls_name):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        if is_excluded(cls_name, dataset_name, method, acc):
            continue
        path = local_path(dataset_name, cls_name, method, acc)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def experimental_protocol(cls_name, dataset_name, h, V, U, L_prevalence):
    # test generation protocol
    test_prot = UPP(
        U,
        repeats=NUM_TEST,
        return_type="labelled_collection",
        random_state=qp.environ["_R_SEED"],
    )

    # split validation set
    V1, V2_prot = split_validation(V)

    # precomumpute model posteriors for validation sets
    V_posteriors = h.predict_proba(V.X)
    V1_posteriors = h.predict_proba(V1.X)
    V2_prot_posteriors = []
    for sample in V2_prot():
        V2_prot_posteriors.append(h.predict_proba(sample.X))

    # precomumpute model posteriors for test samples
    test_prot_posteriors, test_prot_y_hat, test_prot_true_cts = [], [], []
    for sample in test_prot():
        P = h.predict_proba(sample.X)
        test_prot_posteriors.append(P)
        y_hat = np.argmax(P, axis=-1)
        test_prot_y_hat.append(y_hat)
        test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

    # precompute the actual accuracy values
    true_accs = {}
    for acc_name, acc_fn in gen_acc_measure():
        true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

    L_prev = get_plain_prev(L_prevalence)
    for method_name, method, val, val_posteriors in gen_methods(
        h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors
    ):
        val_prev = get_plain_prev(val.prevalence())
        t_train = None
        for acc_name, acc_fn in gen_acc_measure():
            if is_excluded(cls_name, dataset_name, method_name, acc_name):
                continue
            path = local_path(dataset_name, cls_name, method_name, acc_name)
            if os.path.exists(path):
                log.info(f"{method_name} on {acc_name} exists, skipping")
                continue

            try:
                method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
                t_train = t_train if _t_train is None else _t_train

                test_shift = get_shift(np.array([Ui.prevalence() for Ui in test_prot()]), L_prevalence)
                estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, test_prot, test_prot_posteriors)
                if estim_cts is None:
                    estim_cts = [None] * len(estim_accs)
                else:
                    estim_cts = [ct.tolist() for ct in estim_cts]
            except Exception as e:
                print_exception(e)
                log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
                continue

            ae = qc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs))
            method_df = pd.DataFrame(
                np.vstack([test_shift, true_accs[acc_name], estim_accs, ae]).T,
                columns=["shifts", "true_accs", "estim_accs", "acc_err"],
            )
            method_df["estim_cts"] = estim_cts
            method_df["true_cts"] = test_prot_true_cts
            method_df["classifier"] = cls_name
            method_df["method"] = method_name
            method_df["dataset"] = dataset_name
            method_df["acc_name"] = acc_name
            method_df["train_prev"] = [L_prev] * len(method_df)
            method_df["val_prev"] = [val_prev] * len(method_df)
            method_df["t_train"] = t_train
            method_df["t_test_ave"] = t_test_ave

            get_extra_from_method(method_df, method)

            log.info(f"{method_name} on {acc_name} done [{timestamp(t_train, t_test_ave)}]")
            method_df.to_json(path)


def experiments():
    for (cls_name, h), (dataset_name, (L, V, U)) in gen_model_dataset(gen_classifiers, gen_datasets):
        # compute and set the SAMPLE_SIZE for each dataset
        qp.environ["SAMPLE_SIZE"] = sample_size(len(U))

        # check if all results for current combination already exist
        # if so, skip the combination
        if all_exist_pre_check(dataset_name, cls_name):
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
            continue

        # fit model
        log.info(f"Training {cls_name} over {dataset_name}")
        h.fit(*L.Xy)

        experimental_protocol(cls_name, dataset_name, h, V, U, L.prevalence())

    for (cls_name, h), (dataset_name, (V, U), L_prev) in gen_transformer_model_dataset():
        # compute and set the SAMPLE_SIZE for each dataset
        qp.environ["SAMPLE_SIZE"] = 1000

        # check if all results for current combination already exist
        # if so, skip the combination
        if all_exist_pre_check(dataset_name, cls_name):
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
            continue

        log.info(f"Computing {cls_name} over {dataset_name}")
        experimental_protocol(cls_name, dataset_name, h, V, U, L_prev)


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
