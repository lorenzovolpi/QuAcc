import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.data.datasets import fetch_IMDBDataset, fetch_RCV1BinaryDataset
from quacc.data.util import split_train
from quacc.error import f1_macro, vanilla_acc
from quacc.experiments.util import fit_or_switch, get_logger, get_plain_prev, get_predictions, prevs_from_prot
from quacc.models.cont_table import LEAP
from quacc.utils.commons import get_shift, true_acc

qp.environ["SAMPLE_SIZE"] = 1000
NUM_TEST = 100
qp.environ["_R_SEED"] = 0

CSV_SEP = ","

log = get_logger(id="rleap")


def sld():
    return EMQ(LogisticRegression(), val_split=5)


def kdey():
    return KDEyML(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()


def gen_datasets():
    yield "IMDB", fetch_IMDBDataset()
    for dn in ["CCAT", "GCAT", "MCAT"]:
        dval = fetch_RCV1BinaryDataset(dn)
        yield dn, dval


def get_train_samples(dataset):
    L, V, U = dataset

    train_prevs = np.linspace(0.1, 1, 9, endpoint=False)
    L_size = np.min(np.around(np.min(L.counts()) / train_prevs, decimals=0))
    V_size = np.min(np.around(np.min(V.counts()) / train_prevs, decimals=0))

    datasets = [(L.sampling(int(L_size), p), V.sampling(int(V_size), p), U) for p in train_prevs]
    return datasets


def prev_str(L: LabelledCollection):
    return round(L.prevalence()[1] * 100)


def local_path(dataset_name, cls_name, method_name, acc_name, L: LabelledCollection):
    L_prev = str(prev_str(L))
    parent_dir = os.path.join(qc.env["OUT_DIR"], "rleap", cls_name, acc_name, dataset_name, L_prev)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


def gen_methods(h):
    _, acc_fn = next(gen_accs())
    yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h)


def get_method_names():
    mock_h = LogisticRegression()

    return [m for m, _ in gen_methods(mock_h)]


def gen_accs():
    yield "vanilla_acc", vanilla_acc
    yield "f1", f1_macro


def all_exist_pre_check(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_accs()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(dataset_name, cls_name, method, acc, L)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def preload_existing(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_accs()]

    dfs = []
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(dataset_name, cls_name, method, acc, L)
        method_df = pd.read_csv(path, sep=CSV_SEP)
        # if "dataset" not in method_df:
        #     method_df["dataset"] = dataset_name
        #     method_df.to_csv(path, sep=CSV_SEP)
        dfs.append(method_df)

    return dfs


def experiments():
    log.info("-" * 31 + "  start  " + "-" * 31)

    dfs = []
    for cls_name, h in gen_classifiers():
        for dataset_name, dataset in gen_datasets():
            for L, V, U in get_train_samples(dataset):
                if all_exist_pre_check(dataset_name, cls_name, L):
                    log.info(f"All results for {cls_name} over {dataset_name}[{prev_str(L)}] exist, skipping")
                    dfs.extend(preload_existing(dataset_name, cls_name, L))
                    continue

                test_prot = APP(
                    U,
                    n_prevalences=21,
                    repeats=NUM_TEST,
                    return_type="labelled_collection",
                    random_state=qp.environ["_R_SEED"],
                )

                log.info(f"Training {cls_name} over {dataset_name}[{prev_str(L)}]")
                h.fit(*L.Xy)

                # generate posteriors
                V_posteriors = h.predict_proba(V.X)
                test_prot_posteriors, test_prot_y_hat = [], []
                for sample in test_prot():
                    P = h.predict_proba(sample.X)
                    test_prot_posteriors.append(P)
                    test_prot_y_hat.append(np.argmax(P, axis=-1))

                # precompute the actual accuracy values
                true_accs = {}
                for acc_name, acc_fn in gen_accs():
                    true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

                for method_name, method in gen_methods(h):
                    t_train = None
                    for acc_name, acc_fn in gen_accs():
                        path = local_path(dataset_name, cls_name, method_name, acc_name, L)

                        if os.path.exists(path):
                            method_df = pd.read_csv(path, sep=CSV_SEP)
                            dfs.append(method_df)
                            log.info(f"{method_name} on {acc_name} exists, skipping")
                            continue

                        try:
                            method, _t_train = fit_or_switch(method, V, V_posteriors, acc_fn, t_train is not None)
                            t_train = t_train if _t_train is None else _t_train
                            test_shift = get_shift(np.array([Ui.prevalence() for Ui in test_prot()]), L.prevalence())
                            estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors, False)
                        except Exception as e:
                            print_exception(e)
                            log.warning(f"{method_name}[{prev_str(L)}]: {acc_name} gave error '{e}' - skipping")
                            continue

                        ae = qc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs))
                        method_df = pd.DataFrame(
                            np.vstack([test_shift, true_accs[acc_name], estim_accs, ae]).T,
                            columns=["shifts", "true_accs", "estim_accs", "acc_err"],
                        )
                        method_df["method"] = method_name
                        method_df["dataset"] = dataset_name
                        method_df["acc_name"] = acc_name
                        method_df["train_prev"] = np.around(L.prevalence(), decimals=2)[1]
                        log.info(f"{method_name} on {acc_name} done")
                        method_df.to_csv(path, sep=CSV_SEP)
                        dfs.append(method_df)

    log.info("-" * 32 + "  end  " + "-" * 32)

    results = pd.concat(dfs, axis=0)

    pivot = (
        results.groupby(by=["dataset", "acc_name", "method"])
        .mean()
        .reset_index()
        .pivot(index=["acc_name", "method"], columns=["dataset"], values="acc_err")
    )
    print(pivot)
    print(pivot.to_latex())


if __name__ == "__main__":
    try:
        experiments()
    except Exception as e:
        log.error(e)
        print_exception(e)
