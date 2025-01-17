import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import APP, UPP
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.data.datasets import (
    RCV1_MULTICLASS_DATASETS,
    fetch_IMDBDataset,
    fetch_RCV1BinaryDataset,
    fetch_RCV1MulticlassDataset,
    fetch_UCIMulticlassDataset,
)
from quacc.experiments.generators import gen_acc_measure, gen_model_dataset
from quacc.experiments.util import (
    fit_or_switch,
    get_logger,
    get_plain_prev,
    get_predictions,
    split_validation,
)
from quacc.models.cont_table import LEAP, QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.direct import ATC, DoC
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import get_shift, true_acc

root_dir = os.path.join(qc.env["OUT_DIR"], "sampling")
qp.environ["SAMPLE_SIZE"] = 1000
NUM_TEST = 100
qp.environ["_R_SEED"] = 0
PROBLEM = "binary"

CSV_SEP = ","

log = get_logger(id="sampling")


def sld():
    return EMQ(LogisticRegression(), val_split=5)


def kdey():
    return KDEyML(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()


def gen_datasets():
    if PROBLEM == "binary":
        yield "IMDB", fetch_IMDBDataset()
        for dn in ["CCAT", "GCAT", "MCAT"]:
            dval = fetch_RCV1BinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        for dataset_name in _uci_names:
            yield dataset_name, fetch_UCIMulticlassDataset(dataset_name)
        for dataset_name in RCV1_MULTICLASS_DATASETS:
            yield dataset_name, fetch_RCV1MulticlassDataset(dataset_name)


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
    parent_dir = os.path.join(root_dir, cls_name, acc_name, dataset_name, L_prev)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


def gen_baselines(acc_fn):
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")


def gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
    yield "DoC", DoC(acc_fn, V2_prot, V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h)


# fmt: off
def gen_CAP_cont_table_opt(acc_fn, V2_prot, V2_prot_posteriors):
    pacc_lr_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }
    emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
    kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}

    yield "QuAcc(SLD)1xn2", GSCAP(QuAcc1xN2(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    yield "QuAcc(SLD)nxn", GSCAP(QuAccNxN(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if PROBLEM == "binary":
        yield "QuAcc(SLD)1xnp1", GSCAP(QuAcc1xNp1(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    yield "QuAcc(KDEy)1xn2", GSCAP(QuAcc1xN2(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    yield "QuAcc(KDEy)nxn", GSCAP(QuAccNxN(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if PROBLEM == "binary":
        yield "QuAcc(KDEy)1xnp1", GSCAP(QuAcc1xNp1(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
# fmt: on


def gen_methods(h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, V, V_posteriors
    for name, method in gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, V, V_posteriors
    # for name, method in gen_CAP_cont_table_opt(acc_fn, V2_prot, V2_prot_posteriors):
    #     yield name, method, V1, V1_posteriors


def get_method_names():
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_V2_prot = UPP(None)
    mock_V2_post = np.empty((1,))
    return (
        [m for m, _ in gen_baselines(mock_acc_fn)]
        + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
        # + [m for m, _ in gen_CAP_cont_table_opt(mock_acc_fn, mock_V2_prot, mock_V2_post)]
    )


def all_exist_pre_check(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(dataset_name, cls_name, method, acc, L)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def preload_existing(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    dfs = []
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(dataset_name, cls_name, method, acc, L)
        method_df = pd.read_csv(path, sep=CSV_SEP)
        dfs.append(method_df)

    return dfs


def experiments():
    log.info("-" * 31 + "  start  " + "-" * 31)

    dfs = []
    for (cls_name, h), (dataset_name, dataset) in gen_model_dataset(gen_classifiers, gen_datasets):
        for L, V, U in get_train_samples(dataset):
            # check if all results for current combination already exist
            # if so, skip the combination
            if all_exist_pre_check(dataset_name, cls_name, L):
                log.info(f"All results for {cls_name} over {dataset_name}[{prev_str(L)}] exist, skipping")
                dfs.extend(preload_existing(dataset_name, cls_name, L))
                continue

            # fit model
            log.info(f"Training {cls_name} over {dataset_name}[{prev_str(L)}]")
            h.fit(*L.Xy)

            # test generation protocol
            test_prot = APP(
                U,
                n_prevalences=21,
                repeats=NUM_TEST,
                return_type="labelled_collection",
                random_state=qp.environ["_R_SEED"],
            )

            # split validation set
            V1, V2_prot = split_validation(V)

            # generate posteriors
            V_posteriors = h.predict_proba(V.X)
            V1_posteriors = h.predict_proba(V1.X)
            V2_prot_posteriors = []
            for sample in V2_prot():
                V2_prot_posteriors.append(h.predict_proba(sample.X))

            # get posteriors for test samples
            test_prot_posteriors, test_prot_y_hat = [], []
            for sample in test_prot():
                P = h.predict_proba(sample.X)
                test_prot_posteriors.append(P)
                test_prot_y_hat.append(np.argmax(P, axis=-1))

            # precompute the actual accuracy values
            true_accs = {}
            for acc_name, acc_fn in gen_acc_measure(multiclass=PROBLEM == "multiclass"):
                true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

            L_prev = get_plain_prev(L.prevalence())
            for method_name, method, val, val_posteriors in gen_methods(
                h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors
            ):
                val_prev = get_plain_prev(val.prevalence())

                t_train = None
                for acc_name, acc_fn in gen_acc_measure(multiclass=PROBLEM == "multiclasss"):
                    path = local_path(dataset_name, cls_name, method_name, acc_name, L)

                    if os.path.exists(path):
                        method_df = pd.read_csv(path, sep=CSV_SEP)
                        dfs.append(method_df)
                        log.info(f"{method_name} on {acc_name} exists, skipping")
                        continue

                    try:
                        method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
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
                    method_df["classifier"] = cls_name
                    method_df["method"] = method_name
                    method_df["dataset"] = dataset_name
                    method_df["acc_name"] = acc_name
                    method_df["train_prev"] = np.around(L_prev, decimals=2)
                    method_df["val_prev"] = np.around(val_prev, decimals=2)
                    method_df["t_train"] = t_train
                    method_df["t_test_ave"] = t_test_ave

                    # TODO: add scores for GSCAP methods

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
    # print(pivot.to_latex())


if __name__ == "__main__":
    try:
        experiments()
    except Exception as e:
        log.error(e)
        print_exception(e)
