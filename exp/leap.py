import glob
import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

import quacc as qc
from exp.util import fit_or_switch, gen_model_dataset, get_logger, get_plain_prev, get_predictions, split_validation
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models.cont_table import LEAP, NaiveCAP
from quacc.models.direct import ATC, DoC
from quacc.utils.commons import get_shift, true_acc

PROJECT = "leap"

root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
qp.environ["SAMPLE_SIZE"] = 1000
NUM_TEST = 200
qp.environ["_R_SEED"] = 0
PROBLEM = "multiclass"
CSV_SEP = ","

_toggle = {
    "cc_quacc": True,
    "sld_quacc": True,
    "kde_quacc": True,
    "sld_leap": False,
    "kde_leap": True,
    "vanilla": True,
    "f1": True,
}

log = get_logger(id=PROJECT)


def sld():
    emq = EMQ(LogisticRegression(), val_split=5)
    emq.SUPPRESS_WARNINGS = True
    return emq


def kdey():
    return KDEyML(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        for dn in _uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        for dataset_name in _uci_names:
            yield dataset_name, None if only_names else fetch_UCIMulticlassDataset(dataset_name)


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def gen_acc_measure():
    multiclass = PROBLEM == "multiclass"
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        yield "macro-F1", f1_macro if multiclass else f1


def gen_baselines(acc_fn):
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")


def gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
    yield "DoC", DoC(acc_fn, V2_prot, V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    if _toggle["sld_leap"]:
        yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h)
    if _toggle["kde_leap"]:
        yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h)


def gen_methods(h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, V, V_posteriors
    for name, method in gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, V, V_posteriors


def get_method_names():
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_V2_prot = UPP(None)
    mock_V2_post = np.empty((1,))
    return (
        [m for m, _ in gen_baselines(mock_acc_fn)]
        + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
    )


def local_path(dataset_name, cls_name, method_name, acc_name, train_prev):
    parent_dir = os.path.join(root_dir, PROBLEM, cls_name, acc_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


def is_excluded(classifier, dataset, method, acc):
    return False


def all_exist_pre_check(dataset_name, cls_name, train_prev):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        if is_excluded(cls_name, dataset_name, method, acc):
            continue
        path = local_path(dataset_name, cls_name, method, acc, train_prev)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def experiments():
    for (cls_name, h), (dataset_name, (L, V, U)) in gen_model_dataset(gen_classifiers, gen_datasets):
        # check if all results for current combination already exist
        # if so, skip the combination
        if all_exist_pre_check(dataset_name, cls_name):
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
            continue

        # fit model
        log.info(f"Training {cls_name} over {dataset_name}")
        h.fit(*L.Xy)

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
                if is_excluded(cls_name, dataset_name, method_name, acc_name):
                    continue
                path = local_path(dataset_name, cls_name, method_name, acc_name)
                if os.path.exists(path):
                    log.info(f"{method_name} on {acc_name} exists, skipping")
                    continue

                try:
                    method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
                    t_train = t_train if _t_train is None else _t_train

                    test_shift = get_shift(np.array([Ui.prevalence() for Ui in test_prot()]), L.prevalence())
                    estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors, False)
                except Exception as e:
                    print_exception(e)
                    log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
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
                method_df["train_prev"] = [L_prev] * len(method_df)
                method_df["val_prev"] = [val_prev] * len(method_df)
                method_df["t_train"] = t_train
                method_df["t_test_ave"] = t_test_ave

                log.info(f"{method_name} on {acc_name} done [{t_train=}s; {t_test_ave=}s]")
                method_df.to_csv(path, sep=CSV_SEP)


def load_results() -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.csv"), recursive=True):
        dfs.append(pd.read_csv(path, sep=CSV_SEP))
    return pd.concat(dfs, axis=0)


if __name__ == "__main__":
    pass
