import glob
import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import EMQ, HDy, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import quacc as qc
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_logger,
    get_plain_prev,
    get_predictions,
    split_validation,
    timestamp,
)
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models.cont_table import LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, DoC
from quacc.table import Format, Table
from quacc.utils.commons import get_shift, true_acc

PROJECT = "leap"

root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
qp.environ["SAMPLE_SIZE"] = 100
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
PROBLEM = "binary"
CSV_SEP = ","

_toggle = {
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


def kdey_auto():
    return KDEyML(LogisticRegression(), bandwidth="auto")


def hdy():
    return HDy(LogisticRegression())


def gen_classifiers():
    yield "LR", LogisticRegression()
    # yield "kNN", KNN(n_neighbors=10)
    # yield "SVM", SVC(kernel="rbf", probability=True)
    # yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=qp.environ["_R_SEED"])


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
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


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
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
    yield "OCE(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP")
    yield "OCE(KDEy)-SLSQP-c", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP-c")
    yield "OCE(KDEy)-L-BFGS-B", OCE(acc_fn, kdey(), reuse_h=h, optim_method="L-BFGS-B")
    # yield "OCE(KDEy)-BFGS", OCE(acc_fn, kdey(), reuse_h=h, optim_method="BFGS")
    # yield "LEAP(KDEy-a)", LEAP(acc_fn, kdey_auto(), reuse_h=h, log_true_solve=True)
    # yield "PHD(KDEy-a)", PHD(acc_fn, kdey_auto(), reuse_h=h)
    # yield "OCE(KDEy-a)-SLSQP", OCE(acc_fn, kdey_auto(), reuse_h=h, optim_method="SLSQP")
    # if PROBLEM == "binary":
    #     yield "LEAP(HDy)", LEAP(acc_fn, hdy(), reuse_h=h, log_true_solve=True)
    #     yield "PHD(HDy)", PHD(acc_fn, hdy(), reuse_h=h)
    #     yield "OCE(HDy)-SLSQP", OCE(acc_fn, hdy(), reuse_h=h, optim_method="SLSQP")


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


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, cls_name, acc_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


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
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        L_prev = get_plain_prev(L.prevalence())
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

                    test_shift = get_shift(np.array([Ui.prevalence() for Ui in test_prot()]), L.prevalence())
                    estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors)
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

                get_extra_from_method(method_df, method)

                log.info(f"{method_name} on {acc_name} done [{timestamp(t_train, t_test_ave)}]")
                method_df.to_csv(path, sep=CSV_SEP)


def load_results() -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.csv"), recursive=True):
        dfs.append(pd.read_csv(path, sep=CSV_SEP))
    return pd.concat(dfs, axis=0)


def tables():
    res = load_results()

    def gen_table(df: pd.DataFrame, name, datasets, methods):
        tbl = Table(name=name, benchmarks=datasets, methods=methods)
        tbl.format = Format(
            mean_prec=4, show_std=True, remove_zero=True, with_rank_mean=False, with_mean=False, color=True
        )
        tbl.format.mean_macro = False
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    classifiers = res["classifier"].unique()
    datasets = get_dataset_names()
    methods = get_method_names()
    accs = res["acc_name"].unique()

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        _df = res.loc[(res["classifier"] == classifier) & (res["acc_name"] == acc), :]
        name = f"{PROBLEM}_{classifier}_{acc}"
        tbls.append(gen_table(_df, name, datasets, methods))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False)


def leap_true_solve():
    res = load_results()
    methods = ["LEAP(KDEy)", "LEAP(KDEy-a)", "LEAP(HDy)"]
    md_path = os.path.join(root_dir, "tables", f"{PROBLEM}_true_solve.md")

    pd.pivot_table(
        res.loc[res["method"].isin(methods)],
        columns=["classifier", "method"],
        index=["dataset"],
        values="true_solve",
    ).to_markdown(md_path)


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        # experiments()
        tables()
        # leap_true_solve()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
