import itertools as IT
import os
from collections import defaultdict
from contextlib import ExitStack
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.functional import uniform_prevalence_sampling
from quapy.method.aggregative import EMQ, ClassifyAndCount, KDEyML
from quapy.protocol import APP, UPP
from sklearn.linear_model import LogisticRegression

import cap
from cap.data.datasets import (
    RCV1_MULTICLASS_DATASETS,
    fetch_IMDBDataset,
    fetch_RCV1BinaryDataset,
    fetch_RCV1MulticlassDataset,
)
from cap.error import f1, f1_macro, vanilla_acc
from cap.experiments.util import (
    fit_or_switch,
    get_logger,
    get_predictions,
    split_validation,
)
from cap.models.cont_table import LEAP, NaiveCAP, QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from cap.models.direct import ATC, DoC
from cap.models.model_selection import GridSearchCAP as GSCAP
from cap.utils.commons import get_shift, true_acc

PROJECT = "cap"

root_dir = os.path.join(cap.env["OUT_DIR"], PROJECT)
qp.environ["SAMPLE_SIZE"] = 1000
NUM_TEST = 100
N_PREVS = 21
qp.environ["_R_SEED"] = 0
PROBLEM = "binary"

CSV_SEP = ","

_toggle = {
    "cc_quacc": True,
    "sld_quacc": True,
    "kde_quacc": True,
    "sld_leap": False,
    "kde_leap": True,
    "vanilla": True,
    "f1": False,
}

log = get_logger(id=PROJECT)


def cc():
    return ClassifyAndCount()


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


def gen_acc_measure(multiclass=False):
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        yield "macro-F1", f1_macro if multiclass else f1


def gen_model_dataset(_gen_model, _gen_dataset):
    for model in _gen_model():
        for dataset in _gen_dataset():
            yield model, dataset


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        yield "IMDB", None if only_names else fetch_IMDBDataset()
        for dn in ["CCAT", "GCAT", "MCAT"]:
            yield dn, None if only_names else fetch_RCV1BinaryDataset(dn)
    elif PROBLEM == "multiclass":
        # _uci_skip = ["isolet", "wine-quality", "letter"]
        # _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        # for dataset_name in _uci_names:
        #     yield dataset_name, fetch_UCIMulticlassDataset(dataset_name)
        for dataset_name in RCV1_MULTICLASS_DATASETS:
            yield (
                dataset_name,
                None if only_names else fetch_RCV1MulticlassDataset(dataset_name),
            )


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_uniform_prevalences(n_classes, repeats, seed=None):
    qp_seed = qp.environ.get("_R_SEED", None)
    seed = qp_seed if seed is None else seed
    with ExitStack() as stack:
        if seed is not None:
            stack.enter_context(qp.util.temp_seed(seed))
        return uniform_prevalence_sampling(n_classes, repeats)


def get_train_samples(dataset):
    L, V, U = dataset

    if PROBLEM == "binary":
        train_prevs = np.linspace(0.1, 1, 9, endpoint=False)
        L_size = np.min(np.around(np.min(L.counts()) / train_prevs, decimals=0))
        V_size = np.min(np.around(np.min(V.counts()) / train_prevs, decimals=0))
        prev_datasets = [
            (p, (L.sampling(int(L_size), p), V.sampling(int(V_size), p), U))
            for p in train_prevs
        ]
    elif PROBLEM == "multiclass":
        train_prevs = np.around(
            get_uniform_prevalences(L.n_classes, 9)[:, 1:], decimals=4
        )
        L_size = len(L) // 2
        V_size = len(V) // 2
        prev_datasets = [
            (p, (L.sampling(int(L_size), *p), V.sampling(int(V_size), *p), U))
            for p in train_prevs
        ]

    return prev_datasets


def prev_str(train_prev):
    if PROBLEM == "binary":
        return str(round(train_prev * 100))
    elif PROBLEM == "multiclass":
        print(train_prev)
        print(np.around(train_prev, decimals=2))
        return "_".join([str(int(x)) for x in np.around(train_prev, decimals=2) * 100])


def local_path(dataset_name, cls_name, method_name, acc_name, train_prev):
    L_prev = prev_str(train_prev)
    parent_dir = os.path.join(
        root_dir, PROBLEM, cls_name, acc_name, dataset_name, L_prev
    )
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


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


# fmt: off
def gen_CAP_cont_table_opt(acc_fn, V2_prot, V2_prot_posteriors):
    cc_lr_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        "add_posteriors": [True],
        "add_y_hat": [False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }
    emq_lr_params = cc_lr_params | {"q_class__recalib": [None, "bcts"]}
    kde_lr_params = cc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}

    if _toggle["cc_quacc"]:
        yield "QuAcc(CC)1xn2", GSCAP(QuAcc1xN2(acc_fn, cc()), cc_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(CC)nxn", GSCAP(QuAccNxN(acc_fn, cc()), cc_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(CC)1xnp1", GSCAP(QuAcc1xNp1(acc_fn, cc()), cc_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if _toggle["sld_quacc"]:
        yield "QuAcc(SLD)1xn2", GSCAP(QuAcc1xN2(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(SLD)nxn", GSCAP(QuAccNxN(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        # if PROBLEM == "binary":
        yield "QuAcc(SLD)1xnp1", GSCAP(QuAcc1xNp1(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if _toggle["kde_quacc"]:
        yield "QuAcc(KDEy)1xn2", GSCAP(QuAcc1xN2(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(KDEy)nxn", GSCAP(QuAccNxN(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        # if PROBLEM == "binary":
        yield "QuAcc(KDEy)1xnp1", GSCAP(QuAcc1xNp1(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
# fmt: on


def is_excluded(classifier, dataset, method, acc):
    _excluded = [
        {
            "problem": "multiclass",
            "classifiers": get_classifier_names(),
            "datasets": get_dataset_names(),
            "methods": ["QuAcc(CC)1xnp1", "QuAcc(SLD)1xnp1", "QuAcc(KDEy)1xnp1"],
            "accs": ["macro-F1"],
        }
    ]

    for _ex in _excluded:
        if (
            _ex["problem"] == PROBLEM
            and classifier in _ex["classifiers"]
            and dataset in _ex["datasets"]
            and method in _ex["methods"]
            and acc in _ex["accs"]
        ):
            return True

    return False


def gen_methods(h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, V, V_posteriors
    for name, method in gen_baselines_vp(acc_fn, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, V, V_posteriors
    for name, method in gen_CAP_cont_table_opt(acc_fn, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors


def get_method_names():
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_V2_prot = UPP(None)
    mock_V2_post = np.empty((1,))
    return (
        [m for m, _ in gen_baselines(mock_acc_fn)]
        + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]
        + [
            m
            for m, _ in gen_CAP_cont_table_opt(mock_acc_fn, mock_V2_prot, mock_V2_post)
        ]
    )


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


def get_extra_from_method(method, df):
    if isinstance(method, GSCAP) and hasattr(method, "best_model_"):
        df["fit_score"] = method.best_score_
        _bp = method.best_params_
        df["add_cov"] = any(
            [_bp["add_maxconf"], _bp["add_negentropy"], _bp["add_maxinfsoft"]]
        )
        df["balance"] = _bp["q_class__classifier__class_weight"]
        if "q_class__recalib" in _bp:
            df["recalib"] = str(_bp["q_class__recalib"])


def add_prev_to_df(df, key, prevs):
    if PROBLEM == "binary":
        df[key] = np.around(prevs, decimals=2)
    elif PROBLEM == "multiclass":
        df[key] = [tuple(prevs.tolist())] * len(df)


def experiments():
    for (cls_name, h), (dataset_name, dataset) in gen_model_dataset(
        gen_classifiers, gen_datasets
    ):
        for train_prev, (L, V, U) in get_train_samples(dataset):
            # check if all results for current combination already exist
            # if so, skip the combination
            if all_exist_pre_check(dataset_name, cls_name, train_prev):
                log.info(
                    f"All results for {cls_name} over {dataset_name}[{prev_str(train_prev)}] exist, skipping"
                )
                continue

            # fit model
            log.info(f"Training {cls_name} over {dataset_name}[{prev_str(train_prev)}]")
            h.fit(*L.Xy)

            # test generation protocol
            if PROBLEM == "binary":
                test_prot = APP(
                    U,
                    n_prevalences=N_PREVS,
                    repeats=NUM_TEST,
                    return_type="labelled_collection",
                    random_state=qp.environ["_R_SEED"],
                )
            elif PROBLEM == "multiclass":
                test_prot = UPP(
                    U,
                    repeats=NUM_TEST * N_PREVS,
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

            for method_name, method, val, val_posteriors in gen_methods(
                h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors
            ):
                t_train = None
                for acc_name, acc_fn in gen_acc_measure(
                    multiclass=PROBLEM == "multiclasss"
                ):
                    if is_excluded(cls_name, dataset_name, method_name, acc_name):
                        continue
                    path = local_path(
                        dataset_name, cls_name, method_name, acc_name, train_prev
                    )
                    if os.path.exists(path):
                        log.info(f"{method_name} on {acc_name} exists, skipping")
                        continue

                    try:
                        method, _t_train = fit_or_switch(
                            method, val, val_posteriors, acc_fn, t_train is not None
                        )
                        t_train = t_train if _t_train is None else _t_train

                        test_shift = get_shift(
                            np.array([Ui.prevalence() for Ui in test_prot()]),
                            L.prevalence(),
                        )
                        estim_accs, t_test_ave = get_predictions(
                            method, test_prot, test_prot_posteriors, False
                        )
                    except Exception as e:
                        print_exception(e)
                        log.warning(
                            f"{method_name}[{prev_str(L)}]: {acc_name} gave error '{e}' - skipping"
                        )
                        continue

                    ae = cap.error.ae(
                        np.array(true_accs[acc_name]), np.array(estim_accs)
                    )
                    method_df = pd.DataFrame(
                        np.vstack([test_shift, true_accs[acc_name], estim_accs, ae]).T,
                        columns=["shifts", "true_accs", "estim_accs", "acc_err"],
                    )
                    method_df["classifier"] = cls_name
                    method_df["method"] = method_name
                    method_df["dataset"] = dataset_name
                    method_df["acc_name"] = acc_name
                    add_prev_to_df(method_df, "train_prev", train_prev)
                    method_df["t_train"] = t_train
                    method_df["t_test_ave"] = t_test_ave

                    get_extra_from_method(method, method_df)

                    log.info(f"{method_name} on {acc_name} done")
                    method_df.to_csv(path, sep=CSV_SEP)


def dataset_info():
    rows, vals = [], []
    for name, dataset in gen_datasets():
        rows.append(name)
        L, V, U = dataset
        _, (Li, Vi, _) = get_train_samples(dataset)[0]
        L_prev = np.around(L.prevalence(), decimals=3)
        assert np.isclose(L_prev.sum(), 1), f"invalid prevalence: {L_prev}"
        L_prev = str(L_prev[1:].tolist())
        vals.append([len(L), len(U), L.n_classes, L_prev, len(Li)])

    _info = pd.DataFrame(vals, index=rows, columns=["|T|", "|U|", "|Y|", "pT", "|Ti|"])
    _info.to_latex(os.path.join(root_dir, "dataset_info.tex"))
    log.info("Dataset info generated")


def selection_rates(df: pd.DataFrame):
    quants = ["SLD", "KDEy"]
    datasets = ["IMDB", "CCAT", "GCAT", "MCAT"]
    train_prevs = df["train_prev"].unique()
    series = defaultdict(lambda: [])
    for q, dataset in IT.product(quants, datasets):
        methods = [f"QuAcc({q})1xn2", f"QuAcc({q})nxn", f"QuAcc({q})1xnp1"]
        sel_df = df.loc[(df["method"].isin(methods)) & (df["dataset"] == dataset), :]
        n_best = (
            (
                pd.pivot_table(
                    sel_df,
                    index=["train_prev", "acc_name"],
                    columns=["method"],
                    values="fit_score",
                )
                .idxmin(axis="columns")
                .to_numpy()[:, np.newaxis]
                == np.tile(methods, (len(train_prevs) * 2, 1))
            )
            .sum(axis=0)
            .tolist()
        )
        n_bal = np.around(
            (sel_df["balance"].to_numpy() == "balanced").sum() / len(sel_df) * 100,
            decimals=2,
        )
        n_cov = np.around(
            sel_df["add_cov"].to_numpy().sum() / len(sel_df) * 100, decimals=2
        )
        series["QuAcc-1xn2"].append(n_best[0])
        series["QuAcc-nxn"].append(n_best[1])
        series["QuAcc-1xnp1"].append(n_best[2])
        series["balance"].append(n_bal)
        series["covariates"].append(n_cov)
        series["recalib"].append(np.nan)
        series["quant"].append(q)
        series["dataset"].append(dataset)
        if q == "SLD":
            n_recalib = np.around(
                (sel_df["recalib"].to_numpy() == "bcts").sum() / len(sel_df) * 100,
                decimals=2,
            )
            series["recalib"][-1] = n_recalib

    info = pd.DataFrame.from_dict(series)
    print(
        pd.pivot_table(
            info,
            index=["dataset", "quant"],
            values=[
                "QuAcc-1xn2",
                "QuAcc-nxn",
                "QuAcc-1xnp1",
                "balance",
                "covariates",
                "recalib",
            ],
        )
    )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        # results = load_results()
        # tables(results)
        # plots(results)
        # dataset_info()
        # selection_rates(results)
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
