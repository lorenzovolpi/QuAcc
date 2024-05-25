import os
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from time import time

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import EMQ, PACC, KDEyML
from quapy.protocol import APP, UPP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import quacc as qc
import quacc.error
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1_macro, vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import N2E, QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import get_shift, true_acc

qp.environ["SAMPLE_SIZE"] = 1000
NUM_TEST = 100
qp.environ["_R_SEED"] = 0

CSV_SEP = ","

pacc_lr_params = {
    "q_class__classifier__C": np.logspace(-3, 3, 7),
    "q_class__classifier__class_weight": [None, "balanced"],
    # "add_X": [True, False],
    "add_posteriors": [True, False],
    "add_y_hat": [True, False],
    "add_maxconf": [True, False],
    "add_negentropy": [True, False],
    "add_maxinfsoft": [True, False],
}
emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}


def get_imdbs():
    train, U = qp.datasets.fetch_reviews(
        "imdb", tfidf=True, min_df=10, pickle=True, data_home=qc.env["QUAPY_DATA"]
    ).train_test

    train_prevs = np.linspace(0.1, 1, 9, endpoint=False)
    L_size = np.min(np.around(np.min(train.counts()) / train_prevs, decimals=0))

    datasets = [(*DP._split_train(train.sampling(int(L_size), p)), U) for p in train_prevs]
    return datasets


def local_path(method_name, acc_name, L: LabelledCollection):
    return os.path.join(qc.env["OUT_DIR"], "pg_imdb", f"{method_name}_{acc_name}_{round(L.prevalence()[1]*100)}.csv")


def gen_quants():
    yield "EMQ", EMQ(LogisticRegression(), val_split=5), emq_lr_params
    yield "KDEy", KDEyML(LogisticRegression()), kde_lr_params


def gen_methods(q, q_name, h, acc_fn, params, val_prot):
    yield f"QuAcc({q_name})1xn2", GSCAP(QuAcc1xN2(h, acc_fn, q), params, val_prot, acc_fn, refit=False)
    yield f"QuAcc({q_name})1xnp1", GSCAP(QuAcc1xNp1(h, acc_fn, q), params, val_prot, acc_fn, refit=False)
    yield f"QuAcc({q_name})nxn", GSCAP(QuAccNxN(h, acc_fn, q), params, val_prot, acc_fn, refit=False)


def main():
    dfs = []
    quant_methods = defaultdict(lambda: [])
    for L, V, U in get_imdbs():
        print(f"train prev: {L.prevalence()[1]}")
        V, val_prot = split_validation(V)
        h = LogisticRegression()
        # h_param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}
        # h = GridSearchCV(LogisticRegression(), h_param_grid, cv=5, n_jobs=-1)
        # test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
        test_prot = APP(
            U, n_prevalences=21, repeats=NUM_TEST, return_type="labelled_collection", random_state=qp.environ["_R_SEED"]
        )
        accs = [("vanilla_acc", vanilla_acc), ("f1", f1_macro)]

        h.fit(*L.Xy)
        results = []
        for acc_name, acc_fn in accs:
            print(f"{acc_name}")
            for q_name, q, params in gen_quants():
                for method_name, method in gen_methods(q, q_name, h, acc_fn, params, val_prot):
                    path = local_path(method_name, acc_name, L)
                    if os.path.exists(path):
                        method_df = pd.read_csv(path, sep=CSV_SEP)
                        dfs.append(method_df)
                        print(f"\t{method_name} exists, skipping")
                        continue
                    t_init = time()
                    method.fit(V)
                    test_shift = get_shift(np.array([Ui.prevalence() for Ui in test_prot()]), L.prevalence())
                    true_accs = np.array([true_acc(h, acc_fn, Ui) for Ui in test_prot()])
                    estim_accs = np.array([method.predict(Ui.X) for Ui in test_prot()])
                    ae = quacc.error.ae(true_accs, estim_accs)
                    t_method = time() - t_init
                    print(f"\t{method_name} took {t_method:.3f}s")
                    method_df = pd.DataFrame(
                        np.vstack([test_shift, true_accs, estim_accs, ae]).T,
                        columns=["shifts", "true_accs", "estim_accs", "acc_err"],
                    )
                    method_df["method"] = method_name
                    method_df["acc_name"] = acc_name
                    method_df["train_prev"] = np.around(L.prevalence(), decimals=2)[1]
                    method_df["fit_score"] = method.best_score_
                    method_df.to_csv(path, sep=CSV_SEP)
                    dfs.append(method_df)

                    quant_methods[q_name].append(method_name)

    results = pd.concat(dfs, axis=0)

    selected_methods_df = []
    for q_name, methods in quant_methods.items():
        _df = results.loc[results["method"].isin(methods)]
        _pivot = _df.pivot_table(values="fit_score", columns="method", index="train_prev")
        idx_max = _pivot.idxmin(axis=1)
        train_prevs, best_methods = idx_max.index.to_list(), idx_max.to_list()
        _dfs = []
        for tp, bm in zip(train_prevs, best_methods):
            _dfs.append(_df.loc[(_df["train_prevs"] == tp) & (_df["method"] == bm)])
        method_df = pd.concat(_dfs, axis=0)
        method_df["method"] = f"QuAcc({q_name})"
        selected_methods_df.append(method_df)

    results = pd.concat([results] + selected_methods_df, axis=0)

    print(results.groupby(["method", "acc_name"]).mean())


if __name__ == "__main__":
    main()
