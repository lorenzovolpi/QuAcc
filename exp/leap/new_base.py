import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS
from sklearn.linear_model import LogisticRegression

import quacc as qc
from exp.leap.config import DatasetBundle, gen_acc_measure, kdey_mlp, root_dir
from exp.leap.util import gen_method_df
from exp.util import fit_or_switch, get_ct_predictions
from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.models.cont_table import CBPE, OCE
from quacc.models.direct import ATC, DoC
from quacc.utils.commons import get_shift, true_acc

qp.environ["SAMPLE_SIZE"] = 100


def gen_methods(D: DatasetBundle):
    _, acc_fn = next(gen_acc_measure())
    yield "ATC", ATC(acc_fn, scoring_fn="maxconf"), D.V, D.V_posteriors
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors), D.V1, D.V1_posteriors
    # yield "DS", DispersionScore(acc_fn), D.V, D.V_posteriors
    # yield "COT", COT(acc_fn), D.V, D.V_posteriors
    # yield "COTT", COTT(acc_fn), D.V, D.V_posteriors
    yield "CBPE", CBPE(acc_fn), D.V, D.V_posteriors
    yield "O-LEAP", OCE(acc_fn, kdey_mlp()), D.V, D.V_posteriors


if __name__ == "__main__":
    dfs = []
    for dataset_name in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        h = LogisticRegression().fit(*L.Xy)
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(h)

        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]

        for method_name, method, val, val_posteriors in gen_methods(D):
            t_train = None
            for acc_name, acc_fn in gen_acc_measure():
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
                    method=method_name,
                    dataset=dataset_name,
                    acc_name=acc_name,
                )
                dfs.append(method_df)

        print(f"{dataset_name} OK")

    df = pd.concat(dfs, axis=0)

    pivot = pd.pivot_table(df, index=["dataset"], columns=["method"], values="acc_err")
    print(pivot)

    # parent_dir = os.path.join(root_dir, "plots")
    # os.makedirs(parent_dir, exist_ok=True)
    # plot_shift(df, basedir=parent_dir, filename=f"new_base_{PROBLEM}")

    parent_dir = os.path.join(root_dir, "tables")
    path = os.path.join(parent_dir, "new_base.html")
    with open(path, "w") as f:
        pivot.to_html(f)
