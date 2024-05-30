import os
from contextlib import redirect_stderr, redirect_stdout
from glob import glob
from time import time

import numpy as np
import pandas as pd
import quapy as qp
from quapy.method.aggregative import EMQ, PACC, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import quacc as qc
import quacc.error
from quacc.dataset import RCV1_MULTICLASS_DATASETS
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1_macro, vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import N2E, QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.models.requa import ReQua
from quacc.utils.commons import parallel as qc_parallel
from quacc.utils.commons import true_acc

NUM_TEST = 1000
qp.environ["_R_SEED"] = 0


CSV_SEP = ","
LOCAL_DIR = os.path.join(qc.env["OUT_DIR"], "pg_results", "requa")
CONFIG = "multiclass"
VERBOSE = False


def sld():
    return EMQ(LogisticRegression(), val_split=5)


def kdey():
    return KDEyML(LogisticRegression())


def get_bin_quaccs(h, acc_fn):
    return [
        QuAcc1xN2(h, acc_fn, sld()),
        QuAcc1xNp1(h, acc_fn, sld()),
        QuAccNxN(h, acc_fn, sld()),
    ]


def get_multi_quaccs(h, acc_fn):
    return [
        QuAcc1xN2(h, acc_fn, sld()),
        QuAccNxN(h, acc_fn, sld()),
    ]


if CONFIG == "multiclass":
    get_quaccs = get_multi_quaccs
    qp.environ["SAMPLE_SIZE"] = 250
    basedir = CONFIG
elif CONFIG == "binary":
    get_quaccs == get_bin_quaccs
    qp.environ["SAMPLE_SIZE"] = 1000
    basedir = CONFIG


# fmt: off

def gen_methods(h, acc_fn):
    quacc_params = {
        # "q_class__classifier__C": np.logspace(-3, 3, 7),
        # "q_class__classifier__class_weight": [None, "balanced"],
        "add_X": [True, False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }
    sample_s = qp.environ["SAMPLE_SIZE"]
    yield "ReQua", ReQua(h, acc_fn, get_quaccs(h, acc_fn), param_grid=quacc_params, sample_size=sample_s, verbose=VERBOSE)
    yield "ReQua-linfeat", ReQua(h, acc_fn, get_quaccs(h, acc_fn), param_grid=quacc_params, sample_size=sample_s, add_linear_features=True, verbose=VERBOSE)

# fmt: on


def get_local_path(dataset, method_name, acc_name):
    return os.path.join(LOCAL_DIR, basedir, f"{dataset}_{method_name}_{acc_name}.csv")


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    # dataset_names = ["C18", "E1"]
    dataset_names = RCV1_MULTICLASS_DATASETS
    dfs = []
    for dataset_name in dataset_names:
        L, V, U = DP.rcv1_multiclass(dataset_name)
        V, val_prot = split_validation(V)
        # h = LogisticRegression()
        h_param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}
        h_name, h = "LR-OPT", GridSearchCV(LogisticRegression(), h_param_grid, cv=5, n_jobs=-1)
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
        accs = [("vanilla_acc", vanilla_acc), ("f1", f1_macro)]

        h.fit(*L.Xy)
        print(f"trained {h_name} trained over {dataset_name}")
        for acc_name, acc_fn in accs:
            for method_name, method in gen_methods(h, acc_fn):
                local_path = get_local_path(dataset_name, method_name, acc_name)

                if os.path.exists(local_path):
                    method_df = pd.read_csv(local_path, sep=CSV_SEP)
                    dfs.append(method_df)
                    print(f"method {method_name} for {acc_name} exists, skipping")
                    continue

                t_init = time()
                method.fit(V)
                true_accs = np.array([true_acc(h, acc_fn, Ui) for Ui in test_prot()])
                estim_accs = method.batch_predict(test_prot)

                ae = quacc.error.ae(true_accs, estim_accs)
                t_method = time() - t_init
                print(f"method {method_name} for {acc_name} took {t_method:.3f}s")

                method_df = pd.DataFrame(
                    np.vstack([true_accs, estim_accs, ae]).T,
                    columns=["true_accs", "estim_accs", "ae"],
                )
                method_df["method"] = method_name
                method_df["acc_name"] = acc_name
                method_df["dataset"] = dataset_name
                method_df.to_csv(local_path, sep=CSV_SEP)
                dfs.append(method_df)

    results = pd.concat(dfs)

    print(results.pivot_table(values="ae", index="dataset", columns=["method", "acc_name"]))


def clean_results(path="*.csv"):
    glob_path = os.path.join(LOCAL_DIR, basedir, path)
    for f in glob(glob_path):
        os.remove(f)


if __name__ == "__main__":
    # clean_results()
    main()
