from pdb import post_mortem

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

# from abstention.calibration import TempScaling
from calibration.ts import TempScaling
from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import vanilla_acc
from quacc.models.cont_table import QuAcc1xN2

dataset_name = "haberman"

qp.environ["_R_SEED"] = 0


def lblM_from_P(P, n_classes):
    return np.eye(n_classes)[np.argmax(P, axis=-1)]


def calib_testing():
    ts = TempScaling(verbose=False)
    # bcts = TempScaling(bias_positions="all")

    L, V, U = fetch_UCIBinaryDataset(dataset_name)
    h = LogisticRegression().fit(*L.Xy)

    V_P = h.predict_proba(V.X)
    ts_fn = ts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
    # bcts_fn = bcts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)

    test_prot = UPP(U, sample_size=100, repeats=100, return_type="labelled_collection")
    test_P = [h.predict_proba(U_i.X) for U_i in test_prot()]

    test_ts_Ps = [ts_fn(P_i) for P_i in test_P]
    # test_bcts_Ps = [bcts_fn(P_i) for P_i in test_P]

    print(np.hstack([np.mean(np.abs(test_Pi - test_ts_Pi)) for test_Pi, test_ts_Pi in zip(test_P, test_ts_Ps)]))


def recalib_quacc():
    dfs = []
    for dataset_name in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        h = LogisticRegression().fit(*L.Xy)

        V_P = h.predict_proba(V.X)

        q = QuAcc1xN2(vanilla_acc, KDEyML(), add_maxinfsoft=False).fit(V, V_P)

        test_prot = UPP(U, sample_size=100, return_type="labelled_collection")
        test_P = [h.predict_proba(U_i.X) for U_i in test_prot()]
        true_acc = np.array([vanilla_acc(np.argmax(P_i, axis=-1), U_i.y) for P_i, U_i in zip(test_P, test_prot())])

        def add_df(P, method):
            _preds = np.array([q.predict(U_i.X, P_i) for U_i, P_i in zip(test_prot(), P)])
            _acc_err = np.abs(_preds - true_acc)[:, None]
            _df = pd.DataFrame(_acc_err, columns=["acc_err"])
            _df["dataset"] = dataset_name
            _df["method"] = method
            dfs.append(_df)

        add_df(test_P, "base")

        test_P_r = [post for _, post in [EMQ.EM(V.prevalence(), P_i) for P_i in test_P]]
        add_df(test_P_r, "em")

        ts = TempScaling()(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
        test_P_r = [ts(P_i) for P_i in test_P]
        add_df(test_P_r, "ts")

        print(dataset_name)

    df = pd.concat(dfs, axis=0)
    # print(df.groupby(["dataset"]).mean())
    print(df.pivot_table(index="dataset", columns="method", values="acc_err"))


if __name__ == "__main__":
    # calib_testing()
    recalib_quacc()
