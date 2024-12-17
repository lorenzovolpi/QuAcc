import numpy as np
import pandas as pd
import quapy as qp
from abstention.calibration import TempScaling
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import vanilla_acc
from quacc.models.cont_table import QuAcc1xN2

dataset_name = "haberman"

qp.environ["_R_SEED"] = 0


def lblM_from_P(P, n_classes):
    return np.eye(n_classes)[np.argmax(P, axis=-1)]


def calib_testing():
    ts = TempScaling()
    bcts = TempScaling(bias_positions="all")

    L, V, U = fetch_UCIBinaryDataset(dataset_name)
    h = LogisticRegression().fit(*L.Xy)

    V_P = h.predict_proba(V.X)
    ts_fn = ts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
    bcts_fn = bcts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)

    test_prot = UPP(U, sample_size=100, repeats=100, return_type="labelled_collection")
    test_P = [h.predict_proba(U_i.X) for U_i in test_prot()]

    test_ts_Ps = [ts_fn(P_i) for P_i in test_P]
    test_bcts_Ps = [bcts_fn(P_i) for P_i in test_P]

    print(
        np.array(
            [np.mean(np.abs(test_Pi - test_ts_Pi), axis=0) for test_Pi, test_ts_Pi in zip(test_P, test_ts_Ps)]
        ).mean(axis=0)
    )
    print(
        np.array(
            [np.mean(np.abs(test_Pi - test_bcts_Pi), axis=0) for test_Pi, test_bcts_Pi in zip(test_P, test_bcts_Ps)]
        ).mean(axis=0)
    )


def recalib_quacc():
    dfs = []
    for dataset_name in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        h = LogisticRegression().fit(*L.Xy)

        V_P = h.predict_proba(V.X)

        q = QuAcc1xN2(vanilla_acc, KDEyML(), add_maxinfsoft=True).fit(V, V_P)
        qr = QuAcc1xN2(vanilla_acc, KDEyML(), add_maxinfsoft=True).fit(V, V_P)

        test_prot = UPP(U, sample_size=100, return_type="labelled_collection")
        test_P = [h.predict_proba(U_i.X) for U_i in test_prot()]
        test_P_r = [post for _, post in [EMQ.EM(V.prevalence(), P_i) for P_i in test_P]]
        true_acc = [vanilla_acc(np.argmax(P_i, axis=-1), U_i.y) for P_i, U_i in zip(test_P, test_prot())]

        q_acc_err = np.array(
            [np.abs(q.predict(U_i.X, P_i) - ta) for U_i, P_i, ta in zip(test_prot(), test_P, true_acc)]
        )
        qr_acc_err = np.array(
            [np.abs(qr.predict(U_i.X, P_i) - ta) for U_i, P_i, ta in zip(test_prot(), test_P_r, true_acc)]
        )
        bias = q_acc_err - qr_acc_err

        df_dataset = pd.DataFrame(
            np.hstack([q_acc_err[:, np.newaxis], qr_acc_err[:, np.newaxis], bias[:, np.newaxis]]),
            columns=["q", "qr", "bias"],
        )
        df_dataset["dataset"] = dataset_name
        dfs.append(df_dataset)
        print(dataset_name)

    df = pd.concat(dfs, axis=0)
    print(df.groupby(["dataset"]).mean())


if __name__ == "__main__":
    # calib_testing()
    recalib_quacc()
