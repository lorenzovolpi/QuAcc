from sklearn.linear_model import LogisticRegression
from quacc.dataset import DatasetProvider as DP
from quapy.data.datasets import UCI_BINARY_DATASETS

from quacc.models.direct import KFCV
from quacc.error import vanilla_acc

if __name__ == "__main__":
    for d in UCI_BINARY_DATASETS:
        L, V, U = DP.uci_binary(d)
        h = LogisticRegression()
        h.fit(*L.Xy)

        m = KFCV(h, vanilla_acc)
        m.fit(V)

        print(f"{d}: {m.cv_score}")
