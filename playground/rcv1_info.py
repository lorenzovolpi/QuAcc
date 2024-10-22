import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import quacc as qc
from quacc.data.datasets import fetch_RCV1MulticlassDataset
from quacc.data.util import get_rcv1_class_info

qp.environ["_R_SEED"] = 0

if __name__ == "__main__":
    cns, tree, index = get_rcv1_class_info()

    for name in cns:
        if name not in index:
            print(f"{name} - excluded")
            continue

        T, V, U = fetch_RCV1MulticlassDataset(name)
        print(
            f"{name} - Tsize: {len(T)}; Tprev: {T.prevalence()}; Usize: {len(U)}; T/Uclasses: {T.n_classes}/{U.n_classes}"
        )
