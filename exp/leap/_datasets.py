from collections import defaultdict

import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_MULTICLASS_DATASETS

from quacc.data.datasets import fetch_UCIMulticlassDataset, sort_datasets_by_size

qp.environ["_R_SEED"] = 0


if __name__ == "__main__":
    print(sort_datasets_by_size(UCI_MULTICLASS_DATASETS, fetch_UCIMulticlassDataset))
    data = defaultdict(lambda: [])
    for d in UCI_MULTICLASS_DATASETS:
        L, V, U = fetch_UCIMulticlassDataset(d)
        data["name"].append(d)
        data["L"].append(len(L))
        data["V"].append(len(V))
        data["U"].append(len(U))
        data["len"].append(len(L) + len(V) + len(U))

    df = pd.DataFrame.from_dict(data).sort_values(by="len", ascending=False)
    print(df)
