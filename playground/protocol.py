import numpy as np
from quapy.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quapy.functional import uniform_prevalence_sampling, uniform_simplex_sampling
from quapy.protocol import APP, UPP

if __name__ == "__main__":
    L, U = fetch_UCIMulticlassDataset("digits").train_test
    prevs_u = uniform_prevalence_sampling(L.n_classes, 9)
    print(prevs_u)
    print(L.counts() / prevs_u)
