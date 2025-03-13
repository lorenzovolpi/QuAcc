import os
from time import time

import numpy as np
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.protocol import UPP
from sklearn.base import BaseEstimator

qp.environ["_R_SEED"] = 0


class BaseEstimatorAdapter(BaseEstimator):
    def __init__(self, V_hidden_states, U_hidden_states, V_logits, U_logits):
        self.hs = np.vstack([V_hidden_states, U_hidden_states])
        self.logits = np.vstack([V_logits, U_logits])

    def decision_function(self, X: np.ndarray):
        nzs_row, nzs_col = np.nonzero((X[:, np.newaxis, :] == self.hs).all(axis=-1))
        found_row_idx = np.searchsorted(nzs_row, np.unique(nzs_row))
        logits_idx = nzs_col[found_row_idx]
        print(logits_idx)
        return self.logits[logits_idx, :]

        # def find_idx(a):
        #     return np.nonzero((self.V_hs == a).all(axis=-1))[0][0]
        #
        # idxs = np.vectorize(find_idx, signature="(m,n)->(m)")(X)
        # print(idxs)


if __name__ == "__main__":
    dataset_name = "imdb"
    model_name = "bert-base-uncased"
    parent_dir = os.path.join("embeds", dataset_name, model_name)

    V_hidden_states = torch.load(os.path.join(parent_dir, "hidden_states.validation.pt")).numpy()
    V_logits = torch.load(os.path.join(parent_dir, "logits.validation.pt")).numpy()
    U_hidden_states = torch.load(os.path.join(parent_dir, "hidden_states.test.pt")).numpy()
    U_logits = torch.load(os.path.join(parent_dir, "logits.test.pt")).numpy()

    V = LabelledCollection(V_hidden_states, np.argmax(V_logits, axis=-1), [0, 1])
    U = LabelledCollection(U_hidden_states, np.argmax(U_logits, axis=-1), [0, 1])

    model = BaseEstimatorAdapter(V_hidden_states, U_hidden_states, V_logits, U_logits)

    prot = UPP(U, sample_size=1000, repeats=100, random_state=0, return_type="labelled_collection")

    start = time()
    V_P = model.decision_function(V.X)
    U_Ps = [model.decision_function(Ui.X) for Ui in prot()]
    print(time() - start)
