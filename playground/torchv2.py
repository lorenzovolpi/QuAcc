import os

import numpy as np
import quapy as qp
import torch
from sklearn.base import BaseEstimator

qp.environ["_R_SEED"] = 0


class BaseEstimatorAdapter(BaseEstimator):
    def __init__(self, V_hidden_states, U_hidden_states, V_logits, U_logits):
        self.V_hs = V_hidden_states
        self.U_hs = U_hidden_states
        self.V_logits = V_logits
        self.U_logits = U_logits

    def decision_function(self, X: np.ndarray):
        print(X[:, np.newaxis, :])
        print(np.nonzero((X[:, np.newaxis, :] == self.V_hs).all(axis=-1).sum(axis=0))[0])

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

    model = BaseEstimatorAdapter(V_hidden_states, U_hidden_states, V_logits, U_logits)

    test_X = V_hidden_states[[12, 115, 257], :]
    print(test_X)
    model.decision_function(test_X)
