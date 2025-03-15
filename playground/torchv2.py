import os
from collections import defaultdict
from time import time

import numpy as np
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from scipy.special import softmax
from sklearn.base import BaseEstimator

qp.environ["_R_SEED"] = 0


class BaseEstimatorAdapter(BaseEstimator):
    def __init__(self, V_hidden_states, U_hidden_states, V_logits, U_logits):
        self.hs = np.vstack([V_hidden_states, U_hidden_states])
        self.logits = np.vstack([V_logits, U_logits])

        hashes = self._hash(self.hs)
        self._dict = defaultdict(lambda: [])
        for i, hash in enumerate(hashes):
            self._dict[hash].append(i)

    def _hash(self, X):
        return np.around(np.abs(X).sum(axis=-1) * self.hs.shape[0])

    def predict_proba(self, X: np.ndarray):
        def f(data, hash):
            _ids = np.array(self._dict[hash])
            _m = self.hs[_ids, :]
            _eq_idx = np.nonzero((_m == data).all(axis=-1))[0][0]
            return _ids[_eq_idx]

        hashes = self._hash(X)
        logits_idx = np.vectorize(f, signature="(m),()->()")(X, hashes)
        _logits = self.logits[logits_idx, :]
        return softmax(_logits, axis=-1)

    def decision_function(self, X: np.ndarray):
        return self.predict_proba(X)


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

    t0 = time()
    V_P = model.decision_function(V.X)
    print(V_P)
    print(V_P.shape, type(V_P))
    t1 = time()
    print("val.:", f"{t1 - t0:.3f}s")
    U_Ps = [model.decision_function(Ui.X) for Ui in prot()]
    t2 = time()
    print("test (avg.):", f"{(t2 - t1) / prot.repeats:.3f}s")

    q = KDEyML(model)
    q.fit(V, fit_classifier=False)
    print([q.quantify(Ui.X) for Ui in prot()])
