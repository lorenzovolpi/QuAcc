import json

import numpy as np
from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import quacc as qc
from quacc.dataset import RCV1_MULTICLASS_DATASETS
from quacc.utils.dataset import get_rcv1_class_info

_, _, index = get_rcv1_class_info()

training = fetch_rcv1(subset="train", data_home=qc.env["SKLEARN_DATA"])
test = fetch_rcv1(subset="test", data_home=qc.env["SKLEARN_DATA"])

tr_len = training.target.shape[0]
te_len = test.target.shape[0]
print(f"{tr_len=} {te_len=}")


def parse_labels(labels):
    valid = np.nonzero(np.sum(labels, axis=-1) <= 1)[0]
    labels = labels[valid, :]
    ones, nonzero_vals = np.where(labels == np.ones((len(valid), 1)))
    labels = np.sum(labels, axis=-1)
    labels[ones] = nonzero_vals
    return valid, labels


report = {}
for target in RCV1_MULTICLASS_DATASETS:
    class_idx = index[target]
    tr_labels = training.target[:, class_idx].toarray()
    te_labels = test.target[:, class_idx].toarray()
    tr_valid, tr_labels = parse_labels(tr_labels)
    te_valid, te_labels = parse_labels(te_labels)
    classes = np.unique(np.hstack([tr_labels, te_labels]))
    n_tr_invalid, n_te_invalid = tr_len - len(tr_valid), te_len - len(te_valid)
    T = LabelledCollection(training.data[tr_valid, :], tr_labels, classes=classes)
    U = LabelledCollection(test.data[te_valid, :], te_labels, classes=classes)
    tr_prev, te_prev = T.prevalence(), U.prevalence()
    tr_counts, te_counts = T.counts(), U.counts()
    report[target] = dict(
        classes=classes.tolist(),
        n_tr_invalid=n_tr_invalid,
        n_te_invalid=n_te_invalid,
        tr_prev=tr_prev.tolist(),
        te_prev=te_prev.tolist(),
        tr_counts=tr_counts.tolist(),
        te_counts=te_counts.tolist(),
    )

with open("playground/rcv1_multi.json", "w") as f:
    json.dump(report, f, indent=2)
