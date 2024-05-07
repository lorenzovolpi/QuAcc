from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import quacc as qc

train = fetch_rcv1(data_home=qc.env["SKLEARN_DATA"], subset="train")
test = fetch_rcv1(data_home=qc.env["SKLEARN_DATA"], subset="test")

stats = {}
class_names = train.target_names.tolist()
for target in class_names:
    class_idx = class_names.index(target)
    tr_labels = train.target[:, class_idx].toarray().flatten()
    tr = LabelledCollection(train.data, tr_labels)
    stats[target] = tr.prevalence()

with open("playground/rcv1.info", "w") as f:
    for target, prev in sorted(stats.items(), key=lambda x: x[1][0]):
        f.write(f"{target}: {prev}\n")
