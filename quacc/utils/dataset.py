import json
import os
import urllib.request
from collections import defaultdict

import numpy as np

import quacc as qc

RCV1_HIERARCHY_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a03-expanded-topics-hierarchy/rcv1.topics.hier.expanded"
RCV1_CLASS_NAMES_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt"


def get_rcv1_class_info():
    json_path = os.path.join(qc.env["QUACC_DATA"], "rcv1_class_info.json")
    if not os.path.exists(json_path):
        # retrieve hierarchy file
        hierarchy_tmp_path = os.path.join(qc.env["QUACC_DATA"], "rcv1_hierarchy.tmp")
        urllib.request.urlretrieve(RCV1_HIERARCHY_URL, filename=hierarchy_tmp_path)
        tree = defaultdict(lambda: [])
        with open(hierarchy_tmp_path, "r") as tf:
            lines = tf.readlines()
        for line in lines:
            tokens = [s for s in line.strip().split(" ") if len(s) > 0]
            parent, child = tokens[1], tokens[3]
            if parent == "None":
                continue
            tree[parent].append(child)

        # retrieve class names files
        cn_tmp_path = os.path.join(qc.env["QUACC_DATA"], "rcv1_class_names.tmp")
        urllib.request.urlretrieve(RCV1_CLASS_NAMES_URL, filename=cn_tmp_path)
        class_names = []
        with open(cn_tmp_path, "r") as tf:
            lines = tf.readlines()
        for line in lines:
            class_names.append(line.strip())

        with open(json_path, "w") as jf:
            json.dump(dict(class_names=class_names, tree=tree), jf, indent=2)

        if os.path.exists(hierarchy_tmp_path):
            os.remove(hierarchy_tmp_path)
        if os.path.exists(cn_tmp_path):
            os.remove(cn_tmp_path)

    with open(json_path, "r") as jf:
        class_info = json.load(jf)

    class_names = np.array(class_info["class_names"])
    tree, index = {}, {}
    for parent, children in {k: np.array(v) for k, v in class_info["tree"].items()}.items():
        idxs = np.where(np.in1d(class_names, children))[0]
        if len(idxs) == len(children):
            tree[parent] = children
            index[parent] = idxs

    return class_names, tree, index
