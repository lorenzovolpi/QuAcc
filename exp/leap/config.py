import os

import quapy as qp

import quacc as qc

PROJECT = "leap"

root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
qp.environ["SAMPLE_SIZE"] = 100
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

PROBLEM = "multiclass"

_toggle = {
    "vanilla": True,
    "f1": True,
}
