import os

import quapy as qp

import quacc as qc

qc.env["OUT_DIR"] = os.getenv("EXP_OUT_DIR", "./output")
qc.env["QUACC_DATA"] = os.getenv("QUACC_DATA", qc.env["QUACC_DATA"])
qc.env["QUAPY_DATA"] = os.getenv("QUACC_QUAPY_DATA", qc.env["QUAPY_DATA"])
qc.env["SKLEARN_DATA"] = os.getenv("QUACC_SKLEARN_DATA", qc.env["SKLEARN_DATA"])
qc.env["N_JOBS"] = int(os.getenv("QUACC_N_JOBS", qc.env["N_JOBS"]))
_force_njobs = os.getenv("QUACC_FORCE_NJOBS", None)
if _force_njobs is not None:
    qc.env["FORCE_NJOBS"] = int(_force_njobs) > 0


PROJECT = "leap"
root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

_valid_problems = ["binary", "multiclass"]
PROBLEM = "binary"
