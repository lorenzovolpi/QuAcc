import os

env = {
    "OUT_DIR": os.getenv("QUACC_OUT_DIR", "."),
    "QUAPY_DATA": os.getenv("QUACC_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
    "SKLEARN_DATA": os.getenv("QUACC_SKLEARN_DATA", os.path.expanduser("~/scikit_learn_data")),
    "N_JOBS": -2,
}
