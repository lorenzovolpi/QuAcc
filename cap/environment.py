import os

env = {
    "OUT_DIR": os.getenv("CAP_OUT_DIR", "./output"),
    "CAP_DATA": os.getenv("CAP_DATA", os.path.expanduser("~/cap_data")),
    "QUAPY_DATA": os.getenv("CAP_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
    "SKLEARN_DATA": os.getenv(
        "CAP_SKLEARN_DATA", os.path.expanduser("~/scikit_learn_data")
    ),
    "N_JOBS": int(os.getenv("CAP_N_JOBS", -2)),
    "FORCE_NJOBS": int(os.getenv("CAP_FORCE_NJOBS", 0)) > 0,
}
