import os

env = {
    "OUT_DIR": os.getenv("QUACC_OUT_DIR", "."),
    "N_JOBS": -2,
}
