import os
from glob import glob

from dotenv import load_dotenv

for _env in glob("./*.env"):
    load_dotenv(_env)

env = {
    "QUACC_DATA": os.path.expanduser("~/quacc_data"),
    "QUAPY_DATA": os.path.expanduser("~/quapy_data"),
    "SKLEARN_DATA": os.path.expanduser("~/scikit_learn_data"),
    "N_JOBS": -2,
    "FORCE_NJOBS": True,
}
