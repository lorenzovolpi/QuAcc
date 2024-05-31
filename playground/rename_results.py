import os
from glob import glob

import quacc as qc

BASEDIR = "results"

replacements = {"EMQ": "SLD"}


def get_path(w_path):
    return os.path.join(qc.env["OUT_DIR"], BASEDIR, w_path)


if __name__ == "__main__":
    w_path = "**/QuAcc(EMQ)*.json"
    paths = glob(get_path(w_path), recursive=True)
    for f in paths:
        for r, s in replacements.items():
            if f.find(r) > 0:
                nf = f.replace(r, s)
                print(f"{f} -> {nf}")
                os.rename(f, nf)
