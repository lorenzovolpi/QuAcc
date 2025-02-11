import glob
import os

import pandas as pd

from exp.leap.config import CSV_SEP, PROBLEM, root_dir


def load_results() -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.csv"), recursive=True):
        dfs.append(pd.read_csv(path, sep=CSV_SEP))
    return pd.concat(dfs, axis=0)
