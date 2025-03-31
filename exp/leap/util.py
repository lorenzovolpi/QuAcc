import glob
import os
from pathlib import Path

import pandas as pd

from exp.leap.config import PROBLEM, get_method_names, root_dir


def load_results() -> pd.DataFrame:
    dfs = []
    _methods = get_method_names()
    for path in glob.glob(os.path.join(root_dir, PROBLEM, "**", "*.json"), recursive=True):
        if Path(path).stem in _methods:
            dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def rename_datasets(mapping, df, datasets):
    _datasets = [mapping.get(d, d) for d in datasets]
    for d, rd in mapping.items():
        df.loc[df["dataset"] == d, "dataset"] = rd
    return df, _datasets


def rename_methods(mapping, df, methods, baselines=None):
    _methods = [mapping.get(m, m) for m in methods]
    for m, rm in mapping.items():
        df.loc[df["method"] == m, "method"] = rm

    if baselines is None:
        return df, _methods
    else:
        _baselines = [mapping.get(b, b) for b in baselines]
        return df, _methods, _baselines
