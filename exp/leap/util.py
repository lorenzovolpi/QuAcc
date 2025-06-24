import glob
import os
from pathlib import Path

import pandas as pd

from exp.leap.config import PROBLEM, get_method_names, root_dir
from quacc.models.cont_table import LEAP


def load_results(base_dir=None, filter_methods=None) -> pd.DataFrame:
    base_dir = root_dir if base_dir is None else base_dir
    dfs = []
    _methods = get_method_names() if filter_methods is None else filter_methods
    for path in glob.glob(os.path.join(base_dir, PROBLEM, "**", "*.json"), recursive=True):
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


def is_excluded(classifier, dataset, method, acc):
    return False


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")
