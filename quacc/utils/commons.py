import functools
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from time import time
from urllib.request import urlretrieve

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from quacc import logger
from quacc.legacy.environment import env, environ


def combine_dataframes(dfs, df_index=[]) -> pd.DataFrame:
    if len(dfs) < 1:
        raise ValueError
    if len(dfs) == 1:
        return dfs[0]
    df = dfs[0]
    for ndf in dfs[1:]:
        df = df.join(ndf.set_index(df_index), on=df_index)

    return df


def avg_group_report(df: pd.DataFrame) -> pd.DataFrame:
    def _reduce_func(s1, s2):
        return {(n1, n2): v + s2[(n1, n2)] for ((n1, n2), v) in s1.items()}

    lst = df.to_dict(orient="records")[1:-1]
    summed_series = functools.reduce(_reduce_func, lst)
    idx = df.columns.drop([("base", "T"), ("base", "F")])
    avg_report = {
        (n1, n2): (v / len(lst))
        for ((n1, n2), v) in summed_series.items()
        if n1 != "base"
    }
    return pd.DataFrame([avg_report], columns=idx)


def fmt_line_md(s):
    return f"> {s}  \n"


def create_dataser_dir(dir_name, update=False):
    dataset_dir = Path(env.OUT_DIR_NAME) / dir_name
    env.OUT_DIR = dataset_dir
    if update:
        os.makedirs(dataset_dir, exist_ok=True)
    else:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        os.makedirs(dataset_dir)


def get_quacc_home():
    home = Path("~/quacc_home").expanduser()
    os.makedirs(home, exist_ok=True)
    return home


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, downloaded_path: Path):
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=downloaded_path.name,
    ) as t:
        urlretrieve(url, filename=downloaded_path, reporthook=t.update_to)


def parallel(
    func,
    f_args=None,
    parallel: Parallel = None,
    n_jobs=1,
    verbose=0,
    _env: environ | dict = None,
    seed=None,
):
    f_args = f_args or []

    if _env is None:
        _env = {}
    elif isinstance(_env, environ):
        _env = _env.to_dict()

    def wrapper(*args):
        if seed is not None:
            nonlocal _env
            _env = _env | dict(_R_SEED=seed)

        with env.load(_env):
            return func(*args)

    parallel = (
        Parallel(n_jobs=n_jobs, verbose=verbose) if parallel is None else parallel
    )
    return parallel(delayed(wrapper)(*_args) for _args in f_args)


def save_json_file(path, data):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_file(path, object_hook=None):
    if not os.path.exists(path):
        raise ValueError("Ivalid path for json file")
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)
