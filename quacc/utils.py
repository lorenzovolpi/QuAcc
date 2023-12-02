import functools
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from quacc.environment import env


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


def parallel(func, args, n_jobs, seed=None):
    """
    A wrapper of multiprocessing:

    >>> Parallel(n_jobs=n_jobs)(
    >>>      delayed(func)(args_i) for args_i in args
    >>> )

    that takes the `quapy.environ` variable as input silently.
    Seeds the child processes to ensure reproducibility when n_jobs>1
    """

    return Parallel(n_jobs=n_jobs, verbose=1)(delayed(func)(_args) for _args in args)
