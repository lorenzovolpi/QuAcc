from typing import List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport


def shapiro(
    r: DatasetReport | CompReport, metric: str = None, estimators: List[str] = None
) -> pd.DataFrame:
    _data = r.data(metric, estimators)
    shapiro_data = np.array(
        [sp_stats.shapiro(_data.loc[:, e]) for e in _data.columns.unique(0)]
    ).T
    dr_index = ["shapiro_W", "shapiro_p"]
    dr_columns = _data.columns.unique(0)
    return pd.DataFrame(shapiro_data, columns=dr_columns, index=dr_index)


def wilcoxon(
    r: DatasetReport | CompReport, metric: str = None, estimators: List[str] = None
) -> pd.DataFrame:
    _data = r.data(metric, estimators)

    _wilcoxon = {}
    for est in _data.columns.unique(0):
        _wilcoxon[est] = [
            sp_stats.wilcoxon(_data.loc[:, est], _data.loc[:, e]).pvalue
            if e != est
            else 1.0
            for e in _data.columns.unique(0)
        ]
    wilcoxon_data = np.array(list(_wilcoxon.values()))

    dr_index = list(_wilcoxon.keys())
    dr_columns = _data.columns.unique(0)
    return pd.DataFrame(wilcoxon_data, columns=dr_columns, index=dr_index)
