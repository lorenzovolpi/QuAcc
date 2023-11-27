from typing import List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quacc.evaluation.estimators import CE
from quacc.evaluation.report import DatasetReport


def ttest_rel(
    dr: DatasetReport, metric: str = None, estimators: List[str] = None
) -> pd.DataFrame:
    _data = dr.data(metric, estimators)

    shapiro_data = np.array(
        [sp_stats.shapiro(_data.loc[:, e]) for e in _data.columns.unique(0)]
    ).T

    _ttest_rel = {}
    for bs in np.intersect1d(CE.name.baselines, _data.columns.unique(0)):
        _ttest_rel[f"ttr_{bs}"] = [
            sp_stats.ttest_rel(_data.loc[:, bs], _data.loc[:, e]).statistic
            if e not in CE.name.baselines
            else np.nan
            for e in _data.columns.unique(0)
        ]
    ttr_data = np.array(list(_ttest_rel.values()))

    dr_index = ["shapiro_W", "shapiro_p"] + list(_ttest_rel.keys())
    dr_columns = _data.columns.unique(0)
    dr_data = (
        np.concatenate([shapiro_data, ttr_data], axis=0)
        if ttr_data.shape[0] > 0
        else shapiro_data
    )
    return pd.DataFrame(dr_data, columns=dr_columns, index=dr_index)
