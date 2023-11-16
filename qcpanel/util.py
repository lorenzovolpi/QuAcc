import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import panel as pn

from quacc.evaluation.comp import CE
from quacc.evaluation.report import DatasetReport

_plot_sizing_mode = "stretch_both"
valid_plot_modes = defaultdict(
    lambda: ["delta", "delta_stdev", "diagonal", "shift", "table", "shift_table"]
)
valid_plot_modes["avg"] = [
    "delta_train",
    "stdev_train",
    "delta_test",
    "stdev_test",
    "shift",
    "train_table",
    "test_table",
    "shift_table",
]


def create_cr_plots(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    prev=None,
):
    _prevs = [round(cr.train_prev[1] * 100) for cr in dr.crs]
    idx = _prevs.index(prev)
    cr = dr.crs[idx]
    estimators = CE.name[estimators]
    if mode is None:
        mode = valid_plot_modes[str(prev)][0]
    _dpi = 112
    if mode == "table":
        return pn.pane.DataFrame(
            cr.data(metric=metric, estimators=estimators).groupby(level=0).mean(),
            align="center",
        )
    elif mode == "shift_table":
        return pn.pane.DataFrame(
            cr.shift_data(metric=metric, estimators=estimators).groupby(level=0).mean(),
            align="center",
        )
    else:
        return pn.pane.Matplotlib(
            cr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="panel",
                return_fig=True,
            ),
            tight=True,
            format="png",
            sizing_mode=_plot_sizing_mode,
            # sizing_mode="scale_height",
            # sizing_mode="scale_both",
        )


def create_avg_plots(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
):
    estimators = CE.name[estimators]
    if mode is None:
        mode = valid_plot_modes["avg"][0]

    if mode == "train_table":
        return pn.pane.DataFrame(
            dr.data(metric=metric, estimators=estimators).groupby(level=1).mean(),
            align="center",
        )
    elif mode == "test_table":
        return pn.pane.DataFrame(
            dr.data(metric=metric, estimators=estimators).groupby(level=0).mean(),
            align="center",
        )
    elif mode == "shift_table":
        return pn.pane.DataFrame(
            dr.shift_data(metric=metric, estimators=estimators).groupby(level=0).mean(),
            align="center",
        )
    return pn.pane.Matplotlib(
        dr.get_plots(
            mode=mode,
            metric=metric,
            estimators=estimators,
            conf="panel",
            return_fig=True,
        ),
        tight=True,
        format="png",
        # sizing_mode="scale_height",
        sizing_mode=_plot_sizing_mode,
        # sizing_mode="scale_both",
    )


def build_plot(
    datasets: Dict[str, DatasetReport],
    dst: str,
    metric: str,
    estimators: List[str],
    view: str,
    mode: str,
):
    _dr = datasets[dst]
    if view == "avg":
        return create_avg_plots(_dr, mode=mode, metric=metric, estimators=estimators)
    else:
        prev = int(view)
        return create_cr_plots(
            _dr, mode=mode, metric=metric, estimators=estimators, prev=prev
        )


def explore_datasets(root: Path | str):
    if isinstance(root, str):
        root = Path(root)

    if root.name == "plot":
        return []

    if not root.exists():
        return []

    drs = []
    for f in os.listdir(root):
        if (root / f).is_dir():
            drs += explore_datasets(root / f)
        elif f == f"{root.name}.pickle":
            drs.append(root / f)
            # drs.append((str(root),))

    return drs
