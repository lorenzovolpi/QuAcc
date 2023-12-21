import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import panel as pn

from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.evaluation.stats import wilcoxon

_plot_sizing_mode = "stretch_both"
valid_plot_modes = defaultdict(lambda: CompReport._default_modes)
valid_plot_modes["avg"] = DatasetReport._default_dr_modes


def _get_prev_str(prev: np.ndarray):
    return str(tuple(np.around(prev, decimals=2)))


def create_plot(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    plot_view=None,
):
    _prevs = [_get_prev_str(cr.train_prev) for cr in dr.crs]
    estimators = CE.name[estimators]
    if mode is None:
        mode = valid_plot_modes[plot_view][0]
    match (plot_view, mode):
        case ("avg", _ as plot_mode):
            _plot = dr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="panel",
                save_fig=False,
            )
        case (_, _ as plot_mode):
            cr = dr.crs[_prevs.index(plot_view)]
            _plot = cr.get_plots(
                mode=plot_mode,
                metric=metric,
                estimators=estimators,
                conf="panel",
                save_fig=False,
            )
    if _plot is None:
        return None

    return pn.pane.Matplotlib(
        _plot,
        tight=True,
        format="png",
        # sizing_mode="scale_height",
        sizing_mode=_plot_sizing_mode,
        styles=dict(margin="0"),
        # sizing_mode="scale_both",
    )


def create_table(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    plot_view=None,
):
    _prevs = [round(cr.train_prev[1] * 100) for cr in dr.crs]
    estimators = CE.name[estimators]
    if mode is None:
        mode = valid_plot_modes[plot_view][0]
    match (plot_view, mode):
        case ("avg", "train_table"):
            _data = (
                dr.data(metric=metric, estimators=estimators).groupby(level=1).mean()
            )
        case ("avg", "test_table"):
            _data = (
                dr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            )
        case ("avg", "shift_table"):
            _data = (
                dr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
        case ("avg", "stats_table"):
            _data = wilcoxon(dr, metric=metric, estimators=estimators)
        case (_, "train_table"):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _data = (
                cr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            )
        case (_, "shift_table"):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _data = (
                cr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
        case (_, "stats_table"):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _data = wilcoxon(cr, metric=metric, estimators=estimators)

    return (
        pn.Column(
            pn.pane.DataFrame(
                _data,
                align="center",
                float_format=lambda v: f"{v:6e}",
                styles={"font-size-adjust": "0.62"},
            ),
            sizing_mode="stretch_both",
            # scroll=True,
        )
        if not _data.empty
        else None
    )


def create_result(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    plot_view=None,
):
    match mode:
        case m if m.endswith("table"):
            return create_table(dr, mode, metric, estimators, plot_view)
        case _:
            return create_plot(dr, mode, metric, estimators, plot_view)


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
