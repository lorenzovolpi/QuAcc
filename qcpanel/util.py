import os
from collections import defaultdict
from pathlib import Path

import panel as pn

from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.evaluation.stats import ttest_rel

_plot_sizing_mode = "stretch_both"
valid_plot_modes = defaultdict(lambda: CompReport._default_modes)
valid_plot_modes["avg"] = DatasetReport._default_dr_modes


def create_plots(
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
    _dpi = 112
    match (plot_view, mode):
        case ("avg", "train_table"):
            _data = (
                dr.data(metric=metric, estimators=estimators).groupby(level=1).mean()
            )
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case ("avg", "test_table"):
            _data = (
                dr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            )
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case ("avg", "shift_table"):
            _data = (
                dr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case ("avg", "stats_table"):
            _data = ttest_rel(dr, metric=metric, estimators=estimators)
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case ("avg", _ as plot_mode):
            _plot = dr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="panel",
                save_fig=False,
            )
            return (
                pn.pane.Matplotlib(
                    _plot,
                    tight=True,
                    format="png",
                    # sizing_mode="scale_height",
                    sizing_mode=_plot_sizing_mode,
                    # sizing_mode="scale_both",
                )
                if _plot is not None
                else None
            )
        case (_, "train_table"):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _data = (
                cr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            )
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case (_, "shift_table"):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _data = (
                cr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
            return pn.pane.DataFrame(_data, align="center") if not _data.empty else None
        case (_, _ as plot_mode):
            cr = dr.crs[_prevs.index(int(plot_view))]
            _plot = cr.get_plots(
                mode=plot_mode,
                metric=metric,
                estimators=estimators,
                conf="panel",
                save_fig=False,
            )
            return (
                pn.pane.Matplotlib(
                    _plot,
                    tight=True,
                    format="png",
                    sizing_mode=_plot_sizing_mode,
                    # sizing_mode="scale_height",
                    # sizing_mode="scale_both",
                )
                if _plot is not None
                else None
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
