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


def build_widgets(datasets: Dict[str, DatasetReport]):
    available_datasets = list(datasets.keys())
    dataset_widget = pn.widgets.Select(
        name="dataset",
        options=available_datasets,
        align="center",
    )

    _dr = datasets[dataset_widget.value]
    _data = _dr.data()
    _metrics = _data.columns.unique(0)
    _estimators = _data.columns.unique(1)

    valid_metrics = [m for m in _metrics if not m.endswith("_score")]
    metric_widget = pn.widgets.Select(
        name="metric",
        value="acc",
        options=valid_metrics,
        align="center",
    )

    valid_estimators = [e for e in _estimators if e != "ref"]
    estimators_widget = pn.widgets.CheckButtonGroup(
        name="estimators",
        options=valid_estimators,
        value=valid_estimators,
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
        sizing_mode="scale_width",
    )

    valid_views = [str(round(cr.train_prev[1] * 100)) for cr in _dr.crs]
    view_widget = pn.widgets.RadioButtonGroup(
        name="view",
        options=valid_views + ["avg"],
        value="avg",
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
    )

    @pn.depends(dataset_widget.param.value, watch=True)
    def _update_from_dataset(_dataset):
        l_dr = datasets[dataset_widget.value]
        l_data = l_dr.data()
        l_metrics = l_data.columns.unique(0)
        l_estimators = l_data.columns.unique(1)

        l_valid_estimators = [e for e in l_estimators if e != "ref"]
        l_valid_metrics = [m for m in l_metrics if not m.endswith("_score")]
        l_valid_views = [str(round(cr.train_prev[1] * 100)) for cr in l_dr.crs]

        metric_widget.options = l_valid_metrics
        metric_widget.value = l_valid_metrics[0]

        estimators_widget.options = l_valid_estimators
        estimators_widget.value = l_valid_estimators

        view_widget.options = l_valid_views + ["avg"]
        view_widget.value = "avg"

    plot_mode_widget = pn.widgets.RadioButtonGroup(
        name="mode",
        value=valid_plot_modes["avg"][0],
        options=valid_plot_modes["avg"],
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
        sizing_mode="scale_width",
    )

    @pn.depends(view_widget.param.value, watch=True)
    def _update_from_view(_view):
        _modes = valid_plot_modes[_view]
        plot_mode_widget.options = _modes
        plot_mode_widget.value = _modes[0]

    widget_pane = pn.Column(
        dataset_widget,
        metric_widget,
        pn.Row(
            view_widget,
            plot_mode_widget,
        ),
        estimators_widget,
    )

    return (
        widget_pane,
        {
            "dataset": dataset_widget,
            "metric": metric_widget,
            "view": view_widget,
            "plot_mode": plot_mode_widget,
            "estimators": estimators_widget,
        },
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
