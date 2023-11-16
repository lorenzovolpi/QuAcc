import argparse
import os
from pathlib import Path

import panel as pn
import param

from quacc.evaluation.comp import CE
from quacc.evaluation.report import DatasetReport

pn.extension(design="bootstrap")


def create_cr_plots(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    prev=None,
):
    idx = [round(cr.train_prev[1] * 100) for cr in dr.crs].index(prev)
    cr = dr.crs[idx]
    estimators = CE.name[estimators]
    _dpi = 112
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
        sizing_mode="scale_height",
        # sizing_mode="scale_both",
    )


def create_avg_plots(
    dr: DatasetReport,
    mode="delta",
    metric="acc",
    estimators=None,
    prev=None,
):
    estimators = CE.name[estimators]
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
        sizing_mode="scale_height",
        # sizing_mode="scale_both",
    )


def build_cr_tab(dr: DatasetReport):
    _data = dr.data()
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

    valid_plot_modes = ["delta", "delta_stdev", "diagonal", "shift"]
    plot_mode_widget = pn.widgets.RadioButtonGroup(
        name="mode",
        value=valid_plot_modes[0],
        options=valid_plot_modes,
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
        sizing_mode="scale_width",
    )

    valid_prevs = [round(cr.train_prev[1] * 100) for cr in dr.crs]
    prevs_widget = pn.widgets.RadioButtonGroup(
        name="train prevalence",
        value=valid_prevs[0],
        options=valid_prevs,
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
    )

    plot_pane = pn.bind(
        create_cr_plots,
        dr=dr,
        mode=plot_mode_widget,
        metric=metric_widget,
        estimators=estimators_widget,
        prev=prevs_widget,
    )

    return pn.Row(
        pn.Spacer(width=20),
        pn.Column(
            metric_widget,
            pn.Row(
                prevs_widget,
                plot_mode_widget,
            ),
            estimators_widget,
            align="center",
        ),
        pn.Spacer(sizing_mode="scale_width"),
        plot_pane,
    )


def build_avg_tab(dr: DatasetReport):
    _data = dr.data()
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

    valid_plot_modes = [
        "delta_train",
        "stdev_train",
        "delta_test",
        "stdev_test",
        "shift",
    ]
    plot_mode_widget = pn.widgets.RadioButtonGroup(
        name="mode",
        value=valid_plot_modes[0],
        options=valid_plot_modes,
        button_style="outline",
        button_type="primary",
        align="center",
        orientation="vertical",
        sizing_mode="scale_width",
    )

    plot_pane = pn.bind(
        create_avg_plots,
        dr=dr,
        mode=plot_mode_widget,
        metric=metric_widget,
        estimators=estimators_widget,
    )

    return pn.Row(
        pn.Spacer(width=20),
        pn.Column(
            metric_widget,
            plot_mode_widget,
            estimators_widget,
            align="center",
        ),
        pn.Spacer(sizing_mode="scale_width"),
        plot_pane,
    )


def build_dataset(dataset_path: Path):
    dr: DatasetReport = DatasetReport.unpickle(dataset_path)

    prevs_tab = ("train prevs.", build_cr_tab(dr))
    avg_tab = ("avg", build_avg_tab(dr))

    app = pn.Tabs(objects=[avg_tab, prevs_tab], dynamic=False)
    app.servable()
    return app


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
            drs.append((root, build_dataset(root / f)))
            # drs.append((str(root),))

    return drs


class PlotSelector(param.Parameterized):
    metric = param.Selector(objects=["acc", "f1"])
    view = param.Selector(objects=["train prevs", "avg"])


def plot_selector_widget():
    return pn.Param(
        PlotSelector.param,
        widgets={
            "metric": pn.widgets.Select,
            "view": pn.widgets.Select,
        },
    )


def serve(address="localhost"):
    # app = build_dataset(Path("output/rcv1_CCAT_9prevs/rcv1_CCAT_9prevs.pickle"))
    __base_path = "output"
    __tabs = sorted(
        explore_datasets(__base_path), key=lambda t: (len(t[0].parts), t[0])
    )
    __tabs = [(str(p.relative_to(Path(__base_path))), d) for (p, d) in __tabs]
    if len(__tabs) > 0:
        app = pn.Tabs(
            objects=__tabs,
            tabs_location="left",
            dynamic=False,
        )
    else:
        app = pn.Column(
            pn.pane.Str("No Dataset Found", styles={"font-size": "24pt"}),
            align="center",
        )

    __port = 33420
    __allowed = [address]
    if address == "localhost":
        __allowed.append("127.0.0.1")

    pn.serve(
        app,
        autoreload=True,
        port=__port,
        show=False,
        address=address,
        websocket_origin=[f"{_a}:{__port}" for _a in __allowed],
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        action="store",
        dest="address",
        default="localhost",
    )
    args = parser.parse_args()
    serve(address=args.address)
