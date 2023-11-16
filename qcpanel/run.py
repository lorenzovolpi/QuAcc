import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import panel as pn
import param

from quacc import utils
from quacc.evaluation.comp import CE
from quacc.evaluation.report import DatasetReport

pn.config.design = pn.theme.Bootstrap
pn.config.theme = "dark"
pn.config.notifications = True

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
            sizing_mode="scale_height",
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
        sizing_mode="scale_height",
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


def build_modal(datasets, dst, metric):
    return pn.pane.Str(f"{dst}_{metric}")


def build_save_pane(datasets: Dict[str, DatasetReport], dst: str, metric: str):
    return pn.pane.Str(f"{datasets[dst]}_{metric}")


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


class QuaccTestViewer(param.Parameterized):
    dataset = param.Selector()
    metric = param.Selector()
    estimators = param.ListSelector()
    plot_view = param.Selector()
    mode = param.Selector()

    modal_estimators = param.ListSelector()
    modal_plot_view = param.ListSelector()
    modal_mode_prev = param.ListSelector(
        objects=valid_plot_modes[0], default=valid_plot_modes[0]
    )
    modal_mode_avg = param.ListSelector(
        objects=valid_plot_modes["avg"], default=valid_plot_modes["avg"]
    )

    param_pane = param.Parameter()
    plot_pane = param.Parameter()
    modal_pane = param.Parameter()

    def __init__(self, **params):
        super().__init__(**params)

        self.__setup_watchers()
        self.__import_datasets()
        # self._update_on_dataset()
        self.__create_param_pane()
        self.__create_modal_pane()

    def __save_callback(self, event):
        _home = utils.get_quacc_home()
        _save_input_val = self.save_input.value_input
        _config = "default" if len(_save_input_val) == 0 else _save_input_val
        base_path = _home / "output" / self.dataset / _config
        os.makedirs(base_path, exist_ok=True)
        base_plot = base_path / "plot"
        os.makedirs(base_plot, exist_ok=True)

        l_dr = self.datasets_[self.dataset]
        res = l_dr.to_md(
            conf=_config,
            metric=self.metric,
            estimators=CE.name[self.modal_estimators],
            dr_modes=self.modal_mode_avg,
            cr_modes=self.modal_mode_prev,
            plot_path=base_plot,
        )
        with open(base_path / f"{self.metric}.md", "w") as f:
            f.write(res)

        pn.state.notifications.success(f'"{_config}" successfully saved')

    def __create_param_pane(self):
        self.dataset_widget = pn.Param(
            self,
            show_name=False,
            parameters=["dataset"],
            widgets={"dataset": {"widget_type": pn.widgets.Select}},
        )
        self.metric_widget = pn.Param(
            self,
            show_name=False,
            parameters=["metric"],
            widgets={"metric": {"widget_type": pn.widgets.Select}},
        )
        self.estimators_widgets = pn.Param(
            self,
            show_name=False,
            parameters=["estimators"],
            widgets={
                "estimators": {
                    "widget_type": pn.widgets.CheckButtonGroup,
                    "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
        )
        self.plot_view_widget = pn.Param(
            self,
            show_name=False,
            parameters=["plot_view"],
            widgets={
                "plot_view": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
        )
        self.mode_widget = pn.Param(
            self,
            show_name=False,
            parameters=["mode"],
            widgets={
                "mode": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
            align="center",
        )
        self.param_pane = pn.Column(
            self.dataset_widget,
            self.metric_widget,
            pn.Row(
                self.plot_view_widget,
                self.mode_widget,
            ),
            self.estimators_widgets,
        )

    def __create_modal_pane(self):
        self.modal_estimators_widgets = pn.Param(
            self,
            show_name=False,
            parameters=["modal_estimators"],
            widgets={
                "modal_estimators": {
                    "widget_type": pn.widgets.CheckButtonGroup,
                    "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
        )
        self.modal_plot_view_widget = pn.Param(
            self,
            show_name=False,
            parameters=["modal_plot_view"],
            widgets={
                "modal_plot_view": {
                    "widget_type": pn.widgets.CheckButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
        )
        self.modal_mode_prev_widget = pn.Param(
            self,
            show_name=False,
            parameters=["modal_mode_prev"],
            widgets={
                "modal_mode_prev": {
                    "widget_type": pn.widgets.CheckButtonGroup,
                    "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
            align="center",
        )
        self.modal_mode_avg_widget = pn.Param(
            self,
            show_name=False,
            parameters=["modal_mode_avg"],
            widgets={
                "modal_mode_avg": {
                    "widget_type": pn.widgets.CheckButtonGroup,
                    "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    "button_type": "primary",
                    "button_style": "outline",
                }
            },
            align="center",
        )

        self.save_input = pn.widgets.TextInput(
            name="Configuration Name", placeholder="default", sizing_mode="scale_width"
        )
        self.save_button = pn.widgets.Button(
            name="Saverrr",
            sizing_mode="scale_width",
            button_style="solid",
            button_type="success",
        )
        self.save_button.on_click(self.__save_callback)

        _title_styles = {
            "font-size": "14pt",
            "font-weight": "bold",
        }
        self.modal_pane = pn.Column(
            pn.Column(
                pn.pane.Str("Avg. configuration", styles=_title_styles),
                self.modal_mode_avg_widget,
                pn.pane.Str("Train prevs. configuration", styles=_title_styles),
                pn.Row(
                    self.modal_plot_view_widget,
                    self.modal_mode_prev_widget,
                ),
                pn.pane.Str("Estimators configuration", styles=_title_styles),
                self.modal_estimators_widgets,
                self.save_input,
                self.save_button,
                width=450,
                align="center",
                scroll=True,
            ),
            sizing_mode="stretch_both",
        )

    def __import_datasets(self):
        __base_path = "output"
        dataset_paths = sorted(
            explore_datasets(__base_path), key=lambda t: (-len(t.parts), t)
        )
        self.datasets_ = {
            str(dp.parent.relative_to(Path(__base_path))): DatasetReport.unpickle(dp)
            for dp in dataset_paths
        }

        self.available_datasets = list(self.datasets_.keys())
        self.param["dataset"].objects = self.available_datasets
        self.dataset = self.available_datasets[0]

    def __setup_watchers(self):
        self.param.watch(
            self._update_on_dataset,
            ["dataset"],
            queued=True,
            precedence=0,
        )
        self.param.watch(self._update_on_view, ["plot_view"], queued=True, precedence=1)
        self.param.watch(
            self._update_plot,
            ["dataset", "metric", "estimators", "plot_view", "mode"],
            # ["metric", "estimators", "mode"],
            onlychanged=False,
            precedence=2,
        )
        self.param.watch(
            self._update_on_estimators,
            ["estimators"],
            queued=True,
            precedence=3,
        )

    def _update_on_dataset(self, *events):
        l_dr = self.datasets_[self.dataset]
        l_data = l_dr.data()
        l_metrics = l_data.columns.unique(0)
        l_estimators = l_data.columns.unique(1)

        l_valid_estimators = [e for e in l_estimators if e != "ref"]
        l_valid_metrics = [m for m in l_metrics if not m.endswith("_score")]
        l_valid_views = [str(round(cr.train_prev[1] * 100)) for cr in l_dr.crs]

        self.param["metric"].objects = l_valid_metrics
        self.metric = l_valid_metrics[0]

        self.param["estimators"].objects = l_valid_estimators
        self.estimators = l_valid_estimators

        self.param["plot_view"].objects = ["avg"] + l_valid_views
        self.plot_view = "avg"

        self.param["mode"].objects = valid_plot_modes["avg"]
        self.mode = valid_plot_modes["avg"][0]

        self.param["modal_estimators"].objects = l_valid_estimators
        self.modal_estimators = []

        self.param["modal_plot_view"].objects = l_valid_views
        self.modal_plot_view = l_valid_views.copy()

    def _update_on_view(self, *events):
        self.param["mode"].objects = valid_plot_modes[self.plot_view]
        self.mode = valid_plot_modes[self.plot_view][0]

    def _update_on_estimators(self, *events):
        self.modal_estimators = self.estimators.copy()

    def _update_plot(self, *events):
        self.plot_pane = build_plot(
            datasets=self.datasets_,
            dst=self.dataset,
            metric=self.metric,
            estimators=self.estimators,
            view=self.plot_view,
            mode=self.mode,
        )

    def get_plot(self):
        return self.plot_pane

    def get_param_pane(self):
        return self.param_pane


def serve(address="localhost"):
    qtv = QuaccTestViewer()

    def save_callback(event):
        app.open_modal()

    save_button = pn.widgets.Button(
        name="Save",
        sizing_mode="scale_width",
        button_style="solid",
        button_type="success",
    )
    save_button.on_click(save_callback)

    app = pn.template.MaterialTemplate(
        title="quacc tests",
        sidebar=[save_button, qtv.get_param_pane],
        main=[qtv.get_plot],
        modal=[qtv.modal_pane],
    )

    app.servable()
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


if __name__ == "__main__":
    serve()
