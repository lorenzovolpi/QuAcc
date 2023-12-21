import os
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import param

from qcpanel.util import (
    _get_prev_str,
    create_result,
    explore_datasets,
    valid_plot_modes,
)
from quacc.evaluation.estimators import CE
from quacc.evaluation.report import DatasetReport


class QuaccTestViewer(param.Parameterized):
    __base_path = "output"

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

    root = param.String()

    def __init__(self, param_init=None, **params):
        super().__init__(**params)

        self.param_init = param_init
        self.__setup_watchers()
        self.update_datasets()
        # self._update_on_dataset()
        self.__create_param_pane()
        self.__create_modal_pane()

    def __get_param_init(self, val):
        __b = val in self.param_init
        if __b:
            setattr(self, val, self.param_init[val])
            del self.param_init[val]

        return __b

    def __save_callback(self, event):
        _home = Path("output")
        _save_input_val = self.save_input.value_input
        _config = "default" if len(_save_input_val) == 0 else _save_input_val
        base_path = _home / self.dataset / _config
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
            cr_prevs=self.modal_plot_view,
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
                    "widget_type": pn.widgets.MultiChoice,
                    # "orientation": "vertical",
                    "sizing_mode": "scale_width",
                    # "button_type": "primary",
                    # "button_style": "outline",
                    "solid": True,
                    "search_option_limit": 1000,
                    "option_limit": 1000,
                    "max_items": 1000,
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
            name="Save",
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
                pn.Spacer(height=20),
                width=450,
                align="center",
                scroll=True,
            ),
            sizing_mode="stretch_both",
        )

    def update_datasets(self):
        if not self.__get_param_init("root"):
            self.root = self.__base_path

        dataset_paths = sorted(
            explore_datasets(self.root), key=lambda t: (-len(t.parts), t)
        )
        self.datasets_ = {
            str(dp.parent.relative_to(Path(self.root))): DatasetReport.unpickle(dp)
            for dp in dataset_paths
        }

        self.available_datasets = list(self.datasets_.keys())
        _old_dataset = self.dataset
        self.param["dataset"].objects = self.available_datasets
        if not self.__get_param_init("dataset"):
            self.dataset = (
                _old_dataset
                if _old_dataset in self.available_datasets
                else self.available_datasets[0]
            )

    def __setup_watchers(self):
        self.param.watch(
            self._update_on_dataset,
            ["dataset"],
            queued=True,
            precedence=0,
        )
        self.param.watch(self._update_on_view, ["plot_view"], queued=True, precedence=1)
        self.param.watch(self._update_on_metric, ["metric"], queued=True, precedence=2)
        self.param.watch(
            self._update_plot,
            ["dataset", "metric", "estimators", "plot_view", "mode"],
            # ["metric", "estimators", "mode"],
            onlychanged=False,
            precedence=3,
        )
        self.param.watch(
            self._update_on_estimators,
            ["estimators"],
            queued=True,
            precedence=4,
        )

    def _update_on_dataset(self, *events):
        l_dr = self.datasets_[self.dataset]
        l_data = l_dr.data()

        l_metrics = l_data.columns.unique(0)
        l_valid_metrics = [m for m in l_metrics if not m.endswith("_score")]
        _old_metric = self.metric
        self.param["metric"].objects = l_valid_metrics
        if not self.__get_param_init("metric"):
            self.metric = (
                _old_metric if _old_metric in l_valid_metrics else l_valid_metrics[0]
            )

        _old_estimators = self.estimators
        l_valid_estimators = l_dr.data(metric=self.metric).columns.unique(0).to_numpy()
        _new_estimators = l_valid_estimators[
            np.isin(l_valid_estimators, _old_estimators)
        ].tolist()
        self.param["estimators"].objects = l_valid_estimators
        if not self.__get_param_init("estimators"):
            self.estimators = _new_estimators

        l_valid_views = [_get_prev_str(cr.train_prev) for cr in l_dr.crs]
        l_valid_views = ["avg"] + l_valid_views
        _old_view = self.plot_view
        self.param["plot_view"].objects = l_valid_views
        if not self.__get_param_init("plot_view"):
            self.plot_view = _old_view if _old_view in l_valid_views else "avg"

        self.param["mode"].objects = valid_plot_modes[self.plot_view]
        if not self.__get_param_init("mode"):
            _old_mode = self.mode
            if _old_mode in valid_plot_modes[self.plot_view]:
                self.mode = _old_mode
            else:
                self.mode = valid_plot_modes[self.plot_view][0]

        self.param["modal_estimators"].objects = l_valid_estimators
        self.modal_estimators = []

        self.param["modal_plot_view"].objects = l_valid_views
        self.modal_plot_view = l_valid_views.copy()

    def _update_on_view(self, *events):
        _old_mode = self.mode
        self.param["mode"].objects = valid_plot_modes[self.plot_view]
        if _old_mode in valid_plot_modes[self.plot_view]:
            self.mode = _old_mode
        else:
            self.mode = valid_plot_modes[self.plot_view][0]

    def _update_on_metric(self, *events):
        _old_estimators = self.estimators

        l_dr = self.datasets_[self.dataset]
        l_data: pd.DataFrame = l_dr.data(metric=self.metric)
        l_valid_estimators: np.ndarray = l_data.columns.unique(0).to_numpy()
        _new_estimators = l_valid_estimators[
            np.isin(l_valid_estimators, _old_estimators)
        ].tolist()
        self.param["estimators"].objects = l_valid_estimators
        self.estimators = _new_estimators

    def _update_on_estimators(self, *events):
        self.modal_estimators = self.estimators.copy()

    def _update_plot(self, *events):
        __svg = pn.pane.SVG(
            """<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-chart-area-filled" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                    <path d="M20 18a1 1 0 0 1 .117 1.993l-.117 .007h-16a1 1 0 0 1 -.117 -1.993l.117 -.007h16z" stroke-width="0" fill="currentColor" />
                    <path d="M15.22 5.375a1 1 0 0 1 1.393 -.165l.094 .083l4 4a1 1 0 0 1 .284 .576l.009 .131v5a1 1 0 0 1 -.883 .993l-.117 .007h-16.022l-.11 -.009l-.11 -.02l-.107 -.034l-.105 -.046l-.1 -.059l-.094 -.07l-.06 -.055l-.072 -.082l-.064 -.089l-.054 -.096l-.016 -.035l-.04 -.103l-.027 -.106l-.015 -.108l-.004 -.11l.009 -.11l.019 -.105c.01 -.04 .022 -.077 .035 -.112l.046 -.105l.059 -.1l4 -6a1 1 0 0 1 1.165 -.39l.114 .05l3.277 1.638l3.495 -4.369z" stroke-width="0" fill="currentColor" />
                </svg>""",
            sizing_mode="stretch_both",
        )
        if len(self.estimators) == 0:
            self.plot_pane = __svg
        else:
            _dr = self.datasets_[self.dataset]
            __plot = create_result(
                _dr,
                mode=self.mode,
                metric=self.metric,
                estimators=self.estimators,
                plot_view=self.plot_view,
            )
            self.plot_pane = __svg if __plot is None else __plot

    def get_plot(self):
        return self.plot_pane

    def get_param_pane(self):
        return self.param_pane
