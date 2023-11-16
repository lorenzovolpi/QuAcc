import os
from pathlib import Path

import panel as pn
import param

from qcpanel.util import build_plot, explore_datasets, valid_plot_modes
from quacc.evaluation.comp import CE
from quacc.evaluation.report import DatasetReport


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
        self.update_datasets()
        # self._update_on_dataset()
        self.__create_param_pane()
        self.__create_modal_pane()

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
