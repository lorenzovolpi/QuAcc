import os
from collections import defaultdict
from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

from quacc import plot
from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport

backend = plot.get_backend("plotly")

valid_plot_modes = defaultdict(lambda: CompReport._default_modes)
valid_plot_modes["avg"] = DatasetReport._default_dr_modes


def get_datasets(root: str | Path) -> List[DatasetReport]:
    def explore_datasets(root: str | Path) -> List[Path]:
        if isinstance(root, str):
            root = Path(root)

        if root.name == "plot":
            return []

        if not root.exists():
            return []

        dr_paths = []
        for f in os.listdir(root):
            if (root / f).is_dir():
                dr_paths += explore_datasets(root / f)
            elif f == f"{root.name}.pickle":
                dr_paths.append(root / f)

        return dr_paths

    dr_paths = sorted(explore_datasets(root), key=lambda t: (-len(t.parts), t))
    return {str(drp.parent): DatasetReport.unpickle(drp) for drp in dr_paths}


def get_fig(dr: DatasetReport, metric, estimators, view, mode):
    estimators = CE.name[estimators]
    match (view, mode):
        case ("avg", _):
            return dr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="plotly",
                save_fig=False,
                backend=backend,
            )
        case (_, _):
            cr = dr.crs[[str(round(c.train_prev[1] * 100)) for c in dr.crs].index(view)]
            return cr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="plotly",
                save_fig=False,
                backend=backend,
            )


datasets = get_datasets("output")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

sidebar_style = {
    "top": 0,
    "left": 0,
    "bottom": 0,
    "padding": "1vw",
    "padding-top": "2vw",
    "margin": "0px",
    "flex": 2,
}

content_style = {
    # "margin-left": "18vw",
    "flex": 9,
}


sidebar = html.Div(
    children=[
        html.H4("Parameters:", style={"margin-bottom": "1vw"}),
        dbc.Select(
            options=list(datasets.keys()),
            value=list(datasets.keys())[0],
            id="dataset",
        ),
        dbc.Select(
            # clearable=False,
            # searchable=False,
            id="metric",
            style={"margin-top": "1vh"},
        ),
        html.Div(
            [
                dbc.RadioItems(
                    id="view",
                    class_name="btn-group mt-3",
                    input_class_name="btn-check",
                    label_class_name="btn btn-outline-primary",
                    label_checked_class_name="active",
                ),
                dbc.RadioItems(
                    id="mode",
                    class_name="btn-group mt-3",
                    input_class_name="btn-check",
                    label_class_name="btn btn-outline-primary",
                    label_checked_class_name="active",
                ),
            ],
            className="radio-group-v d-flex justify-content-around",
        ),
        html.Div(
            [
                dbc.Checklist(
                    id="estimators",
                    className="btn-group mt-3",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                ),
            ],
            className="radio-group-wide",
        ),
    ],
    style=sidebar_style,
    id="app-sidebar",
)

content = html.Div(
    children=[
        dcc.Graph(
            style={"margin": 0, "height": "100vh"},
            id="graph1",
        ),
    ],
    style=content_style,
)

app.layout = html.Div(
    children=[sidebar, content],
    style={"display": "flex", "flexDirection": "row"},
)


@callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Input("dataset", "value"),
    State("metric", "value"),
)
def update_metrics(dataset, old_metric):
    dr = datasets[dataset]
    valid_metrics = [m for m in dr.data().columns.unique(0) if not m.endswith("_score")]
    new_metric = old_metric if old_metric in valid_metrics else valid_metrics[0]
    return valid_metrics, new_metric


@callback(
    Output("estimators", "options"),
    Output("estimators", "value"),
    Input("dataset", "value"),
    Input("metric", "value"),
    State("estimators", "value"),
)
def update_estimators(dataset, metric, old_estimators):
    dr = datasets[dataset]
    valid_estimators = dr.data(metric=metric).columns.unique(0).to_numpy()
    new_estimators = valid_estimators[
        np.isin(valid_estimators, old_estimators)
    ].tolist()
    return valid_estimators, new_estimators


@callback(
    Output("view", "options"),
    Output("view", "value"),
    Input("dataset", "value"),
    State("view", "value"),
)
def update_view(dataset, old_view):
    dr = datasets[dataset]
    valid_views = ["avg"] + [str(round(cr.train_prev[1] * 100)) for cr in dr.crs]
    new_view = old_view if old_view in valid_views else valid_views[0]
    return valid_views, new_view


@callback(
    Output("mode", "options"),
    Output("mode", "value"),
    Input("view", "value"),
    State("mode", "value"),
)
def update_mode(view, old_mode):
    valid_modes = [m for m in valid_plot_modes[view] if not m.endswith("table")]
    new_mode = old_mode if old_mode in valid_modes else valid_modes[0]
    return valid_modes, new_mode


@callback(
    Output("graph1", "figure"),
    Input("dataset", "value"),
    Input("metric", "value"),
    Input("estimators", "value"),
    Input("view", "value"),
    Input("mode", "value"),
)
def update_graph(dataset, metric, estimators, view, mode):
    dr = datasets[dataset]
    return get_fig(dr=dr, metric=metric, estimators=estimators, view=view, mode=mode)


def run():
    app.run(debug=True)


if __name__ == "__main__":
    run()
