import json
import os
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import List
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from quacc import plot
from quacc.evaluation.estimators import CE
from quacc.evaluation.report import CompReport, DatasetReport
from quacc.evaluation.stats import wilcoxon

valid_plot_modes = defaultdict(lambda: CompReport._default_modes)
valid_plot_modes["avg"] = DatasetReport._default_dr_modes


def get_datasets(root: str | Path) -> List[DatasetReport]:
    def load_dataset(dataset):
        dataset = Path(dataset)
        return DatasetReport.unpickle(dataset)

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
    return {str(drp.parent): load_dataset(drp) for drp in dr_paths}


def get_fig(dr: DatasetReport, metric, estimators, view, mode, backend=None):
    _backend = backend or plot.get_backend("plotly")
    print(_backend)
    estimators = CE.name[estimators]
    match (view, mode):
        case ("avg", _):
            return dr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="plotly",
                save_fig=False,
                backend=_backend,
            )
        case (_, _):
            cr = dr.crs[[str(round(c.train_prev[1] * 100)) for c in dr.crs].index(view)]
            return cr.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf="plotly",
                save_fig=False,
                backend=_backend,
            )


def get_table(dr: DatasetReport, metric, estimators, view, mode):
    estimators = CE.name[estimators]
    _prevs = [str(round(cr.train_prev[1] * 100)) for cr in dr.crs]
    match (view, mode):
        case ("avg", "train_table"):
            return dr.data(metric=metric, estimators=estimators).groupby(level=1).mean()
        case ("avg", "test_table"):
            return dr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
        case ("avg", "shift_table"):
            return (
                dr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
        case ("avg", "stats_table"):
            return wilcoxon(dr, metric=metric, estimators=estimators)
        case (_, "train_table"):
            cr = dr.crs[_prevs.index(view)]
            return cr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
        case (_, "shift_table"):
            cr = dr.crs[_prevs.index(view)]
            return (
                cr.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
        case (_, "stats_table"):
            cr = dr.crs[_prevs.index(view)]
            return wilcoxon(cr, metric=metric, estimators=estimators)


def get_DataTable(df):
    _primary = "#0d6efd"
    if df.empty:
        return None

    df = df.reset_index()
    columns = {
        c: dict(
            id=c,
            name=c,
            type="numeric",
            format=Format(precision=6, scheme=Scheme.exponent),
        )
        for c in df.columns
    }
    columns["index"]["format"] = Format(precision=2, scheme=Scheme.fixed)
    columns = list(columns.values())
    data = df.to_dict("records")

    return html.Div(
        [
            dash_table.DataTable(
                data=data,
                columns=columns,
                id="table1",
                style_cell={
                    "padding": "0 12px",
                    "border": "0",
                    "border-bottom": f"1px solid {_primary}",
                },
                style_table={
                    "margin": "6vh 15px",
                    "padding": "15px",
                    "maxWidth": "80vw",
                    "overflowX": "auto",
                    "border": f"0px solid {_primary}",
                    "border-radius": "6px",
                },
            )
        ],
        style={
            "display": "flex",
            "flex-direction": "column",
            # "justify-content": "center",
            "align-items": "center",
            "height": "100vh",
        },
    )


def get_Graph(fig):
    if fig is None:
        return None

    return dcc.Graph(
        id="graph1",
        figure=fig,
        style={
            "margin": 0,
            "height": "100vh",
        },
    )


datasets = get_datasets("output")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.config.suppress_callback_exceptions = True
sidebar_style = {
    "top": 0,
    "left": 0,
    "bottom": 0,
    "padding": "1vw",
    "padding-top": "2vw",
    "margin": "0px",
    "flex": 1,
    "overflow": "scroll",
    "height": "100vh",
}

content_style = {
    "flex": 5,
    "maxWidth": "84vw",
}


def parse_href(href: str):
    parse_result = urlparse(href)
    params = parse_qsl(parse_result.query)
    return dict(params)


def get_sidebar():
    return [
        html.H4("Parameters:", style={"margin-bottom": "1vw"}),
        dbc.Select(
            # options=list(datasets.keys()),
            # value=list(datasets.keys())[0],
            id="dataset",
        ),
        dbc.Select(
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
    ]


app.layout = html.Div(
    [
        dcc.Interval(id="reload", interval=10 * 60 * 1000),
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                html.Div(get_sidebar(), id="app_sidebar", style=sidebar_style),
                html.Div(id="app_content", style=content_style),
            ],
            id="page_layout",
            style={"display": "flex", "flexDirection": "row"},
        ),
    ]
)

server = app.server


def apply_param(href, triggered_id, id, curr):
    match triggered_id:
        case "url":
            params = parse_href(href)
            return params.get(id, None)
        case _:
            return curr


@callback(
    Output("dataset", "value"),
    Output("dataset", "options"),
    Input("url", "href"),
    Input("reload", "n_intervals"),
    State("dataset", "value"),
)
def update_dataset(href, n_intervals, dataset):
    match ctx.triggered_id:
        case "reload":
            new_datasets = get_datasets("output")
            global datasets
            datasets = new_datasets
            req_dataset = dataset
        case "url":
            params = parse_href(href)
            req_dataset = params.get("dataset", None)

    available_datasets = list(datasets.keys())
    new_dataset = (
        req_dataset if req_dataset in available_datasets else available_datasets[0]
    )
    return new_dataset, available_datasets


@callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Input("url", "href"),
    Input("dataset", "value"),
    State("metric", "value"),
)
def update_metrics(href, dataset, curr_metric):
    dr = datasets[dataset]
    old_metric = apply_param(href, ctx.triggered_id, "metric", curr_metric)
    valid_metrics = [m for m in dr.data().columns.unique(0) if not m.endswith("_score")]
    new_metric = old_metric if old_metric in valid_metrics else valid_metrics[0]
    return valid_metrics, new_metric


@callback(
    Output("estimators", "options"),
    Output("estimators", "value"),
    Input("url", "href"),
    Input("dataset", "value"),
    Input("metric", "value"),
    State("estimators", "value"),
)
def update_estimators(href, dataset, metric, curr_estimators):
    dr = datasets[dataset]
    old_estimators = apply_param(href, ctx.triggered_id, "estimators", curr_estimators)
    if isinstance(old_estimators, str):
        try:
            old_estimators = json.loads(old_estimators)
        except JSONDecodeError:
            old_estimators = []
    valid_estimators = dr.data(metric=metric).columns.unique(0).to_numpy()
    new_estimators = valid_estimators[
        np.isin(valid_estimators, old_estimators)
    ].tolist()
    return valid_estimators, new_estimators


@callback(
    Output("view", "options"),
    Output("view", "value"),
    Input("url", "href"),
    Input("dataset", "value"),
    State("view", "value"),
)
def update_view(href, dataset, curr_view):
    dr = datasets[dataset]
    old_view = apply_param(href, ctx.triggered_id, "view", curr_view)
    valid_views = ["avg"] + [str(round(cr.train_prev[1] * 100)) for cr in dr.crs]
    new_view = old_view if old_view in valid_views else valid_views[0]
    return valid_views, new_view


@callback(
    Output("mode", "options"),
    Output("mode", "value"),
    Input("url", "href"),
    Input("view", "value"),
    State("mode", "value"),
)
def update_mode(href, view, curr_mode):
    old_mode = apply_param(href, ctx.triggered_id, "mode", curr_mode)
    valid_modes = valid_plot_modes[view]
    new_mode = old_mode if old_mode in valid_modes else valid_modes[0]
    return valid_modes, new_mode


@callback(
    Output("app_content", "children"),
    Output("url", "search"),
    Input("dataset", "value"),
    Input("metric", "value"),
    Input("estimators", "value"),
    Input("view", "value"),
    Input("mode", "value"),
)
def update_content(dataset, metric, estimators, view, mode):
    search = urlencode(
        dict(
            dataset=dataset,
            metric=metric,
            estimators=json.dumps(estimators),
            view=view,
            mode=mode,
        ),
        quote_via=quote,
    )
    dr = datasets[dataset]
    match mode:
        case m if m.endswith("table"):
            df = get_table(
                dr=dr,
                metric=metric,
                estimators=estimators,
                view=view,
                mode=mode,
            )
            dt = get_DataTable(df)
            app_content = [] if dt is None else [dt]
        case _:
            fig = get_fig(
                dr=dr,
                metric=metric,
                estimators=estimators,
                view=view,
                mode=mode,
            )
            g = get_Graph(fig)
            app_content = [] if g is None else [g]

    return app_content, f"?{search}"


def run():
    app.run(debug=True)


if __name__ == "__main__":
    run()
