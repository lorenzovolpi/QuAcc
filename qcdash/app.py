import json
import os
from collections import defaultdict
from json import JSONDecodeError
from operator import index
from pathlib import Path
from typing import List
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.dash_table.Format import Align, Format, Scheme

from quacc.experiments.report import Report
from quacc.experiments.util import get_acc_name
from quacc.legacy.evaluation.estimators import CE, _renames
from quacc.legacy.evaluation.report import CompReport, DatasetReport
from quacc.legacy.evaluation.stats import wilcoxon
from quacc.plot.plotly import plot_delta, plot_diagonal, plot_shift

valid_plot_modes = defaultdict(lambda: CompReport._default_modes)
valid_plot_modes["avg"] = DatasetReport._default_dr_modes
root_folder = "output"


def _get_prev_str(prev: np.ndarray):
    return str(tuple(np.around(prev, decimals=2)))


def rename_estimators(estimators, rev=False):
    if estimators is None:
        return None

    _rnm = _renames
    if rev:
        _rnm = {v: k for k, v in _renames.items()}

    new_estimators = []
    for c in estimators:
        nc = c
        for old, new in _rnm.items():
            if c.startswith(old):
                nc = new + c[len(old) :]

        new_estimators.append(nc)

    return new_estimators


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


def get_fig(rep: Report, cls_name, acc_name, dataset_name, estimators, view, mode):
    match (view, mode):
        case ("avg", "diagonal"):
            true_accs, estim_accs = rep.diagonal_plot_data(
                dataset_name=dataset_name,
                method_names=estimators,
                acc_name=acc_name,
            )
            return plot_diagonal(
                method_names=estimators,
                true_accs=true_accs,
                estim_accs=estim_accs,
                cls_name=cls_name,
                acc_name=acc_name,
                dataset_name=dataset_name,
            )
        case ("avg", "delta_train"):
            prevs, acc_errs = rep.delta_train_plot_data(
                dataset_name=dataset_name,
                method_names=estimators,
                acc_name=acc_name,
            )
            return plot_delta(
                method_names=estimators,
                prevs=prevs,
                acc_errs=acc_errs,
                cls_name=cls_name,
                acc_mame=acc_name,
                dataset_name=dataset_name,
                prev_name="Test",
            )
        case ("avg", "stdev_train"):
            prevs, acc_errs, stdevs = rep.delta_train_plot_data(
                dataset_name=dataset_name,
                method_names=estimators,
                acc_name=acc_name,
                stdev=True,
            )
            return plot_delta(
                method_names=estimators,
                prevs=prevs,
                acc_errs=acc_errs,
                cls_name=cls_name,
                acc_mame=acc_name,
                dataset_name=dataset_name,
                prev_name="Test",
                stdevs=stdevs,
            )
        case ("avg", "shift"):
            prevs, acc_errs, counts = rep.shift_plot_data(
                dataset_name=dataset_name, method_names=estimators, acc_name=acc_name
            )
            return plot_shift(
                method_names=estimators,
                prevs=prevs,
                acc_errs=acc_errs,
                cls_name=cls_name,
                acc_name=acc_name,
                dataset_name=dataset_name,
                counts=counts,
            )
        case (_, _):
            return None


def get_table(dr: DatasetReport, metric, estimators, view, mode):
    estimators = CE.name[estimators]
    match (view, mode):
        case ("avg", "train_table"):
            # return dr.data(metric=metric, estimators=estimators).groupby(level=1).mean()
            return dr.train_table(metric=metric, estimators=estimators)
        case ("avg", "train_std_table"):
            return dr.train_std_table(metric=metric, estimators=estimators)
        case ("avg", "test_table"):
            # return dr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            return dr.test_table(metric=metric, estimators=estimators)
        case ("avg", "shift_table"):
            # return (
            #     dr.shift_data(metric=metric, estimators=estimators)
            #     .groupby(level=0)
            #     .mean()
            # )
            return dr.shift_table(metric=metric, estimators=estimators)
        case ("avg", "stats_table"):
            return wilcoxon(dr, metric=metric, estimators=estimators)
        case (_, "train_table"):
            cr = dr.crs[[_get_prev_str(c.train_prev) for c in dr.crs].index(view)]
            # return cr.data(metric=metric, estimators=estimators).groupby(level=0).mean()
            return cr.train_table(metric=metric, estimators=estimators)
        case (_, "shift_table"):
            cr = dr.crs[[_get_prev_str(c.train_prev) for c in dr.crs].index(view)]
            # return (
            #     cr.shift_data(metric=metric, estimators=estimators)
            #     .groupby(level=0)
            #     .mean()
            # )
            return cr.shift_table(metric=metric, estimators=estimators)
        case (_, "stats_table"):
            cr = dr.crs[[_get_prev_str(c.train_prev) for c in dr.crs].index(view)]
            return wilcoxon(cr, metric=metric, estimators=estimators)


def get_DataTable(df, mode):
    _primary = "#0d6efd"
    if df.empty:
        return None

    _index_name = dict(
        train_table="test prev.",
        train_std_table="train prev.",
        test_table="train prev.",
        shift_table="shift",
        stats_table="method",
    )
    df = df.reset_index()

    if mode == "train_std_table":
        columns_format = Format()
        df_columns = np.concatenate([["index"], df.columns.unique(1)[1:]])
        data = [
            dict(
                index="(" + ", ".join([f"{v:.2f}" for v in idx]) + ")"
                if isinstance(idx, tuple | list | np.ndarray)
                else str(idx)
            )
            | {
                k: f"{df.loc[i,('avg',k)]:.4f}~{df.loc[i,('std',k)]:.3f}"
                for k in df.columns.unique(1)[1:]
            }
            for i, idx in zip(df.index, df.loc[:, ("index", "")])
        ]
    else:
        columns_format = Format(precision=6, scheme=Scheme.exponent, nully="nan")
        df_columns = df.columns
        data = df.to_dict("records")

    columns = {
        c: dict(
            id=c,
            name=_index_name[mode] if c == "index" else rename_estimators([c])[0],
            type="numeric",
            format=columns_format,
        )
        for c in df_columns
    }
    columns["index"]["format"] = Format()
    columns = list(columns.values())
    for d in data:
        if isinstance(d["index"], tuple | list | np.ndarray):
            d["index"] = "(" + ", ".join([f"{v:.2f}" for v in d["index"]]) + ")"
        elif isinstance(d["index"], float):
            d["index"] = f"{d['index']:.2f}"

    _style_cell = {
        "padding": "0 12px",
        "border": "0",
        "border-bottom": f"1px solid {_primary}",
    }

    _style_cell_conditional = [
        {
            "if": {"column_id": "index"},
            "text_align": "center",
        },
    ]

    _style_data_conditional = []
    if mode != "stats_table":
        _style_data_conditional += [
            {
                "if": {"column_id": "index", "row_index": len(data) - 1},
                "font_weight": "bold",
            },
            {
                "if": {"row_index": len(data) - 1},
                "background_color": "#0d6efd",
                "color": "white",
            },
        ]

    _style_table = {
        "margin": "6vh 15px",
        "padding": "15px",
        "maxWidth": "80vw",
        "overflowX": "auto",
        "border": f"0px solid {_primary}",
        "border-radius": "6px",
    }

    return html.Div(
        [
            dash_table.DataTable(
                data=data,
                columns=columns,
                id="table1",
                style_cell=_style_cell,
                style_cell_conditional=_style_cell_conditional,
                style_data_conditional=_style_data_conditional,
                style_table=_style_table,
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


datasets = get_datasets(root_folder)


def get_dr(root, dataset):
    ds = str(Path(root) / dataset)
    return datasets[ds]


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
        dcc.Store(id="root", storage_type="session", data=root_folder),
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
    Output("root", "data"),
    Input("url", "href"),
    Input("reload", "n_intervals"),
    State("dataset", "value"),
)
def update_dataset(href, n_intervals, dataset):
    if ctx.triggered_id == "reload":
        new_datasets = get_datasets("output")
        global datasets
        datasets = new_datasets

    params = parse_href(href)
    req_dataset = params.get("dataset", None)
    root = params.get("root", "")

    def filter_datasets(root, available_datasets):
        return [
            str(Path(d).relative_to(root))
            for d in available_datasets
            if d.startswith(root)
        ]

    available_datasets = filter_datasets(root, datasets.keys())
    new_dataset = (
        req_dataset if req_dataset in available_datasets else available_datasets[0]
    )
    return new_dataset, available_datasets, root


@callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Input("url", "href"),
    Input("dataset", "value"),
    State("metric", "value"),
    State("root", "data"),
)
def update_metrics(href, dataset, curr_metric, root):
    dr = get_dr(root, dataset)
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
    State("root", "data"),
)
def update_estimators(href, dataset, metric, curr_estimators, root):
    dr = get_dr(root, dataset)
    old_estimators = apply_param(href, ctx.triggered_id, "estimators", curr_estimators)
    if isinstance(old_estimators, str):
        try:
            old_estimators = json.loads(old_estimators)
        except JSONDecodeError:
            old_estimators = []
    old_estimators = rename_estimators(old_estimators, rev=True)
    valid_estimators: np.ndarray = dr.data(metric=metric).columns.unique(0).to_numpy()
    new_estimators = valid_estimators[
        np.isin(valid_estimators, old_estimators)
    ].tolist()
    valid_estimators = CE.name.sort(valid_estimators.tolist())
    return rename_estimators(valid_estimators), rename_estimators(new_estimators)


@callback(
    Output("view", "options"),
    Output("view", "value"),
    Input("url", "href"),
    Input("dataset", "value"),
    State("view", "value"),
    State("root", "data"),
)
def update_view(href, dataset, curr_view, root):
    dr = get_dr(root, dataset)
    old_view = apply_param(href, ctx.triggered_id, "view", curr_view)
    valid_views = ["avg"] + [_get_prev_str(cr.train_prev) for cr in dr.crs]
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
    State("root", "data"),
)
def update_content(dataset, metric, estimators, view, mode, root):
    search = urlencode(
        dict(
            dataset=dataset,
            metric=metric,
            estimators=json.dumps(estimators),
            view=view,
            mode=mode,
            root=root,
        ),
        quote_via=quote,
    )
    dr = get_dr(root, dataset)
    estimators = rename_estimators(estimators, rev=True)
    match mode:
        case m if m.endswith("table"):
            df = get_table(
                dr=dr,
                metric=metric,
                estimators=estimators,
                view=view,
                mode=mode,
            )
            dt = get_DataTable(df, mode)
            app_content = [] if dt is None else [dt]
        case _:
            fig = get_fig(
                dr=dr,
                acc_name=metric,
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
