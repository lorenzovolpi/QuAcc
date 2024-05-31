import json
import os
from glob import glob
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html, register_page
from dash.dash_table.Format import Format, Scheme

import quacc as qc
from qcdash.navbar import APP_NAME
from quacc.experiments.report import Report

register_page(__name__, name=f"{APP_NAME} - table", top_nav=True, path="/table")

root_folder = os.path.join(qc.env["OUT_DIR"], "results")


def get_df(rep: Report):
    df = rep.table_data()
    df = df.pivot_table(values="acc_err", index=["method"], columns=["dataset"], fill_value=np.nan)
    return df


def get_Table(df: pd.DataFrame):
    if df is None:
        return None

    df_idxmin = df.idxmin(axis=0, numeric_only=True)
    df_idxmin_cols = df_idxmin.index.to_list()
    df_idxmin_rows = df_idxmin.to_list()

    columns = {
        c: dict(
            id=c,
            name=c,
            type="numeric",
            format=Format(precision=6, scheme=Scheme.exponent, nully="nan"),
        )
        for c in df.columns
    }
    # columns["dataset"] |= dict(type="text", format=Format())
    columns = list(columns.values())

    _style_table = {
        "margin": "6vh 0px",
        "padding": "15px 0px",
        "maxWidth": "75vw",
        "overflowX": "auto",
        "border-radius": "6px",
    }
    _style_idx_table = {
        "margin": "6vh 0px",
        "padding": "15px 0px",
    }

    _style_cell = {"padding": "0 12px", "text_align": "center", "font_family": "sans"}
    _style_idx_cell = {"padding": "0 12px", "text_align": "right", "font_family": "sans"}
    # _style_cell_cond = [{"if": {"column_id": "dataset"}, "text_align": "right"}]
    _style_data_cond = []
    # for _r, _c in enumerate(df_idxmin_cols):
    for _r, _c in zip(df_idxmin_rows, df_idxmin_cols):
        _style_data_cond.append(
            {
                "if": {"column_id": _c, "row_index": _r},
                "font_weight": "bold",
            }
        )

    idx_table = dash_table.DataTable(
        df.index.to_frame().to_dict("records"), style_table=_style_idx_table, style_cell=_style_idx_cell
    )
    table = dash_table.DataTable(
        df.to_dict("records"),
        columns=columns,
        style_table=_style_table,
        style_cell=_style_cell,
        # style_cell_conditional=_style_cell_cond,
        style_data_conditional=_style_data_cond,
    )
    return html.Div(
        [idx_table, table],
        style={
            "display": "flex",
            "flex-direction": "row",
            "align-items": "top",
            "justify-content": "center",
            "height": "100vh",
        },
    )
    return table


def get_report(config, classifier, acc, dataset, methods):
    return Report.load_results(config, classifier, acc, dataset, methods)


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


def get_sidebar(**kwargs):
    config = kwargs.get("config", None)
    classifier = kwargs.get("classifier", None)
    acc = kwargs.get("acc", None)
    dataset = kwargs.get("dataset", None)
    methods = kwargs.get("methods", [])
    return [
        html.H4("Parameters:", style={"margin-bottom": "1vw"}),
        dbc.Row(
            [
                dbc.Label("config", width=3),
                dbc.Col(dbc.Select(id="tbl_config", value=config), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("classifier", width=3),
                dbc.Col(dbc.Select(id="tbl_classifier", value=classifier), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("accuracy", width=3),
                dbc.Col(dbc.Select(id="tbl_acc", value=acc), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("dataset", width=3),
                dbc.Col(dbc.Select(id="tbl_dataset", value=dataset), width=9),
            ],
            className="mb-3",
        ),
        dbc.Col([dbc.Label("Methods"), dcc.Dropdown(id="tbl_methods", value=methods, multi=True)]),
    ]


def layout(**kwargs):
    layout = html.Div(
        [
            # dcc.Interval(id="reload", interval=10 * 60 * 1000),
            dcc.Location(id="tbl_url", refresh=False),
            dcc.Store(id="tbl_root", storage_type="session", data=root_folder),
            dcc.Store(id="tbl_tree", storage_type="session", data={}),
            html.Div(
                [
                    html.Div(get_sidebar(**kwargs), id="tbl_app_sidebar", style=sidebar_style),
                    html.Div(id="tbl_app_content", style=content_style),
                ],
                id="tbl_page_layout",
                style={"display": "flex", "flexDirection": "row"},
            ),
        ]
    )

    return layout


def build_tree():
    tree = {}
    for path, dirs, files in os.walk(root_folder):
        if len(files) == 0:
            tree[path] = dirs
        else:
            tree[path] = [Path(f).stem for f in files]

    return tree


def apply_param(href, triggered_id, id, curr):
    match triggered_id:
        case "tbl_url":
            params = parse_href(href)
            return params.get(id, None)
        case _:
            return curr


def get_valid_fields(tree, req, *args):
    if req == "config":
        assert len(args) == 0
    elif req == "classifier":
        assert len(args) == 1 and None not in args
    elif req == "acc":
        assert len(args) == 2 and None not in args
    elif req == "dataset":
        assert len(args) == 3 and None not in args
    elif req == "methods":
        assert len(args) == 4 and None not in args

    idx = os.path.join(root_folder, *args)
    if req == "methods":
        res = []
        for path in glob(idx):
            res += tree.get(path, [])
        res = np.unique(res).tolist()
    else:
        res = tree.get(idx, [])
    return res


@callback(
    Output("tbl_config", "value"),
    Output("tbl_config", "options"),
    Output("tbl_tree", "data"),
    Input("tbl_url", "href"),
)
def tbl_update_config(href):
    tree = build_tree()
    req_config = parse_href(href).get("config", None)
    valid_configs = get_valid_fields(tree, "config")
    assert len(valid_configs) > 0, "no valid configs"
    new_config = req_config if req_config in valid_configs else valid_configs[0]
    return new_config, valid_configs, tree


@callback(
    Output("tbl_classifier", "value"),
    Output("tbl_classifier", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    State("tbl_tree", "data"),
    State("tbl_classifier", "value"),
)
def tbl_update_classifier(href, config, tree, classifier):
    req_classifier = apply_param(href, ctx.triggered_id, "classifier", classifier)
    valid_classifiers = get_valid_fields(tree, "classifier", config)
    assert len(valid_classifiers) > 0, "no valid classifiers"
    new_classifier = req_classifier if req_classifier in valid_classifiers else valid_classifiers[0]
    return new_classifier, valid_classifiers


@callback(
    Output("tbl_acc", "value"),
    Output("tbl_acc", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    State("tbl_tree", "data"),
    State("tbl_acc", "value"),
)
def tbl_update_acc(href, config, classifier, tree, acc):
    req_acc = apply_param(href, ctx.triggered_id, "acc", acc)
    valid_accs = get_valid_fields(tree, "acc", config, classifier)
    assert len(valid_accs) > 0, "no valid accs"
    new_acc = req_acc if req_acc in valid_accs else valid_accs[0]
    return new_acc, valid_accs


@callback(
    Output("tbl_dataset", "value"),
    Output("tbl_dataset", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    State("tbl_tree", "data"),
    State("tbl_dataset", "value"),
)
def tbl_update_dataset(href, config, classifier, acc, tree, dataset):
    req_dataset = apply_param(href, ctx.triggered_id, "dataset", dataset)
    valid_datasets = ["*"] + get_valid_fields(tree, "dataset", config, classifier, acc)
    assert len(valid_datasets) > 0, "no valid datasets"
    new_dataset = req_dataset if req_dataset in valid_datasets else valid_datasets[0]
    return new_dataset, valid_datasets


@callback(
    Output("tbl_methods", "value"),
    Output("tbl_methods", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    Input("tbl_dataset", "value"),
    State("tbl_tree", "data"),
    State("tbl_methods", "value"),
)
def tbl_update_methods(href, config, classifier, acc, dataset, tree, methods):
    req_methods = apply_param(href, ctx.triggered_id, "methods", methods)
    if isinstance(req_methods, str):
        try:
            req_methods = json.loads(req_methods)
        except JSONDecodeError:
            req_methods = []
    valid_methods = get_valid_fields(tree, "methods", config, classifier, acc, dataset)

    if req_methods is None or len(req_methods) == 0:
        return [], valid_methods

    new_methods = np.unique(np.array(req_methods)[np.in1d(req_methods, valid_methods)]).tolist()
    return new_methods, valid_methods


@callback(
    Output("tbl_app_content", "children"),
    Output("tbl_url", "search"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    Input("tbl_dataset", "value"),
    Input("tbl_methods", "value"),
    State("tbl_tree", "data"),
)
def tbl_update_content(config, classifier, acc, dataset, methods, tree):
    search = urlencode(
        dict(
            config=config,
            classifier=classifier,
            acc=acc,
            dataset=dataset,
            methods=json.dumps(methods),
        ),
        quote_via=quote,
    )
    search_str = f"?{search}"

    if methods is None or len(methods) == 0:
        return [], search_str

    report = get_report(config, classifier, acc, dataset, methods)

    if report is None:
        return [], search_str

    df = get_df(report)
    table = get_Table(df)
    app_content = [] if table is None else [table]

    return app_content, search_str
