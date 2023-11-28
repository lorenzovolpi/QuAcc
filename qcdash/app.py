from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html

from quacc.evaluation.report import DatasetReport


def get_fig(data: pd.DataFrame):
    fig = go.Figure()
    xs = data.index.to_numpy()
    for col in data.columns.unique(0):
        _line = go.Scatter(x=xs, y=data.loc[:, col], mode="lines+markers", name=col)
        fig.add_trace(_line)

    fig.update_layout(xaxis_title="test_prevalence", yaxis_title="acc. error")

    return fig


def app_instance():
    dr: DatasetReport = DatasetReport.unpickle(Path("output/debug/imdb/imdb.pickle"))
    data = dr.data(metric="acc").groupby(level=1).mean()

    app = Dash(__name__)

    app.layout = html.Div(
        [
            # html.Div(children="Hello World"),
            # dash_table.DataTable(data=df.to_dict("records")),
            dcc.Graph(figure=get_fig(data), style={"height": "95vh"}),
        ]
    )
    return app


def run():
    app = app_instance()
    app.run(debug=True)


if __name__ == "__main__":
    run()
