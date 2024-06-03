import json
import os
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, dcc, html, page_container

import quacc as qc
from qcdash.navbar import APP_NAME, get_navbar
from quacc.experiments.report import Report
from quacc.plot.plotly import plot_diagonal, plot_shift

NAVBAR = get_navbar()

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA], title=APP_NAME, use_pages=True)

app.layout = html.Div([NAVBAR, page_container])
# dcc.Loading(  # <- Wrap App with Loading Component
#     id="loading_page_content",
#     children=[html.Div([NAVBAR, page_container])],
#     color="primary",  # <- Color of the loading spinner
#     fullscreen=True,  # <- Loading Spinner should take up full screen
# )

server = app.server


def run():
    app.run(debug=True)


if __name__ == "__main__":
    run()
