import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from quacc.plot.utils import get_binned_values, get_ref_limits

MODE = "lines"
L_WIDTH = 5
LEGEND = {
    "font": {
        "family": "DejaVu Sans",
        "size": 24,
    }
}
FONT = {"size": 24}
TEMPLATE = "ggplot2"


def _update_layout(fig, x_label, y_label, **kwargs):
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=TEMPLATE,
        font=FONT,
        legend=LEGEND,
        **kwargs,
    )


def _hex_to_rgb(hex: str, t: float | None = None):
    hex = hex.lstrip("#")
    rgb = [int(hex[i : i + 2], 16) for i in [0, 2, 4]]
    if t is not None:
        rgb.append(t)
    return f"{'rgb' if t is None else 'rgba'}{str(tuple(rgb))}"


def _get_colors(num):
    match num:
        case v if v > 10:
            __colors = plotly.colors.qualitative.Light24
        case _:
            __colors = plotly.colors.qualitative.G10

    def __generator(cs):
        while True:
            for c in cs:
                yield c

    return __generator(__colors)


def plot_diagonal(df: pd.DataFrame, cls_name, acc_name, dataset_name):
    fig = px.scatter(df, x="true_accs", y="estim_accs", color="method", opacity=0.5)
    return fig


def plot_shift(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_name,
    *,
    n_bins=20,
    basedir=None,
    file_name=None,
) -> go.Figure:
    # binning on shift values
    df.loc[:, "shifts_bin"] = get_binned_values(df, "shifts", n_bins)

    fig = px.line(
        df.groupby(["shifts_bin", "method"]).mean().reset_index(),
        x="shifts_bin",
        y="acc_err",
        color="method",
    )
    return fig
