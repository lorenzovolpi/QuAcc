from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go

from quacc.utils.commons import get_plots_path

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


def _save_or_return(
    fig: go.Figure, basedir, cls_name, acc_name, dataset_name, plot_type
) -> go.Figure | None:
    if basedir is None:
        return fig

    path = get_plots_path(basedir, cls_name, acc_name, dataset_name, plot_type)
    fig.write_image(path)


def _update_layout(fig, title, x_label, y_label, **kwargs):
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=TEMPLATE,
        font=FONT,
        legend=LEGEND,
        **kwargs,
    )


def _hex_to_rgb(self, hex: str, t: float | None = None):
    hex = hex.lstrip("#")
    rgb = [int(hex[i : i + 2], 16) for i in [0, 2, 4]]
    if t is not None:
        rgb.append(t)
    return f"{'rgb' if t is None else 'rgba'}{str(tuple(rgb))}"


def _get_colors(self, num):
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


def _get_ref_limits(true_accs: np.ndarray, estim_accs: dict[str, np.ndarray]):
    """get lmits of reference line"""

    _edges = (
        np.min([np.min(true_accs), np.min(estim_accs)]),
        np.max([np.max(true_accs), np.max(estim_accs)]),
    )
    _lims = np.array([[_edges[0], _edges[1]], [_edges[0], _edges[1]]])


def plot_diagonal(
    method_names,
    true_accs,
    estim_accs,
    cls_name,
    acc_name,
    dataset_name,
    *,
    basedir=None,
) -> go.Figure:
    fig = go.Figure()
    x = true_accs
    line_colors = _get_colors(len(method_names))
    _lims = _get_ref_limits(true_accs, estim_accs)

    for name, estim in zip(method_names, estim_accs):
        color = next(line_colors)
        slope, interc = np.polyfit(x, estim, 1)
        fig.add_traces(
            [
                go.Scatter(
                    x=x,
                    y=estim,
                    customdata=np.stack((estim - x,), axis=-1),
                    mode="markers",
                    name=name,
                    marker=dict(color=_hex_to_rgb(color, t=0.5)),
                    hovertemplate="true acc: %{x:,.4f}<br>estim. acc: %{y:,.4f}<br>acc err.: %{customdata[0]:,.4f}",
                ),
            ]
        )
    fig.add_trace(
        go.Scatter(
            x=_lims[0],
            y=_lims[1],
            mode="lines",
            name="reference",
            showlegend=False,
            line=dict(color=_hex_to_rgb("#000000"), dash="dash"),
        )
    )

    _update_layout(
        fig,
        x_label=f"True {acc_name}",
        y_label=f"Estimated {acc_name}",
        autosize=False,
        width=1300,
        height=1000,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1.0,
        yaxis_range=[-0.1, 1.1],
    )
    return _save_or_return(fig, basedir, cls_name, acc_name, dataset_name, "diagonal")


def plot_delta(
    method_names: list[str],
    prevs: np.ndarray,
    acc_errs: np.ndarray,
    cls_name,
    acc_mame,
    dataset_name,
    prev_name,
    *,
    stdevs: np.ndarray | None = None,
    basedir=None,
) -> go.Figure:
    fig = go.Figure()
    x = [str(bp) for bp in prevs]
    line_colors = _get_colors(len(method_names))
    if stdevs is None:
        stdevs = [None] * len(method_names)
    for name, delta, stdev in zip(method_names, acc_errs, stdevs):
        color = next(line_colors)
        _line = [
            go.Scatter(
                x=x,
                y=delta,
                mode=MODE,
                name=name,
                line=dict(color=_hex_to_rgb(color), width=L_WIDTH),
                hovertemplate="prev.: %{x}<br>error: %{y:,.4f}",
            )
        ]
        _error = []
        if stdev is not None:
            _error = [
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([delta - stdev, (delta + stdev)[::-1]]),
                    name=name,
                    fill="toself",
                    fillcolor=_hex_to_rgb(color, t=0.2),
                    line=dict(color="rgba(255, 255, 255, 0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            ]
        fig.add_traces(_line + _error)

    _update_layout(
        fig,
        x_label=f"{prev_name} Prevalence",
        y_label=f"Prediction Error for {acc_mame}",
    )
    return _save_or_return(
        fig,
        basedir,
        cls_name,
        acc_mame,
        dataset_name,
        "delta" if stdevs is None else "stdev",
    )


def plot_shift(
    method_names: list[str],
    prevs: np.ndarray,
    acc_errs: np.ndarray,
    cls_name,
    acc_name,
    dataset_name,
    *,
    counts: np.ndarray | None = None,
    basedir=None,
) -> go.Figure:
    fig = go.Figure()
    x = prevs
    line_colors = _get_colors(len(method_names))
    if counts is None:
        counts = [None] * len(method_names)
    for name, delta, count in zip(method_names, acc_errs, counts):
        color = next(line_colors)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                customdata=np.stack((count,), axis=-1),
                mode=MODE,
                name=name,
                line=dict(color=_hex_to_rgb(color), width=L_WIDTH),
                hovertemplate="shift: %{x}<br>error: %{y}"
                + "<br>count: %{customdata[0]}"
                if count is not None
                else "",
            )
        )

    _update_layout(
        fig,
        x_label="Amount of Prior Probability Shift",
        y_label=f"Prediction Error for {acc_name}",
    )
    return _save_or_return(fig, basedir, cls_name, acc_name, dataset_name, "shift")
