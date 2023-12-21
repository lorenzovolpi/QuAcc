from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go

from quacc.plot.base import BasePlot


class PlotlyPlot(BasePlot):
    __themes = defaultdict(
        lambda: {
            "template": "seaborn",
        }
    )
    __themes = __themes | {
        "dark": {
            "template": "plotly_dark",
        },
    }

    def __init__(self, theme=None):
        self.theme = PlotlyPlot.__themes[theme]

    def hex_to_rgb(self, hex: str, t: float | None = None):
        hex = hex.lstrip("#")
        rgb = [int(hex[i : i + 2], 16) for i in [0, 2, 4]]
        if t is not None:
            rgb.append(t)
        return f"{'rgb' if t is None else 'rgba'}{str(tuple(rgb))}"

    def get_colors(self, num):
        match num:
            case v if v > 10:
                __colors = plotly.colors.qualitative.Light24
            case _:
                __colors = plotly.colors.qualitative.Plotly

        def __generator(cs):
            while True:
                for c in cs:
                    yield c

        return __generator(__colors)

    def update_layout(self, fig, title, x_label, y_label):
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme["template"],
        )

    def save_fig(self, fig, base_path, title) -> Path:
        return None

    def plot_delta(
        self,
        base_prevs,
        columns,
        data,
        *,
        stdevs=None,
        pos_class=1,
        title="default",
        x_label="prevs.",
        y_label="error",
        legend=True,
    ) -> go.Figure:
        fig = go.Figure()
        if isinstance(base_prevs[0], float):
            base_prevs = np.around([(1 - bp, bp) for bp in base_prevs], decimals=4)
        x = [str(tuple(bp)) for bp in base_prevs]
        line_colors = self.get_colors(len(columns))
        for name, delta in zip(columns, data):
            color = next(line_colors)
            _line = [
                go.Scatter(
                    x=x,
                    y=delta,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=self.hex_to_rgb(color)),
                    hovertemplate="prev.: %{x}<br>error: %{y:,.4f}",
                )
            ]
            _error = []
            if stdevs is not None:
                _col_idx = np.where(columns == name)[0]
                stdev = stdevs[_col_idx].flatten()
                _error = [
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([delta - stdev, (delta + stdev)[::-1]]),
                        name=int(_col_idx[0]),
                        fill="toself",
                        fillcolor=self.hex_to_rgb(color, t=0.2),
                        line=dict(color="rgba(255, 255, 255, 0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                ]
            fig.add_traces(_line + _error)

        self.update_layout(fig, title, x_label, y_label)
        return fig

    def plot_diagonal(
        self,
        reference,
        columns,
        data,
        *,
        pos_class=1,
        title="default",
        x_label="true",
        y_label="estim.",
        legend=True,
    ) -> go.Figure:
        fig = go.Figure()
        x = reference
        line_colors = self.get_colors(len(columns))

        _edges = (np.min([np.min(x), np.min(data)]), np.max([np.max(x), np.max(data)]))
        _lims = np.array([[_edges[0], _edges[1]], [_edges[0], _edges[1]]])

        for name, val in zip(columns, data):
            color = next(line_colors)
            slope, interc = np.polyfit(x, val, 1)
            y_lr = np.array([slope * _x + interc for _x in _lims[0]])
            fig.add_traces(
                [
                    go.Scatter(
                        x=x,
                        y=val,
                        customdata=np.stack((val - x,), axis=-1),
                        mode="markers",
                        name=name,
                        line=dict(color=self.hex_to_rgb(color, t=0.5)),
                        hovertemplate="true acc: %{x:,.4f}<br>estim. acc: %{y:,.4f}<br>acc err.: %{customdata[0]:,.4f}",
                    ),
                    go.Scatter(
                        x=_lims[0],
                        y=y_lr,
                        mode="lines",
                        name=name,
                        line=dict(color=self.hex_to_rgb(color), width=3),
                        showlegend=False,
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
                line=dict(color=self.hex_to_rgb("#000000"), dash="dash"),
            )
        )

        self.update_layout(fig, title, x_label, y_label)
        fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1.0)
        return fig

    def plot_shift(
        self,
        shift_prevs,
        columns,
        data,
        *,
        counts=None,
        pos_class=1,
        title="default",
        x_label="true",
        y_label="estim.",
        legend=True,
    ) -> go.Figure:
        fig = go.Figure()
        # x = shift_prevs[:, pos_class]
        x = shift_prevs
        line_colors = self.get_colors(len(columns))
        for name, delta in zip(columns, data):
            col_idx = (columns == name).nonzero()[0][0]
            color = next(line_colors)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=delta,
                    customdata=np.stack((counts[col_idx],), axis=-1),
                    mode="lines+markers",
                    name=name,
                    line=dict(color=self.hex_to_rgb(color)),
                    hovertemplate="shift: %{x}<br>error: %{y}"
                    + "<br>count: %{customdata[0]}"
                    if counts is not None
                    else "",
                )
            )

        self.update_layout(fig, title, x_label, y_label)
        return fig
