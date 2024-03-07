from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go

from quacc.evaluation.estimators import CE, _renames
from quacc.plot.base import BasePlot


class PlotCfg:
    def __init__(self, mode, lwidth, font=None, legend=None, template="seaborn"):
        self.mode = mode
        self.lwidth = lwidth
        self.legend = {} if legend is None else legend
        self.font = {} if font is None else font
        self.template = template


web_cfg = PlotCfg("lines+markers", 2)
png_cfg_old = PlotCfg(
    "lines",
    5,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        xanchor="right",
        y=1.02,
        x=1,
        font=dict(size=24),
    ),
    font=dict(size=24),
    # template="ggplot2",
)
png_cfg = PlotCfg(
    "lines",
    5,
    legend=dict(
        font=dict(
            family="DejaVu Sans",
            size=24,
        ),
    ),
    font=dict(size=24),
    # template="ggplot2",
)

_cfg = png_cfg


class PlotlyPlot(BasePlot):
    __themes = defaultdict(
        lambda: {
            "template": _cfg.template,
        }
    )
    __themes = __themes | {
        "dark": {
            "template": "plotly_dark",
        },
    }

    def __init__(self, theme=None):
        self.theme = PlotlyPlot.__themes[theme]
        self.rename = True

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
                __colors = plotly.colors.qualitative.G10

        def __generator(cs):
            while True:
                for c in cs:
                    yield c

        return __generator(__colors)

    def update_layout(self, fig, title, x_label, y_label):
        fig.update_layout(
            # title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme["template"],
            font=_cfg.font,
            legend=_cfg.legend,
        )

    def save_fig(self, fig, base_path, title) -> Path:
        return None

    def rename_plots(
        self,
        columns,
    ):
        if not self.rename:
            return columns

        new_columns = []
        for c in columns:
            nc = c
            for old, new in _renames.items():
                if c.startswith(old):
                    nc = new + c[len(old) :]

            new_columns.append(nc)

        return np.array(new_columns)

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
        named_data = {c: d for c, d in zip(columns, data)}
        r_columns = {c: r for c, r in zip(columns, self.rename_plots(columns))}
        line_colors = self.get_colors(len(columns))
        # for name, delta in zip(columns, data):
        columns = np.array(CE.name.sort(columns))
        for name in columns:
            delta = named_data[name]
            r_name = r_columns[name]
            color = next(line_colors)
            _line = [
                go.Scatter(
                    x=x,
                    y=delta,
                    mode=_cfg.mode,
                    name=r_name,
                    line=dict(color=self.hex_to_rgb(color), width=_cfg.lwidth),
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
        fixed_lim=False,
        legend=True,
    ) -> go.Figure:
        fig = go.Figure()
        x = reference
        line_colors = self.get_colors(len(columns))

        if fixed_lim:
            _lims = np.array([[0.0, 1.0], [0.0, 1.0]])
        else:
            _edges = (
                np.min([np.min(x), np.min(data)]),
                np.max([np.max(x), np.max(data)]),
            )
            _lims = np.array([[_edges[0], _edges[1]], [_edges[0], _edges[1]]])

        named_data = {c: d for c, d in zip(columns, data)}
        r_columns = {c: r for c, r in zip(columns, self.rename_plots(columns))}
        columns = np.array(CE.name.sort(columns))
        for name in columns:
            val = named_data[name]
            r_name = r_columns[name]
            color = next(line_colors)
            slope, interc = np.polyfit(x, val, 1)
            # y_lr = np.array([slope * _x + interc for _x in _lims[0]])
            fig.add_traces(
                [
                    go.Scatter(
                        x=x,
                        y=val,
                        customdata=np.stack((val - x,), axis=-1),
                        mode="markers",
                        name=r_name,
                        marker=dict(color=self.hex_to_rgb(color, t=0.5)),
                        hovertemplate="true acc: %{x:,.4f}<br>estim. acc: %{y:,.4f}<br>acc err.: %{customdata[0]:,.4f}",
                        # showlegend=False,
                    ),
                    # go.Scatter(
                    #     x=[x[-1]],
                    #     y=[val[-1]],
                    #     mode="markers",
                    #     marker=dict(color=self.hex_to_rgb(color), size=8),
                    #     name=r_name,
                    # ),
                    # go.Scatter(
                    #     x=_lims[0],
                    #     y=y_lr,
                    #     mode="lines",
                    #     name=name,
                    #     line=dict(color=self.hex_to_rgb(color), width=3),
                    #     showlegend=False,
                    # ),
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
        fig.update_layout(
            autosize=False,
            width=1300,
            height=1000,
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1.0,
            yaxis_range=[-0.1, 1.1],
        )
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
        named_data = {c: d for c, d in zip(columns, data)}
        r_columns = {c: r for c, r in zip(columns, self.rename_plots(columns))}
        columns = np.array(CE.name.sort(columns))
        for name in columns:
            delta = named_data[name]
            r_name = r_columns[name]
            col_idx = (columns == name).nonzero()[0][0]
            color = next(line_colors)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=delta,
                    customdata=np.stack((counts[col_idx],), axis=-1),
                    mode=_cfg.mode,
                    name=r_name,
                    line=dict(color=self.hex_to_rgb(color), width=_cfg.lwidth),
                    hovertemplate="shift: %{x}<br>error: %{y}"
                    + "<br>count: %{customdata[0]}"
                    if counts is not None
                    else "",
                )
            )

        self.update_layout(fig, title, x_label, y_label)
        return fig

    def plot_fit_scores(
        self,
        train_prevs,
        scores,
        *,
        pos_class=1,
        title="default",
        x_label="prev.",
        y_label="position",
        legend=True,
    ) -> go.Figure:
        fig = go.Figure()
        # x = train_prevs
        x = [str(tuple(bp)) for bp in train_prevs]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scores,
                mode="lines+markers",
                showlegend=False,
            ),
        )

        self.update_layout(fig, title, x_label, y_label)
        return fig
