import argparse

import panel as pn
from panel.theme.fast import FastDarkTheme, FastDefaultTheme

from qcpanel.viewer import QuaccTestViewer

# pn.config.design = Fast
# pn.config.theme = "dark"
pn.config.notifications = True


def app_instance():
    param_init = {
        k: v
        for k, v in pn.state.location.query_params.items()
        if k in ["root", "dataset", "metric", "plot_view", "mode", "estimators"]
    }
    qtv = QuaccTestViewer(param_init=param_init)
    pn.state.location.sync(
        qtv,
        {
            "root": "root",
            "dataset": "dataset",
            "metric": "metric",
            "plot_view": "plot_view",
            "mode": "mode",
            "estimators": "estimators",
        },
    )

    def save_callback(event):
        app.open_modal()

    def refresh_callback(event):
        qtv.update_datasets()

    save_button = pn.widgets.Button(
        # name="Save",
        icon="device-floppy",
        icon_size="16px",
        # sizing_mode="scale_width",
        button_style="solid",
        button_type="success",
    )
    save_button.on_click(save_callback)

    refresh_button = pn.widgets.Button(
        icon="refresh",
        icon_size="16px",
        button_style="solid",
    )
    refresh_button.on_click(refresh_callback)

    app = pn.template.FastListTemplate(
        title="quacc tests",
        sidebar=[
            pn.FlexBox(save_button, refresh_button, flex_direction="row-reverse"),
            qtv.get_param_pane,
        ],
        main=[pn.Column(qtv.get_plot, sizing_mode="stretch_both")],
        modal=[qtv.modal_pane],
        # theme=FastDefaultTheme,
        theme_toggle=True,
    )

    app.servable()
    return app


def serve(address="localhost"):
    __port = 33420
    __allowed = [address]
    if address == "localhost":
        __allowed.append("127.0.0.1")

    pn.serve(
        app_instance,
        autoreload=True,
        port=__port,
        show=False,
        address=address,
        websocket_origin=[f"{_a}:{__port}" for _a in __allowed],
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        action="store",
        dest="address",
        default="localhost",
    )
    args = parser.parse_args()
    serve(address=args.address)


if __name__ == "__main__":
    run()
