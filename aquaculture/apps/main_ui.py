from aquaculture.common import *

def alert_dialog_ui(main_app,
                    prefix_uuid = "alert-modal", uuid = "01",
                    title = 'Alert',
                    content = 'This is the content of the modal',
                    model_click_inputs = {},
                    is_open = False,
                    registry_event = True,
                    **kwargs):
    uuid = f'{prefix_uuid}-{uuid}'
    ui = dbc.Modal(
        children = [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(html.Div(children=[content], id=f"{uuid}_content")),
            dbc.ModalFooter(
                dbc.Button("Close", id=f"{uuid}_btnClose", className="ms-auto", n_clicks=0)
            ),
        ],
        id=f"{uuid}",
        is_open=is_open,
    )
    ui.uuid = uuid

    if registry_event == True:
        model_click_inputs[f"{uuid}_btnClose"] = Input(f"{uuid}_btnClose", 'n_clicks')
        ui.model_click_inputs = model_click_inputs
        @main_app.app.callback(
            Output(f'{uuid}', "is_open"),
            inputs = dict(
                inputs = ui.model_click_inputs,
                states = {"is_open": State(f"{uuid}", "is_open")})
        )
        def modal_click(inputs, states):
            return False
        pass # registry_event

    return ui
    pass # alert_dialog_ui