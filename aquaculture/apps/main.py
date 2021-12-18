from aquaculture.common import *

from aquaculture.apps.main_ui import *

class MainApp():
    def __init__(self, **kwargs):
        super().__init__()

        self.global_scope = kwargs.get("global_scope", globals())
        fn_get_param = lambda name, none_value: kwargs.get(name, self.global_scope.get(name, none_value))

        self.assets_dir = fn_get_param("assets_dir", "assets")
        self.app_title  = fn_get_param("app_title", "Hello, Dash")
        self.verbose    = True

        self.__app__ = dash.Dash(__name__,
                                 external_stylesheets=[dbc.themes.BOOTSTRAP],
                                 # suppress_callback_exceptions=True,
                                 title=self.app_title,
                                 # long_callback_manager=long_callback_manager,
                                 assets_folder=self.assets_dir,
                                 meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                                 prevent_initial_callbacks=True,
                                 )

        self.__server__ = self.app.server

        self.logo_path = self.app.get_asset_url("fish.png")
        self.title = "Hello"

        self.alert_dlg = alert_dialog_ui(self)
        self.app.layout = self.page_layout()

        self.registry_events()
        pass # __init__

    @property
    def app(self):
        return self.__app__

    @property
    def server(self):
        return self.__server__

    def page_layout(self, **kwargs):
        layout = dbc.Container(
            [
                dcc.Location(id='url', refresh=False),
                html.Div(id='alert-msg', style={"display": "none"}),
                html.Div(id='alert-modal-pos', style={"display": "none"}, children = [self.alert_dlg]),

                # header
                dbc.Navbar(
                   dbc.Container([
                           dbc.Row(
                               children = [
                                   dbc.Col(
                                       # html.Img(src=self.logo_path, height="40px")
                                   ),
                                   dbc.Col(
                                       dbc.NavbarBrand(self.title, className="ms-2", style = {'color': 'green'})
                                   ),
                               ],
                               align='center', className="g-0 mt-0 p-0"
                           ),
                           dbc.Row(
                               children = [
                                    dbc.Col(
                                        dbc.NavItem(dbc.NavLink("Home", href="/")),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.DropdownMenu(
                                            children = [
                                                dbc.DropdownMenuItem("Data Analysis", href="#", id = "menu-data-analysis"),
                                                dbc.DropdownMenuItem("Data Prediction", href="#", id = "menu-predict"),
                                            ],
                                            nav=True,
                                            in_navbar=True,
                                            label="Menu",
                                        ),
                                        width = "auto",
                                    ),
                                ],
                               align = "right",
                               className="g-0 mt-0 p-0",
                           ),
                   ]),
                   color = 'light',
                ),
                dbc.Row(
                    children=[
                        # main panel
                        dbc.Col(
                            html.Div("workspaces", id = "main_center",
                                     style = {"lineHeight": "80vh", 'textAlign': 'center'},
                            ),
                        ),
                    ],
                    align="center",
                    className="g-0 mt-1 p-0",
                ),
                # footer
                dbc.Navbar(
                    dbc.Container([
                        dbc.Row([
                            html.Div('Status', style = {'fontSize': "x-m"}),
                        ], align = 'center'),
                    ]),
                    color = 'light'
                ),
            ]
        )
        return layout
        pass # page_layout

    def debug(self, title, **kwargs):
        if self.verbose:
            print("=" * 20)
            print("[{title}]")
            print("=" * 20)
            ctx = dash.callback_context
            ctx_msg = json.dumps({
                'states': ctx.states,
                'triggered': ctx.triggered,
                'inputs': ctx.inputs
            }, indent=2)
            print(f'+ Triggers:\n{ctx_msg}\n')
            for k in kwargs:
                print(f'+ {k}:\n{kwargs[k]}\n')
            print("=" * 20)
        pass


    def registry_events(self, **kwargs):

        @self.app.callback(
            Output(f"alert-msg", 'data-menu-predict'),
            Input(f"menu-predict", 'n_clicks'),
        )
        def menu_click(n_clicks):
            return f"Hello"
            pass

        @self.app.callback(
            Output(f"alert-modal-pos", 'children'),
            inputs = dict(inputs = {"menu-predict": Input(f"alert-msg", 'data-menu-predict')}),
        )
        def alert_modal(inputs):
            self.debug("alert_modal", inputs = inputs)
            return alert_dialog_ui(self, content = "Hello", is_open = True, registry_event = False)

        pass # page_events

    pass # MainApp

if __name__ == '__main__':
    MainApp().run_server(debug=True)