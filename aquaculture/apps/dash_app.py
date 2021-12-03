from aquaculture.common import *
from aquaculture.apps.utils.qt_utils import *

class MainDash(QDash):
    def __init__(self, parent = None, global_cfg = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.title = "Smart Aquaculture Management"
        self.logo_path  = self.app.get_asset_url("fish.png")

        self.setupLayout()
        self.setupCallbacks()

        pass # __init__

    def setupLayout(self):
        self.app.layout = dbc.Container(
            [
               dcc.Location(id='url', refresh=False),

               # header
               dbc.Navbar(
                   dbc.Container([
                           dbc.Row(
                               children = [
                                   dbc.Col(
                                       html.Img(src=self.logo_path, height="40px")
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
                                     style = {"lineHeight": "400px", 'textAlign': 'center'},
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
        pass # setupLayout

    def setupCallbacks(self):
        pass

    def render_page(self):
        pass

    def dataPreprocessingView(self):
        data_root_input = dbc.Container(
            [
                dbc.Label("Data Root", html_for="data-root"),
                dbc.Input(id="data-root", placeholder="Enter Data Root", value = '{root_dir}/data/AI_competition'),
            ],
        )

        return data_root_input
        pass

    def setupSidePanelView(self):
        layout = dbc.Container(
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        "Abc",
                    ],
                    title="Item 1",
                ),
                dbc.AccordionItem(
                    [
                        html.P("This is the content of the second section"),
                        dbc.Button("Don't click me!", color="danger"),
                    ],
                    title="Item 2",
                ),
                dbc.AccordionItem(
                    "This is the content of the third section",
                    title="Item 3",
                ),
            ]))
        return layout
        pass # setupSidePanelView

    pass # MainDash