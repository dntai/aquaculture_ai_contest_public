# ors: PRLAB - Chonnam Nat'l Univ.
# Last modified: Dec 2021
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import hashlib
import json
import os.path
import pickle
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
from dash import dash_table as dbt
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager

import plotly.express as px
from pytorch_tabnet.tab_model import TabNetRegressor

from skimage.io import imread
import diskcache
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn import linear_model as skln

if __name__ == '__main__':
    import utils
else:
    from aquaculture.app_v2 import utils

#
# import multiprocessing as mp
# mp.set_start_method('spawn')

current_dir = os.path.dirname(__file__)
root_dir    = os.path.abspath(f'{current_dir}/../..')
data_dir    = f'{root_dir}/data'
dataset_dir = f'{data_dir}/a2i_data'

assets_dir  = f'{root_dir}/aquaculture/assets'
cache_dir   = f'{root_dir}/aquaculture/assets/cache'
models_dir  = f'{root_dir}/aquaculture/assets/models'

APP_PATH = f'{root_dir}/aquaculture'
# LOGO_PATH = str(BASE_ASSETS_PATH.joinpath("demo.png").resolve())

img_root = f'{dataset_dir}/먹이생물'
csv_root = f'{dataset_dir}/csv'

df = pd.read_csv(f'{assets_dir}/data/final_info.csv')
df.dropna(inplace=True)

cache = diskcache.Cache(f"{cache_dir}")
long_callback_manager = DiskcacheLongCallbackManager(cache)
app_title = "A2I - Aquaculture with AI"
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],  # suppress_callback_exceptions=True,
           title=app_title, long_callback_manager=long_callback_manager,
           assets_folder = assets_dir,
           meta_tags=[{"name": "viewport", "content": "width=device-width"}],
           prevent_initial_callbacks=True)
app._favicon = (f"{assets_dir}/favicon.ico")

username_pwd_container = dbc.Container([
    dbc.Row([dbc.Col(dbc.Label("Username", html_for='username'), width='3', align='center'),
             dbc.Col(dbc.Input(type='text', id='username', placeholder='Enter your username (at least 6 characters)',
                               minlength=6))]),
    dbc.Row([dbc.Col(dbc.Label("Password", html_for='user_password'), width='3', align='center'),
             dbc.Col(dbc.Input(type='password', id='user_password',
                               placeholder='Enter your password (at least 8 characters)', minlength=8))]),
    dbc.Row(dbc.Col(dbc.Label(id='login_info'), align='center'), style={'display': 'none'},
            id='login_info_row'),
], class_name='d-grid gap-3')

login_modal = dbc.Modal(
    [dbc.ModalHeader(dbc.ModalTitle(app_title), close_button=False, class_name="align-self-center"),
     dbc.ModalBody(username_pwd_container),  # Login Form
     dbc.ModalFooter([dbc.Button("Login", id='login_btn', class_name="ms-auto", n_clicks=0),
                      dbc.Button("Register", id='register_btn', class_name="ms-auto", n_clicks=0)],
                     class_name='align-self-center'), ],
    id='login_modal', is_open=False, centered=True, backdrop='static')

# Data table
data_table = dbt.DataTable(id='data-table',
                           columns=[{"name": i, "id": i, "editable": True if i == 'ncells' else False} for i in
                                    df.columns],
                           data=df.to_dict('records'),
                           fixed_rows={'headers': True, 'data': 0},
                           style_cell={'whiteSpace': 'normal', 'textAlign': 'center'},
                           # page_size=100,
                           style_cell_conditional=[
                               {'if': {'column_id': 'Path'}, 'display': 'none'},
                               {'if': {'column_id': 'ID'}, 'width': '25%'},
                               {'if': {'column_id': 'ncells'}, 'width': '8%'},
                           ],
                           style_data_conditional=[
                               {
                                   'if': {'row_index': 'odd'},
                                   'backgroundColor': 'rgb(220, 220, 220)',
                               }
                           ],
                           style_header={
                               'backgroundColor': 'rgb(210, 210, 210)',
                               'color': 'black',
                               'fontWeight': 'bold'
                           },
                           filter_action='native',
                           row_selectable='single',
                           column_selectable=False,
                           virtualization=True,
                           page_action='none',
                           )

# Groupby table
pd.set_option("display.precision", 2)
df_group = df.groupby(['Place', 'Date']).mean().round(2).reset_index()
group_data_table = dbt.DataTable(id='group-data-table',
                                 columns=[{"name": i, "id": i, "editable": True if i == 'ncells' else False} for i in
                                          df_group.columns],
                                 data=df_group.to_dict('records'),
                                 fixed_rows={'headers': True, 'data': 0},
                                 style_cell={'whiteSpace': 'normal', 'textAlign': 'center'},
                                 # page_size=100,
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Path'}, 'display': 'none'},
                                     {'if': {'column_id': 'ID'}, 'width': '25%'},
                                     # {'if': {'column_id': 'ncells'}, 'width': '8%'},
                                     {'if': {'column_id': 'ncells'}, 'display': 'none'},
                                 ],
                                 style_data_conditional=[
                                     {
                                         'if': {'row_index': 'odd'},
                                         'backgroundColor': 'rgb(220, 220, 220)',
                                     }
                                 ],
                                 style_header={
                                     'backgroundColor': 'rgb(210, 210, 210)',
                                     'color': 'black',
                                     'fontWeight': 'bold'
                                 },
                                 filter_action='native',
                                 row_selectable='single',
                                 column_selectable=False,
                                 virtualization=True,
                                 page_action='none',
                                 )
# Image viewer
image_viewer_config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}

image_viewer = [
    dbc.Row([dbc.Col(dbc.Button('Detect', id='detect_cell_btn', n_clicks=0), width='auto'),
             dbc.Col(dbc.Input(type='number', id='num_cell_count'), width='4'),
             dbc.Col(dbc.Button('Update', id='update_num_cell_btn', n_clicks=0), width='auto')], justify='center'),
    dbc.Row(dbc.Col(dcc.Graph(figure={'layout': {'dragmode': 'drawclosedpath',
                                                 'paper_bgcolor' : 'rgba(0,0,0,0)',  # 'LightSteelBlue'
                                                 'newshape': dict(fillcolor="cyan", opacity=0.8,
                                                                  line=dict(color="darkblue"))}},
                              config=image_viewer_config, id='image_viewer_graph'), width=6), justify='center',
            class_name='mt-2')]

# Filter dropdown
place_filter = dcc.Dropdown(id='place-filter', options=[{'label': '고성', 'value': '고성'}, {'label': '일해', 'value': '일해'}],
                            multi=True, placeholder='Place')
date_filter = dcc.Dropdown(id='date-filter', options=[{'label': x, 'value': x} for x in os.listdir(img_root)],
                           multi=True, placeholder='Date')

main_layout = dbc.Row([dbc.Col(place_filter, width='2'), dbc.Col(date_filter, width='3')])

store_data = [dcc.Store(id='user_database'), dcc.Store(id='user_ID'), dcc.Store(id='curr_selection_data'),
              dcc.Store(id='store_place'), dcc.Store(id='store_date')]

navbar = dbc.NavbarSimple(children=[dbc.NavItem(dbc.Button("Training", n_clicks=0, id='train-btn')),
                                    dbc.NavItem(dbc.Button("Prediction", n_clicks=0, id='pred-btn')),
                                    dbc.NavLink('User', id='cur_user')],
                          brand=app_title, color='primary',
                          brand_href="https://prlabjnu.github.io",
                          fluid=True,
                          style={'opacity': '0.8', 'filter': '(opacity=80)'},
                          class_name='transparent',
                          dark=True, )  # class_name='transparent', style={'opacity': '0.5', 'filter': '(opacity=50)'}

# Model training modal
list_of_models = ['LinearRegression', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR',
                  'TabNet']
models_modal = dbc.Modal([dbc.ModalHeader(dbc.ModalTitle('Modelling', id='models_modal_title'), close_button=True),
                          dbc.ModalBody([dbc.Container([
                              dbc.Row([
                                  dbc.Col(html.Span("Model"), width='1', align='center'),
                                  dbc.Col(dcc.Dropdown(id='model_selection',
                                                       options=[{'label': i, 'value': i} for i in list_of_models],
                                                       multi=False, placeholder='Select a model'), width='4'),
                              ]),
                              dbc.Row([dbc.Col(html.Span("Data partitions"), width='auto', align='center')]),
                              # Development set options
                              dbc.Row([
                                  dbc.Col(html.Span('Development set'), width={'offset': 1})
                              ]),
                              dbc.Row([
                                  dbc.Col(dcc.Dropdown(id='dev-place',
                                                       options=[{'label': i, 'value': i} for i in ['Select all']],
                                                       placeholder='Place',
                                                       multi=True), width={'offset': '2', 'size': 3}),
                                  dbc.Col(dcc.Dropdown(id='dev-date',
                                                       options=[{'label': i, 'value': i} for i in ['Select all']],
                                                       placeholder='Date',
                                                       multi=True)),
                              ]),
                              dbc.Row([
                                  dbc.Col(html.Span('Train ratio'), width={'offset': 2, 'size': '2'}, align='center'),
                                  dbc.Col(dcc.Slider(id='dev-train-ratio', min=0.1, max=1.0, step=0.05, value=0.7,
                                                     updatemode='drag',
                                                     tooltip={"placement": "top", "always_visible": True}, ),
                                          align='center', )
                              ]),

                              # Testing set options
                              dbc.Row([
                                  dbc.Col(html.Span('Test set'), width={'offset': 1, 'size': '2'}),
                                  dbc.Col(dbc.RadioItems(id='test-selector', inline=True, value=True,
                                                         options=[{'label': 'The remaining part', 'value': True},
                                                                  {'label': 'Custom', 'value': False}]), width='6'),
                              ]),
                              dbc.Row(
                                  [dbc.Col(dcc.Dropdown(id='test-place',
                                                        options=[{'label': i, 'value': i} for i in ['Select all']],
                                                        placeholder='Place',
                                                        multi=True), width={'offset': '2', 'size': '3'},
                                           style={'display': 'none'}, id='test-place-col'),
                                   dbc.Col(dcc.Dropdown(id='test-date',
                                                        options=[{'label': i, 'value': i} for i in ['Select all']],
                                                        placeholder='Date',
                                                        multi=True), style={'display': 'none'},
                                           id='test-date-col'), ]),

                              dbc.Row([dbc.Col(
                                  dbc.Button('Run', id='run-models-modal', class_name='ms-auto', n_clicks=0),
                                  width='2'), dbc.Col(
                                  dbc.Button('Cancel', id='cancel-run-btn', n_clicks=0, class_name='ms-auto',
                                             disabled=True), width='2')], justify='center', align='center'),

                              dbc.Row(html.Progress(id='run-progress-bar', style={"visibility": "hidden"}),
                                      align='center'),
                              dbc.Row(id='training-results'),

                          ], class_name='d-grid gap-2')]),
                          dbc.ModalFooter([],
                                          class_name='align-self-center')
                          ],
                         id='models_modal', is_open=False, size='lg', backdrop='static', scrollable=True, )

# TODO Prediction modal
prediction_modal = dbc.Modal([dbc.ModalHeader(dbc.ModalTitle('Prediction', class_name='ms-auto', ), ),
                              dbc.ModalBody(dbc.Container([
                                  dbc.Row([dbc.Col(group_data_table, width='6'),
                                           dbc.Col([
                                               dbc.Row([dbc.Col(html.Span('Model'), width='3', align='center'), dbc.Col(
                                                   dcc.Dropdown(placeholder='Select model', id='sel_model_pred',
                                                                multi=False, options=[
                                                           {'label': x.replace('.joblib', ''), 'title': x,
                                                            'value': x} for x in
                                                           os.listdir(f'{models_dir}') if x.endswith('.joblib')]),
                                                   width='8', ), ]),
                                               dbc.Row([dbc.Col(html.Span('MAPE'), width='3', align='center'),
                                                        dbc.Col(html.Span(id='model_pred_descrip'), align='center')],
                                                       class_name='mt-2'),
                                               dbc.Row([dbc.Col(
                                                   html.Span('Temperature', style={'word-break': 'break-all'}),
                                                   width='3', align='center'),
                                                   dbc.Col(dbc.Input(type='number', id='temp_inp'), width='8',
                                                           align='center')], class_name='mt-2'),
                                               dbc.Row([dbc.Col(html.Span('DO'), width='3', align='center'),
                                                        dbc.Col(dbc.Input(type='number', id='do_inp'), width='8',
                                                                align='center')], class_name='mt-2'),
                                               dbc.Row([dbc.Col(html.Span('pH'), width='3', align='center'),
                                                        dbc.Col(dbc.Input(type='number', id='ph_inp'), width='8',
                                                                align='center')], class_name='mt-2'),
                                               dbc.Row([dbc.Col(html.Span('salinity'), width='3', align='center'),
                                                        dbc.Col(dbc.Input(type='number', id='salinity_inp'), width='8',
                                                                align='center')], class_name='mt-2'),
                                               dbc.Row([dbc.Col(html.Span('NTU'), width='3', align='center'),
                                                        dbc.Col(dbc.Input(type='number', id='NTU_inp'), width='8',
                                                                align='center')], class_name='mt-2'),
                                               dbc.Row(
                                                   [dbc.Col(html.Span('Number of cells'), width='3', align='center'),
                                                    dbc.Col(dbc.Input(type='number', id='cell_inp'), width='8',
                                                            align='center')], class_name='mt-2'),
                                               dbc.Row(html.Span(id='get-pred-ret')),
                                           ], width={'offset': 1, 'size': 3}),
                                           dbc.Col([dbc.Row(
                                               dbc.Button('Run', id='get-pred-btn', class_name='rounded-circle',
                                                          style={'width': '100px', 'height': '100px'})),
                                                    dbc.Row(dbc.Button('NaN', id='cancel-pred-btn',
                                                                       class_name='rounded-circle',
                                                                       disabled=True,
                                                                       style={'width': '100px', 'height': '100px'}),
                                                            class_name='mt-2')], width='1'),

                                           ], justify='center', align='center'),
                                  dbc.Row(id='group-analysis_viewer'),
                              ], class_name='d-grid gap-3 overflow-hidden', fluid=True, style={'width': '100%'})),
                              ],
                             id='prediction_modal', is_open=False, fullscreen=True, backdrop='static', scrollable=True)

training_waiting_modal = dbc.Modal(is_open=False, centered=True, id='progress-modal')

# App Layout
app.layout = dbc.Container(
    [navbar, dbc.Row(login_modal), dbc.Row(training_waiting_modal), dbc.Row(store_data),  # main_layout,
     dbc.Row(
         [
             dbc.Col(
                 data_table,
                 width='6'
             ),
             dbc.Col(image_viewer, width='6')
         ]
     ),
     dbc.Row(id='analysis_viewer'),
     models_modal,
     prediction_modal
     ],
    class_name='d-grid gap-2 overflow-hidden', fluid=True,
    style={'background-image': "url('http://sarc.jnu.ac.kr/layout/index/default/image/index_banner1.png')"})


# TODO: update background image

# Callbacks
# Login modal callback, load user database, check user information, save current user_ID
@app.callback([Output('login_modal', 'is_open'), Output('cur_user', 'children'), Output('login_info', 'children'),
               Output('login_info_row', 'style')], [Input('login_btn', 'n_clicks'), Input('register_btn', 'n_clicks')],
              [State('username', 'value'), State('user_password', 'value')])
def login_modal_callback(login_click, register_click, username, user_password):
    change_id = [p['prop_id'] for p in callback_context.triggered][0]
    if len(change_id) == 0:
        raise PreventUpdate
    # Load user database from json file or create json file if it not exists
    if os.path.isfile(f'{assets_dir}/user_database.json'):
        with open(f'{assets_dir}/user_database.json', 'r') as json_file:
            user_db = json.load(json_file)
    else:
        user_db = {}

    login_modal_is_open = True
    user_ID = None
    login_info = ''
    login_info_display = {'display': 'none'}

    if username is None or user_password is None:
        login_info = 'Please enter username and password'
        login_info_display = {}
    else:
        hash_username = hashlib.sha256(username.encode()).hexdigest()
        hash_pwd = hashlib.sha256(user_password.encode()).hexdigest()
        if 'login_btn' in change_id:
            # Login button click, check login information
            if hash_username not in user_db or hash_pwd != user_db[hash_username]:
                # User does not exist or password incorrect
                login_info = 'Incorrect username or password!'
                login_info_display = {}
            else:
                login_modal_is_open = False
                user_ID = username
                login_info = 'Login successful!'
        elif 'register_btn' in change_id:
            # Register button click, check register information
            if hash_username in user_db:
                login_info = '{} is unavailable. Please choose another username.'
                login_info_display = {}
            else:
                user_db[hash_username] = hash_pwd
                with open(f'{assets_dir}/user_database.json', 'w') as json_file:
                    json.dump(user_db, json_file)
                login_info = 'Register successful. Please login.'
                login_info_display = {}
        else:
            raise PreventUpdate

    return login_modal_is_open, user_ID, login_info, login_info_display


# On select row and show image, count cells
@app.callback([Output('image_viewer_graph', 'figure'), Output('curr_selection_data', 'data')],
              [Input('data-table', 'selected_rows')],
              [State('data-table', 'data'), State('image_viewer_graph', 'figure')])
def update_image_viewer(selected_rows, database, image_viewer_graph_fig):
    if selected_rows is not None:
        img_path = '{}/{}'.format(dataset_dir, database[selected_rows[0]]['Path'])
        img = imread(img_path)
        print(img_path, img.shape)
        fig = px.imshow(img, )  # template='plotly_dark'
        fig.update_layout(dragmode='drawclosedpath', newshape = dict(fillcolor="cyan", opacity=0.8,
                         line=dict(color="darkblue")), paper_bgcolor = 'rgba(0,0,0,0)')
        return (fig, database[selected_rows[0]])
    else:
        raise PreventUpdate


# Update prediction input callbacks
@app.callback(
    [Output('temp_inp', 'value'), Output('do_inp', 'value'), Output('ph_inp', 'value'), Output('salinity_inp', 'value'),
     Output('NTU_inp', 'value'), Output('cell_inp', 'value')], Input('group-data-table', 'selected_rows'),
    State('group-data-table', 'data'),
    prevent_initial_call=True)
def update_pred_inputs(selected_rows, database):
    if selected_rows is not None:
        temp = database[selected_rows[0]]['Temperature']
        DO = database[selected_rows[0]]['DO']
        ph = database[selected_rows[0]]['pH']
        salinity = database[selected_rows[0]]['salinity']
        ntu = database[selected_rows[0]]['NTU']
        ncells = int(database[selected_rows[0]]['ncells'])
        return (temp, DO, ph, salinity, ntu, ncells)
    else:
        raise PreventUpdate


# Get predictions get-pred-btn
@app.long_callback(Output('cancel-pred-btn', 'children'), Input('get-pred-btn', 'n_clicks'),
                   [State('sel_model_pred', 'value'), State('temp_inp', 'value'), State('do_inp', 'value'),
                    State('ph_inp', 'value'), State('salinity_inp', 'value'), State('NTU_inp', 'value')],
                   running=([(Output('get-pred-btn', 'children'), [dbc.Spinner(), 'Running'], 'Run'),
                             (Output('get-pred-btn', 'disabled'), True, False),
                             (Output('cancel-pred-btn', 'disabled'), False, True)]),
                   # progress=[Output('pred-progress-bar', 'value'), Output('pred-progress-bar', 'max')],
                   # cancel=[Input('get-pred-btn', 'n_clicks')],
                   interval=5000, prevent_initial_call=True)
def run_prediction(n_clicks, sel_model_pred, temp_inp, do_inp, ph_inp, salinity_inp, ntu_inp):
    # Load model
    if sel_model_pred is not None:
        print(sel_model_pred)
        base_model_info = joblib.load('{}/{}'.format(models_dir, sel_model_pred))
        x = np.array([temp_inp, do_inp, ph_inp, salinity_inp, ntu_inp]).reshape(1, -1)
        x_tf = base_model_info['scaler'].transform(x)
        if base_model_info['model_name'] == 'TabNet':
            # Do tabnet model
            x_preds = []
            for bmodel in base_model_info['base_model']:
                x_preds.append(bmodel.predict(x_tf))

            ret = np.mean(x_preds)
        else:
            # Do ml model
            ret = base_model_info['base_model'].predict(x_tf)

        print('Predict results: ', ret)

        return str(int(ret))
    else:
        return 'Invalid inputs'


# Get selected models info
@app.callback(Output('model_pred_descrip', 'children'), Input('sel_model_pred', 'value'), prevent_initial_call=True)
def get_selected_model_info(sel_model):
    if sel_model is not None:
        cv_results = joblib.load('{}/{}'.format(models_dir, sel_model))['cv_results']
        return '{:.4f} {} {:.4f}'.format(np.mean(cv_results), u"\u00B1", np.std(cv_results))
    else:
        raise PreventUpdate


# Count number of cells in current image callbacks
@app.long_callback([Output('num_cell_count', 'value')], [Input('detect_cell_btn', 'n_clicks')],
                   State('curr_selection_data', 'data'),
                   running=[(Output('detect_cell_btn', 'disabled'), True, False),
                            (Output('detect_cell_btn', 'children'), [dbc.Spinner(size='sm'), 'Running'], 'Detect')],
                   prevent_initial_call=True, interval = 1000)
def detect_cell_count(n_clicks, curr_sel):
    change_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'detect_cell_btn' in change_id and curr_sel is not None:
        img_path = '{}/{}'.format(dataset_dir, curr_sel['Path'])
        blobs, viz = utils.detect_cells(img_path, verbose=1)
        return (len(blobs),)

    else:
        raise PreventUpdate


# Update analysis viewer callbacks
def update_analysis_graphs_func(rows_func):
    print('update graphs')
    if rows_func is not None and len(rows_func) > 0:
        dff = pd.DataFrame(rows_func)

        data_corr_fig = px.imshow(dff.corr(), zmin=-1, zmax=1, color_continuous_scale='ylgnbu')
        data_corr_fig.update_layout(font_color='blue', paper_bgcolor='rgba(0,0,0,0)', font_size=16)


        wid = None  # 520
        hei = None  # 500
        temperature_plot_fig = px.violin(dff, y='Temperature', width=wid, height=hei, box=True)
        temperature_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        do_plot_fig = px.violin(dff, y='DO', width=wid, height=hei, box=True)
        do_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        ph_plot_fig = px.violin(dff, y='pH', width=wid, height=hei, box=True)
        ph_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        salinity_plot_fig = px.violin(dff, y='salinity', width=wid, height=hei, box=True)
        salinity_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        ntu_plot_fig = px.violin(dff, y='NTU', width=wid, height=hei, box=True)
        ntu_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        cells_plot_fig = px.violin(dff, y='ncells', width=wid, height=hei, box=True)
        cells_plot_fig.update_layout(font_color = 'blue', paper_bgcolor = 'rgba(0,0,0,0)', font_size = 16)

        ret_figs = [dbc.Col(dcc.Graph(figure=data_corr_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=temperature_plot_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=do_plot_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=ph_plot_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=salinity_plot_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=ntu_plot_fig), width='3'),
                    dbc.Col(dcc.Graph(figure=cells_plot_fig), width='3')]
        return ret_figs
    else:
        raise PreventUpdate


for db in ['', 'group-']:
    app.callback(Output(f'{db}analysis_viewer', 'children'), Input(f'{db}data-table', 'derived_viewport_data'),
                 prevent_initial_call=True)(update_analysis_graphs_func)

'''
Model modal callbacks
'''


# Toggle model modal callback
@app.callback(
    [Output('models_modal', 'is_open'), Output('models_modal_title', 'children'), Output('store_place', 'data'),
     Output('store_date', 'data')],
    Input('train-btn', 'n_clicks'), Input('pred-btn', 'n_clicks'),  # Input('run-models-modal', 'n_clicks'),
    State('data-table', 'derived_viewport_data'))
def models_modal_toggle(train_click, pred_click, cur_viewport_datatable):
    change_id = [p['prop_id'] for p in callback_context.triggered][0]
    # print('Trigger models_modal toggle ', train_click, pred_click)
    is_open = False
    if 'train-btn' in change_id:
        # Training
        modal_title = 'Training'
        is_open = True
        pass
    elif 'test-btn' in change_id:
        # Prediction
        modal_title = 'Prediction'
        is_open = True
        pass
    else:
        raise PreventUpdate

    if is_open:
        dff = pd.DataFrame(cur_viewport_datatable)
        places = dff['Place'].unique()
        # print('Trigger modal toggle: ', places, len(cur_viewport_datatable))
        dates_by_places = {}
        for plc in places:
            dates_by_places[plc] = dff.query('Place=="{}"'.format(plc))['Date'].unique()

        # print(places, dates_by_places)
    else:
        places = None
        dates_by_places = None

    return is_open, modal_title, places, dates_by_places


# Toggle prediction_modal
@app.callback(Output('prediction_modal', 'is_open'), Input('pred-btn', 'n_clicks'), prevent_initial_call=True)
def prediction_modal_toggle(n_clicks):
    return True


# Run training
@app.long_callback([Output('training-results', 'children'), Output('sel_model_pred', 'options')],
                   [Input('run-models-modal', 'n_clicks')],
                   [State('model_selection', 'value'), State('dev-place', 'value'), State('dev-date', 'value'),
                    State('dev-train-ratio', 'value'), State('test-selector', 'value'), State('test-place', 'value'),
                    State('test-date', 'value'), State('data-table', 'derived_viewport_data'),
                    State('progress-modal', 'is_open')], prevent_initial_call=True,
                   running=([(Output('run-models-modal', 'disabled'), True, False),
                             (Output('cancel-run-btn', 'disabled'), False, True),
                             (Output('training-results', 'style'), {"visibility": "hidden"}, {"visibility": "visible"}),
                             (Output('run-progress-bar', 'style'), {"visibility": "visible"}, {"visibility": "hidden"}),
                             ]),
                   cancel=[Input('cancel-run-btn', 'n_clicks'), ],
                   progress=[Output('run-progress-bar', 'value'), Output('run-progress-bar', 'max')],
                   interval=5000
                   )
def run_training(set_progress, run_modal_clicks, model_name, dev_place, dev_date, dev_train_ratio, test_selector,
                 test_place,
                 test_date, viewport_data, progress_modal_is_open):
    # change_id = [p['prop_id'] for p in callback_context.triggered][0]
    dff = pd.DataFrame(viewport_data)
    # if 'run-models-modal' in change_id:
    if True:
        print('Run models')
        print(model_name, dev_place, dev_date, dev_train_ratio, test_selector, test_place, test_date)

        # Select partitions
        if 'Select all' in dev_place:
            dev_dff = dff
            if test_selector:
                test_dff = None
        else:
            dev_dff = dff.loc[dff['Place'].isin(dev_place)]
            if test_selector:
                test_dff = dff.loc[~dff['Place'].isin(dev_place)]

        if 'Select all' not in dev_date:
            dev_dff = dev_dff.loc[dev_dff['Date'].isin(dev_date)]
            if test_selector:
                test_dff = pd.concat([test_dff, dev_dff.loc[~dev_dff['Date'].isin(dev_date)]])

        # Test partitions
        if not test_selector:
            if 'Select all' in dev_place:
                test_dff = dff
            else:
                test_dff = dff.loc[dff['Place'].isin(test_place)]

            if 'Select all' not in test_date:
                test_dff = test_dff.loc[test_dff['Date'].isin(test_date)]

        # Check consistent of training and testing
        if len(dev_dff) == 0 or test_dff is None or len(test_dff) == 0:
            print('Please select appropriate dev and test splits')
            raise PreventUpdate

        train_ids = set(dev_dff['ID'])
        test_ids = set(test_dff['ID'])
        if len(train_ids.intersection(test_ids)) > 0:
            print('Train and test contain overlap ids, this may lead to incorrect when evaluating model.')

        # set_progress(('3', '10'))
        # TODO

        # Get x train, y train, x test, y test
        dev_dff = dev_dff.groupby(['Place', 'Date']).mean()
        test_dff = test_dff.groupby(['Place', 'Date']).mean()

        train_data = dev_dff[['Temperature', 'DO', 'pH', 'salinity', 'NTU', 'ncells']].values
        test_data = test_dff[['Temperature', 'DO', 'pH', 'salinity', 'NTU', 'ncells']].values

        x_train, y_train = train_data[:, :-1], train_data[:, -1]
        x_test, y_test = test_data[:, :-1], test_data[:, -1]

        y_test = np.floor(y_test)
        y_train = np.floor(y_train)

        # Scaler
        standard_scaler = StandardScaler()
        standard_scaler.fit(x_test)  # Transform xtest ???
        x_train = standard_scaler.transform(x_train)
        x_test = standard_scaler.transform(x_test)

        print(f'train: {x_train.shape}, {y_train.shape}')
        print(f'test:  {x_test.shape},  {y_test.shape}')
        # set_progress(('6', '10'))

        num_folds = 1 + int(dev_train_ratio / (1 - dev_train_ratio))
        save_model_dict = {}
        # TODO: Machine learning models
        if model_name != 'TabNet':
            if model_name == 'SVR':
                base_model = SVR()
            elif model_name == 'KNeighborsRegressor':
                base_model = KNeighborsRegressor()
            elif model_name == 'DecisionTreeRegressor':
                base_model = DecisionTreeRegressor()
            else:
                base_model = getattr(skln, model_name)()
            scoring = 'neg_mean_absolute_percentage_error'

            seed = 42

            cv = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
            cv_results = cross_val_score(base_model, x_test, y_test, cv=cv, scoring=scoring)
            cv_results = np.absolute(cv_results)

            base_model.fit(x_test, y_test)
            save_model_dict = {'base_model': base_model, 'cv_results': cv_results, 'train_ids': train_ids,
                               'test_ids': test_ids, 'num_folds': num_folds, 'scaler': standard_scaler,
                               'model_name': model_name}
            # Save model to file
            if not os.path.isdir(f'{assets_dir}/models'):
                os.makedirs(f'{assets_dir}/models', exist_ok=True)
            joblib.dump(save_model_dict,
                        '{}/{}_{}.joblib'.format(models_dir, model_name, datetime.now().strftime("%m-%d-%Y_%H-%M-%S")))
            print(model_name, cv_results.mean(), cv_results.std())

            ret = dbc.Col(dcc.Markdown('MAPE: {:.4f} {} {:.4f}'.format(cv_results.mean(), u"\u00B1", cv_results.std())))
            # set_progress(('10', '10'))

            options = [
                {'label': x.replace('.joblib', ''), 'title': x,
                 'value': x} for x in
                os.listdir(f'{models_dir}/') if x.endswith('.joblib')]
            return [ret, options]

        else:
            # TODO TabNet
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            cv = KFold(n_splits=num_folds, random_state=42, shuffle=True)
            predictions_array = []
            CV_score_array = []
            CV_mape_array = []
            CV_mse_array = []
            list_regressor = []

            for train_index, test_index in cv.split(x_train):
                x1_train, x1_valid = x_train[train_index], x_train[test_index]
                y1_train, y1_valid = y_train[train_index], y_train[test_index]
                regressor = TabNetRegressor(verbose=0, seed=42)
                regressor.fit(X_train=x1_train, y_train=y1_train,
                              eval_set=[(x1_valid, y1_valid)],
                              patience=300, max_epochs=2000,
                              eval_metric=['rmse'])
                CV_score_array.append(regressor.best_cost)
                y1_pred_valid = regressor.predict(x1_valid)
                CV_mape_array.append(mean_absolute_percentage_error(y1_valid, y1_pred_valid))
                CV_mse_array.append(mean_squared_error(y1_valid, y1_pred_valid))
                list_regressor.append(regressor)

            mape_arr = np.array(CV_mape_array)
            save_model_dict = {'base_model': list_regressor, 'cv_results': mape_arr, 'train_ids': train_ids,
                               'test_ids': test_ids, 'num_folds': num_folds, 'scaler': standard_scaler,
                               'model_name': model_name}

            ret = dbc.Col(dcc.Markdown('MAPE: {:.4f} {} {:.4f}'.format(np.mean(mape_arr), u"\u00B1", np.std(mape_arr))))
            # set_progress(('10', '10'))

            # Save model to file
            if not os.path.isdir(f'{assets_dir}/models'):
                os.makedirs(f'{assets_dir}/models', exist_ok=True)
            joblib.dump(save_model_dict,
                        '{}/{}_{}.joblib'.format(models_dir, model_name, datetime.now().strftime("%m-%d-%Y_%H-%M-%S")))

            options = [
                {'label': x.replace('.joblib', ''), 'title': x,
                 'value': x} for x in
                os.listdir(f'{models_dir}/') if x.endswith('.joblib')]
            return [ret, options]


# @app.callback(Output('progress-modal', 'is_open'), Input('close-progress-modal', 'n_clicks'), prevent_initial_call=True)
# def toggal_progress_modal(n_clicks):
#     return False


# Test part selector options
@app.callback([Output('test-place-col', 'style'), Output('test-date-col', 'style')],
              Input('test-selector', 'value'), )
def test_selector_update(selector_value, ):
    if selector_value:
        return {'display': 'none'}, {'display': 'none'}
    else:
        return {}, {}


# Search options for models modal
## Update place
def update_option_place_func(search_value, store_place):
    cur_db = [{'label': 'Select all', 'value': 'Select all'}, ] + [{'label': x, 'value': x} for x in store_place]
    if not search_value:
        raise PreventUpdate

    search_results = [o for o in cur_db if search_value in o['label']]
    if len(search_results) > 0:
        return search_results
    else:
        return cur_db


for i in ['dev', 'test']:
    app.callback(Output(f"{i}-place", "options"), Input(f"{i}-place", "search_value"), State("store_place", "data"))(
        update_option_place_func)


## Update dates
def update_option_date_func(search_value, store_date, selected_place):
    list_dates = []

    for pla in selected_place:
        if pla != 'Select all':
            list_dates += store_date[pla]
    list_dates = np.unique(list_dates).tolist()

    cur_db = [{'label': 'Select all', 'value': 'Select all'}, ] + [{'label': x, 'value': x} for x in list_dates]
    if not search_value:
        raise PreventUpdate
    else:
        search_results = [o for o in cur_db if search_value in o['label']]

    if len(search_results) > 0:
        return search_results
    else:
        return cur_db


for i in ['dev', 'test']:
    app.callback(Output(f"{i}-date", "options"), Input(f"{i}-date", "search_value"), State('store_date', 'data'),
                 State(f"{i}-place", "value"))(
        update_option_date_func)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True, dev_tools_hot_reload=False, )