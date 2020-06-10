import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import traceback

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils

from ml.pca import perform_pca
from ml.pca import dot_product

layout = html.Div([
    common.navbar("Principle Component Analysis (PCA)"),
    html.Div([], style = {'padding': '30px'}),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'pca-load-cleaned-files', className="mr-2", style={'display': 'inline-block'}),
        dbc.Button("Clear", color="secondary", id = 'pca-clear-db', className="mr-2", style={'display': 'inline-block'})
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'pca-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "pca-clear-db-do-nothing"),
    html.Div([],id = "pca-selected-scatter-plot")
])

@app.callback(
    Output("pca-selected-cleaned-file", "options"),
    [Input('pca-load-cleaned-files', 'n_clicks')]
)
def selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("pca-clear-db-do-nothing", "options"),
    [Input('pca-clear-db', 'n_clicks')]
)
def selected_file(n_clicks):
    return db.clear('pca.')

@app.callback(
    Output("pca-selected-scatter-plot", "children"),
    [Input('pca-selected-cleaned-file', 'value')]
)
def pca_display_selected_file_scatter_plot(value):
    db_value = db.get("pca.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("pca.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("pca.data", df)

    stats = df.describe(include = 'all').head(6).round(5)
    stats.insert(loc=0, column='Statistics', value=['Count','unique','top','freq','Mean','Standard Deviation'])
    stats = stats.drop(stats.index[[1,2,3]])

    div = html.Div([
        common.msg("Selected cleaned file: "+ file),
        dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True, style = common.table_style),
        html.Div([html.H3("Data Statistics")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        html.Div([html.H2("Scatter Plot")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Class"),
                dcc.Dropdown(
                    id = 'pca-class',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'pca-x-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'pca-y-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'pca-scatter-plot-button'),
                html.Div([], id = "pca-class-do-nothing"),
                html.Div([], id = "pca-x-axis-do-nothing"),
                html.Div([], id = "pca-y-axis-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="pca-scatter-plot")
        ]),
        html.Br(),
        get_pca_model_properties_div(df),
        html.Div([], id = "pca-trained-model", style = {'margin': '10px'}),
    ])

    return div

def get_pca_model_properties_div(df):
    pca_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Perform Principle Component Analysis (PCA)"),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="pca-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            html.Br(),
            dbc.Button("Execute", color="primary", id = 'pca-train-model'),

            html.Div([], id = "pca-model-class-do-nothing"),
            html.Div([], id = "pca-model-variables-do-nothing"),
            html.Div([], id = "pca-prediction-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    pca_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(pca_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return pca_model_properties_div

@app.callback(
    Output('pca-model-variables-do-nothing' , "children"),
    [Input('pca-model-variables', 'value')]
)
def pca_model_variables(value):
    if not value is None:
        db.put("pca.model_variables", value)
    return None

@app.callback(
    Output('pca-trained-model' , "children"),
    [Input('pca-train-model', 'n_clicks')]
)
def pca_model_train(n_clicks):
    var = db.get('pca.model_variables')
    if var is None:
        div = ""
    elif (not var is None):
        try:
            df = db.get('pca.data')
            cov_mat, eig_vals, eig_vecs, eig_pairs = perform_pca(df[var].values)
            cov_df = pd.DataFrame(cov_mat).round(4)
        except Exception as e:
            traceback.print_exc()
            return common.error_msg("Exception during training model: " + str(e))

        list = [
            html.H2('Covariance Matrix:'),
            dbc.Table.from_dataframe(cov_df, striped=True, bordered=True, hover=True, style = common.table_style),
        ]

        i = 0
        for k,v in eig_pairs.items():
            i = i + 1
            list.append(html.H2('Eigen Values: ' + str(round(k, 4))))
            list.append(html.H2('Eigen Vector: ' + str(v)))
            list.append(html.Br())

        if len(var) == 2:
            x_col = var[0]
            y_col = var[1]
            xmax = max(df[x_col])
            i = 0
            for key in sorted(eig_pairs.keys(), reverse=True):
                if i ==0:
                    x1 = eig_pairs[key][0]
                    y1 = eig_pairs[key][1]
                    m = y1/x1
                    ymax = m * xmax
                    x1 = [0, x1, xmax]
                    y1 = [0, y1, ymax]
                    k1 = str(round(key, 4))
                    i = i + 1
                else:
                    x2 = eig_pairs[key][0]
                    y2 = eig_pairs[key][1]
                    x2 = [0, x2]
                    y2 = [0, y2]
                    k2 = str(round(key, 4))


            graph = dcc.Graph(
                id='pca-x-vs-y',
                figure={
                    'data': [
                        go.Scatter(
                            x=df[x_col],
                            y=df[y_col],
                            mode='markers',
                            opacity=0.8,
                            marker={
                                'size': 15,
                                'line': {'width': 0.5, 'color': 'white'}
                            },
                            name='Data Points'
                        ),
                        go.Scatter(
                            x=x1,
                            y=y1,
                            mode='lines',
                            opacity=0.8,
                            name='Eigen Vector - V1 - ' + k1
                        ),
                        go.Scatter(
                            x=x2,
                            y=y2,
                            mode='lines',
                            opacity=0.8,
                            name='Eigen Vector - V2 - ' + k2
                        )
                    ],
                    'layout': dict(
                        title='Scatter Plot',
                        xaxis={'title': x_col},
                        yaxis={'title': y_col},
                        margin={'l': 40, 'b': 40},
                        #legend={'x': 0, 'y': 1},
                        hovermode='closest'
                    )
                }
            )
            list.append(graph)

        div = html.Div(list)
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div
