import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils

layout = html.Div([
    common.navbar("Classification - Linearly Separable"),
    html.Br(),
    html.Div([],id = "linear-classification-selected")
])

@app.callback(
    Output("linear-classification-selected", "children"),
    [Input('linear-classification', 'href')]
)
def selected_file(href):
    file = db.get("file")
    format = db.get("format")
    sep = db.get("file_separator")
    header = db.get("file_header")
    df = db.get("data")
    div = None
    if file is None:
        div =  ""
    elif df is None:
        div  = [common.msg("Selected File: " + file + " Selected Format: " + format), common.error_msg("Please apply file properties!!")]
    else:
        msg = "File=" + file + "  Format=" + format +"  Separator=" + sep + "  Header="+ str(header)
        table = dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True, style = common.table_style)
        div = html.Div([
            common.msg(msg),
            table,
            html.Br(),
            html.Div([common.msg("Scatter Plot")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Class"),
                    dcc.Dropdown(
                        id = 'cl-class',
                        options=[{'label':col, 'value':col} for col in [*df]],
                        value=None,
                        multi=False
                    ),
                    html.Br(),
                    dbc.Label("Select X Axis"),
                    dcc.Dropdown(
                        id = 'cl-x-axis',
                        options=[{'label':col, 'value':col} for col in [*df]],
                        value=None,
                        multi=False
                    ),
                    html.Br(),
                    dbc.Label("Select Y Axis"),
                    dcc.Dropdown(
                        id = 'cl-y-axis',
                        options=[{'label':col, 'value':col} for col in [*df]],
                        value=None,
                        multi=False
                    ),
                    html.Br(),
                    dbc.Button("Plot", color="primary", id = 'cl-scatter-plot-button'),
                    html.Div([], id = "cl-class-do-nothing"),
                    html.Div([], id = "cl-x-axis-do-nothing"),
                    html.Div([], id = "cl-y-axis-do-nothing")
                ], md=2,
                style = {'margin': '10px', 'font-size': '16px'}),
                dbc.Col([], md=9, id="cl-scatter-plot")
            ])


        ])
    return div

@app.callback(
    Output("cl-scatter-plot", "children"),
    [Input('cl-scatter-plot-button', 'n_clicks')]
)
def cl_scatter_plot(n):
    df = db.get("data")
    clazz_col = db.get("cl_class")
    x_col = db.get("cl_x_axis")
    y_col = db.get("cl_y_axis")
    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='cl-x-vs-y',
        figure={
            'data': [
                go.Scatter(
                    x=df[df[clazz_col] == clazz][x_col],
                    y=df[df[clazz_col] == clazz][y_col],
                    text=df[df[clazz_col] == clazz][clazz_col],
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=clazz
                ) for clazz in df[clazz_col].unique()
            ],
            'layout': dict(
                #title='Scatter Plot',
                xaxis={'title': x_col},
                yaxis={'title': y_col},
                margin={'l': 40, 'b': 40},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
    return graph

@app.callback(
    Output('cl-class-do-nothing' , "children"),
    [Input('cl-class', 'value')]
)
def cl_class(value):
    if not value is None:
        db.put("cl_class", value)
    return None

@app.callback(
    Output('cl-x-axis-do-nothing' , "children"),
    [Input('cl-x-axis', 'value')]
)
def cl_x_axis(value):
    if not value is None:
        db.put("cl_x_axis", value)
    return None

@app.callback(
    Output('cl-y-axis-do-nothing' , "children"),
    [Input('cl-y-axis', 'value')]
)
def cl_y_axis(value):
    if not value is None:
        db.put("cl_y_axis", value)
    return None
