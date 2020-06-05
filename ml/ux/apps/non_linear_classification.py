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

from ml.non_linear_classification import non_separable_train

layout = html.Div([
    common.navbar("Classification - Linearly Non-Separable"),
    html.Div([], style = {'padding': '30px'}),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'nlcl-load-cleaned-files', className="mr-2", style={'display': 'inline-block'}),
        dbc.Button("Clear", color="secondary", id = 'nlcl-clear-db', className="mr-2", style={'display': 'inline-block'})
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'nlcl-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "nlcl-clear-db-do-nothing"),
    html.Div([],id = "nlcl-selected-div")
])

@app.callback(
    Output("nlcl-selected-cleaned-file", "options"),
    [Input('nlcl-load-cleaned-files', 'n_clicks')]
)
def selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("nlcl-clear-db-do-nothing", "options"),
    [Input('nlcl-clear-db', 'n_clicks')]
)
def selected_file(n_clicks):
    return db.clear('nlcl.')

@app.callback(
    Output("nlcl-selected-div", "children"),
    [Input('nlcl-selected-cleaned-file', 'value')]
)
def nlcl_display_selected_file_scatter_plot(value):
    db_value = db.get("nlcl.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("nlcl.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("nlcl.data", df)

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
                    id = 'nlcl-class',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'nlcl-x-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'nlcl-y-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'nlcl-scatter-plot-button'),
                html.Div([], id = "nlcl-class-do-nothing"),
                html.Div([], id = "nlcl-x-axis-do-nothing"),
                html.Div([], id = "nlcl-y-axis-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="nlcl-scatter-plot")
        ]),
        html.Br(),
        get_nlcl_model_properties_div(df),
        html.Div([], id = "nlcl-trained-model", style = {'margin': '10px'}),
    ])

    return div

@app.callback(
    Output("nlcl-scatter-plot", "children"),
    [Input('nlcl-scatter-plot-button', 'n_clicks')]
)
def nlcl_scatter_plot(n):
    df = db.get("nlcl.data")
    clazz_col = db.get("nlcl.class")
    x_col = db.get("nlcl.x_axis")
    y_col = db.get("nlcl.y_axis")
    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='nlcl-x-vs-y',
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
    Output('nlcl-class-do-nothing' , "children"),
    [Input('nlcl-class', 'value')]
)
def nlcl_class(value):
    if not value is None:
        db.put("nlcl.class", value)
    return None

@app.callback(
    Output('nlcl-x-axis-do-nothing' , "children"),
    [Input('nlcl-x-axis', 'value')]
)
def nlcl_x_axis(value):
    if not value is None:
        db.put("nlcl.x_axis", value)
    return None

@app.callback(
    Output('nlcl-y-axis-do-nothing' , "children"),
    [Input('nlcl-y-axis', 'value')]
)
def nlcl_y_axis(value):
    if not value is None:
        db.put("nlcl.y_axis", value)
    return None

def get_nlcl_model_properties_div(df):
    nlcl_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train Linear Classification Model"),
            dbc.Label("Class"),
            dcc.Dropdown(
                id="nlcl-model-class",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=False),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="nlcl-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            dbc.Label("Train Data %"),
            dbc.Input(id="nlcl-train-data", placeholder="70,75,80,85,90", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'nlcl-train-model'),

            html.Div([], id = "nlcl-model-class-do-nothing"),
            html.Div([], id = "nlcl-model-variables-do-nothing"),
            html.Div([], id = "nlcl-train-data-do-nothing"),
            html.Div([], id = "nlcl-learning-rate-do-nothing"),
            html.Div([], id = "nlcl-epoch-do-nothing"),
            html.Div([], id = "nlcl-prediction-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    nlcl_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(nlcl_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return nlcl_model_properties_div

@app.callback(
    Output('nlcl-model-class-do-nothing' , "children"),
    [Input('nlcl-model-class', 'value')]
)
def nlcl_model_class(value):
    if not value is None:
        db.put("nlcl.model_class", value)
    return None

@app.callback(
    Output('nlcl-model-variables-do-nothing' , "children"),
    [Input('nlcl-model-variables', 'value')]
)
def nlcl_model_variables(value):
    if not value is None:
        db.put("nlcl.model_variables", value)
    return None

@app.callback(
    Output('nlcl-train-data-do-nothing' , "children"),
    [Input('nlcl-train-data', 'value')]
)
def nlcl_model_train(value):
    if not value is None:
        db.put("nlcl.model_train", value)
    return None

@app.callback(
    Output('nlcl-trained-model' , "children"),
    [Input('nlcl-train-model', 'n_clicks')]
)
def nlcl_model_train(n_clicks):
    c = db.get('nlcl.model_class')
    var = db.get('nlcl.model_variables')
    train = db.get('nlcl.model_train')
    if c is None and var is None and train is None:
        div = ""
    elif train is None or train < 0 or train > 100:
        div = common.error_msg('Training % should be between 0 - 100 !!')
    elif len(var) != 2:
        div = common.error_msg('Select Two Features!!')
    elif (not c is None) and (not var is None) and (not train is None):

        try:
            cols = [] + var
            cols.append(c)
            df = db.get('nlcl.data')
            df = df[cols]


            train_df, test_df = train_test_split(df, test_size=(100-train)/100)
            train_df.columns = ['X1', 'X2', 'Class']

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')

            model = non_separable_train(train_df)
            print(model)
            summary = {}
            summary['Total Training Data'] = len(train_df)
            summary['Total Testing Data'] = len(test_df)
            summary['Total Number of Features in Dataset'] = len(var)
            summary['Model Accuracy %'] = 'TODO'
            summary['Features'] = str(var)
            summary_df = pd.DataFrame(summary.items(), columns=['Parameters', 'Value'])

            db.put('nlcl.data_train', train_df)
            db.put('nlcl.data_test', test_df)
            db.put('nlcl.model_summary', summary)
            db.put('nlcl.model_instance', model)
            #confusion_df = get_confusion_matrix(test_df, c, var, instanceOfLR)
        except Exception as e:
            traceback.print_exc()
            return common.error_msg("Exception during training model: " + str(e))

        clazz_col = c
        train_df.columns = cols
        df = train_df
        x_col = var[0]
        y_col = var[1]
        x1, y1 = get_rect_coordinates(model[0])
        x2, y2 = get_rect_coordinates(model[1])
        x3, y3 = get_rect_coordinates(model[2])
        graph_data = [
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
        ]
        graph_data.append(go.Scatter(x=x1, y=y1, text = 'Specific Rectangle', name = 'Specific Rectangle'))
        graph_data.append(go.Scatter(x=x3, y=y3, text = 'Optimal Rectangle', name = 'Optimal Rectangle'))
        graph_data.append(go.Scatter(x=x2, y=y2, text = 'Generic Rectangle', name = 'Generic Rectangle'))

        graph = dcc.Graph(
            id='nlcl-x-vs-y-rectangle',
            figure={
                'data': graph_data,
                'layout': dict(
                    title='Boundaries & Train Data Set Scatter Plot',
                    xaxis={'title': x_col},
                    yaxis={'title': y_col},
                    margin={'l': 40, 'b': 40},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
            }
        )

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Model Parameters & Summary:'),
            dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            graph,
            #html.H2('Confusion Matrix (Precision & Recall):'),
            #dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Prediction/Classification:'),
            html.P('Features to be Predicted (comma separated): ' + ','.join(var), style = {'font-size': '16px'}),
            dbc.Input(id="nlcl-prediction-data", placeholder=','.join(var), type="text"),
            html.Br(),
            dbc.Button("Predict", color="primary", id = 'nlcl-predict'),
            html.Div([], id = "nlcl-prediction"),
            html.Div([],id = "nlcl-predicted-scatter-plot")
        ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div

@app.callback(
    Output('nlcl-prediction-data-do-nothing' , "children"),
    [Input('nlcl-prediction-data', 'value')]
)
def nlcl_model_prediction_data(value):
    if not value is None:
        db.put("nlcl.model_prediction_data", value)
    return None

@app.callback(
    [Output('nlcl-prediction' , "children"),
    Output('nlcl-predicted-scatter-plot' , "children")],
    [Input('nlcl-predict', 'n_clicks')]
)
def nlcl_model_predict(n_clicks):
    c = db.get('nlcl.model_class')
    predict_data = db.get("nlcl.model_prediction_data")
    test_df = db.get('nlcl.data_test')
    #summary = db.get('nlcl.model_summary')
    model = db.get('nlcl.model_instance')
    var = db.get('nlcl.model_variables')
    n_var = len(var)
    if predict_data is None:
        return ("" , "")
    if len(predict_data.split(',')) != n_var:
        return (common.error_msg('Enter Valid Prediction Data!!'), "")
    try:
        cols = [] + var
        cols.append(c)
        feature_vector = get_predict_data_list(predict_data)
        #TODO Team 3 Predict API is not available.
        ""
    except Exception as e:
        return (common.error_msg("Exception during prediction: " + str(e)), "")

    clazz_col = c
    test_df.columns = cols
    df = test_df
    x_col = var[0]
    y_col = var[1]
    xp = [feature_vector[0]]
    yp = [feature_vector[1]]
    x1, y1 = get_rect_coordinates(model[0])
    x2, y2 = get_rect_coordinates(model[1])
    x3, y3 = get_rect_coordinates(model[2])
    graph_data = [
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
    ]
    graph_data.append(go.Scatter(x=xp, y=yp,
        mode='markers',
        opacity=0.8,
        marker={
            'size': 20,
            'line': {'width': 0.5, 'color': 'white'}
        },
        text = 'Predicted - DataPoint',
        name = 'Predicted - DataPoint'))
    graph_data.append(go.Scatter(x=x1, y=y1, text = 'Specific Rectangle', name = 'Specific Rectangle'))
    graph_data.append(go.Scatter(x=x3, y=y3, text = 'Optimal Rectangle', name = 'Optimal Rectangle'))
    graph_data.append(go.Scatter(x=x2, y=y2, text = 'Generic Rectangle', name = 'Generic Rectangle'))

    graph = dcc.Graph(
        id='nlcl-x-vs-y-predict',
        figure={
            'data': graph_data,
            'layout': dict(
                title='Boundaries, Predict Data Point & Test Data Set Scatter Plot',
                xaxis={'title': x_col},
                yaxis={'title': y_col},
                margin={'l': 40, 'b': 40},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )

    div = html.Div([
        graph
    ])
    return ("", div)

def get_predict_data_list(predict_data: str) -> []:
    predict_data = predict_data.split(',')
    feature_vector = []
    for d in predict_data:
        feature_vector.append(float(d))
    return feature_vector

def get_distinct_count_df(df, c, col):
    classes = df[c].unique()
    distinct_count = {}
    total = 0
    for clazz in classes:
        tdf = df.loc[df[c] == clazz]
        count = len(tdf)
        distinct_count[clazz] = count
        total = total + count
    distinct_count['Gross Total = '] = total
    distinct_count = pd.DataFrame(distinct_count.items(), columns=['Class', col])
    return distinct_count

def get_confusion_matrix(df, c, var, model):
    classes = df[c].unique()
    d = {}
    for clazz in classes:
        d[clazz] = {'t_rel':0, 't_ret':0, 'rr':0}
    for index, row in df.iterrows():
        feature_vector = []
        for v in var:
            feature_vector.append(row[v])
        feature_vector = np.array(feature_vector)
        clazz = row[c]
        prediction = model.predict(feature_vector)
        d[clazz]['t_rel'] = d[clazz]['t_rel'] + 1
        d[prediction]['t_ret'] = d[prediction]['t_ret'] + 1
        if clazz == prediction:
            d[clazz]['rr'] = d[clazz]['rr'] + 1
    df = pd.DataFrame(columns=['Class', 'Total Retrieved Records', 'Total Relevant Records', 'Retrieved & Relevant', 'Precision', 'Recall'])
    i = 0
    for k, v in d.items():
        df.loc[i] = [k, v['t_ret'],v['t_rel'], v['rr'], round(v['rr']/v['t_ret'], 4), round(v['rr']/v['t_rel'], 4)]
        i = i+1
    return df

def get_rect_coordinates(cr: []):
    x = [0,0,0,0,0]
    y = [0,0,0,0,0]
    x[0] = cr[0]
    y[0] = cr[1]

    x[1] = cr[2]
    y[1] = cr[1]

    x[2] = cr[2]
    y[2] = cr[3]

    x[3] = cr[0]
    y[3] = cr[3]

    x[4] = cr[0]
    y[4] = cr[1]

    return x, y
