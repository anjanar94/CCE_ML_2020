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

from ml.knn import knn_predict
from ml.knn import load_input_csv

layout = html.Div([
    common.navbar("K Nearest Neighbors (KNN)"),
    html.Div([], style = {'padding': '30px'}),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'knn-load-cleaned-files', className="mr-2", style={'display': 'inline-block'}),
        dbc.Button("Clear", color="secondary", id = 'knn-clear-db', className="mr-2", style={'display': 'inline-block'})
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'knn-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "knn-clear-db-do-nothing"),
    html.Div([],id = "knn-selected-div")
])

@app.callback(
    Output("knn-selected-cleaned-file", "options"),
    [Input('knn-load-cleaned-files', 'n_clicks')]
)
def selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("knn-clear-db-do-nothing", "options"),
    [Input('knn-clear-db', 'n_clicks')]
)
def selected_file(n_clicks):
    return db.clear('knn.')

@app.callback(
    Output("knn-selected-div", "children"),
    [Input('knn-selected-cleaned-file', 'value')]
)
def knn_display_selected_file_scatter_plot(value):
    db_value = db.get("knn.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("knn.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("knn.data", df)

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
                    id = 'knn-class',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'knn-x-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'knn-y-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'knn-scatter-plot-button'),
                html.Div([], id = "knn-class-do-nothing"),
                html.Div([], id = "knn-x-axis-do-nothing"),
                html.Div([], id = "knn-y-axis-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="knn-scatter-plot")
        ]),
        html.Br(),
        get_knn_model_properties_div(df),
        html.Div([], id = "knn-trained-model", style = {'margin': '10px'}),
    ])

    return div

@app.callback(
    Output("knn-scatter-plot", "children"),
    [Input('knn-scatter-plot-button', 'n_clicks')]
)
def knn_scatter_plot(n):
    df = db.get("knn.data")
    clazz_col = db.get("knn.class")
    x_col = db.get("knn.x_axis")
    y_col = db.get("knn.y_axis")
    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='knn-x-vs-y',
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
    Output('knn-class-do-nothing' , "children"),
    [Input('knn-class', 'value')]
)
def knn_class(value):
    if not value is None:
        db.put("knn.class", value)
    return None

@app.callback(
    Output('knn-x-axis-do-nothing' , "children"),
    [Input('knn-x-axis', 'value')]
)
def knn_x_axis(value):
    if not value is None:
        db.put("knn.x_axis", value)
    return None

@app.callback(
    Output('knn-y-axis-do-nothing' , "children"),
    [Input('knn-y-axis', 'value')]
)
def knn_y_axis(value):
    if not value is None:
        db.put("knn.y_axis", value)
    return None

def get_knn_model_properties_div(df):
    knn_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train K Nearest Neighbors (KNN) Model"),
            dbc.Label("Class"),
            dcc.Dropdown(
                id="knn-model-class",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=False),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="knn-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            dbc.Label("Train Data %"),
            dbc.Input(id="knn-train-data", placeholder="70,75,80,85,90", type="number"),
            dbc.Label("Kth Distance"),
            dbc.Input(id="knn-distance", placeholder="1,2,3,4,5,6,7...", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'knn-train-model'),

            html.Div([], id = "knn-model-class-do-nothing"),
            html.Div([], id = "knn-model-variables-do-nothing"),
            html.Div([], id = "knn-train-data-do-nothing"),
            html.Div([], id = "knn-distance-do-nothing"),
            html.Div([], id = "knn-prediction-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    knn_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(knn_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return knn_model_properties_div

@app.callback(
    Output('knn-model-class-do-nothing' , "children"),
    [Input('knn-model-class', 'value')]
)
def knn_model_class(value):
    if not value is None:
        db.put("knn.model_class", value)
    return None

@app.callback(
    Output('knn-model-variables-do-nothing' , "children"),
    [Input('knn-model-variables', 'value')]
)
def knn_model_variables(value):
    if not value is None:
        db.put("knn.model_variables", value)
    return None

@app.callback(
    Output('knn-train-data-do-nothing' , "children"),
    [Input('knn-train-data', 'value')]
)
def knn_model_train(value):
    if not value is None:
        db.put("knn.model_train", value)
    return None

@app.callback(
    Output('knn-distance-do-nothing' , "children"),
    [Input('knn-distance', 'value')]
)
def knn_model_lr(value):
    if not value is None:
        db.put("knn.distance", value)
    return None


@app.callback(
    Output('knn-trained-model' , "children"),
    [Input('knn-train-model', 'n_clicks')]
)
def knn_model_train(n_clicks):
    c = db.get('knn.model_class')
    var = db.get('knn.model_variables')
    train = db.get('knn.model_train')
    k = db.get('knn.distance')
    file = db.get("knn.file")
    if c is None and var is None and train is None and k is None:
        div = ""
    elif train is None or train < 0 or train > 100:
        div = common.error_msg('Training % should be between 0 - 100 !!')
    elif (not c is None) and (not var is None) and (not train is None) and (not k is None):

        try:
            cols = [] + var
            cols.append(c)
            df = db.get('knn.data')
            df = df[cols]

            train_df, test_df = train_test_split(df, test_size=(100-train)/100)

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')

            train_path = FileUtils.path('knn', file.split('.')[0]+'-train')
            train_df.to_csv(train_path, index=False)
            train_dataset = load_input_csv(train_path)
            print(train_dataset)

            test_path = FileUtils.path('knn', file.split('.')[0]+'-test')
            test_df.to_csv(test_path, index=False)
            test_dataset = load_input_csv(test_path)
            print(test_dataset)

            result = knn_predict(train_dataset, test_dataset, k)
            print(result)

            summary = {}
            summary['Total Training Data'] = len(train_df)
            summary['Total Testing Data'] = len(test_df)
            summary['Total Number of Features in Dataset'] = len(var)
            summary['Model Accuracy %'] = 'TODO'
            summary['Features'] = str(var)
            summary_df = pd.DataFrame(summary.items(), columns=['Parameters', 'Value'])

            db.put('knn.data_train', train_df)
            db.put('knn.data_test', test_df)
            db.put('knn.model_summary', summary)
            #db.put('knn.model_instance', instanceOfLR)
            #confusion_df = get_confusion_matrix(test_df, c, var, instanceOfLR)
        except Exception as e:
            traceback.print_exc()
            return common.error_msg("Exception during training model: " + str(e))

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Model Parameters & Summary:'),
            dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            #html.H2('Confusion Matrix (Precision & Recall):'),
            #dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
            #html.H2('Prediction/Classification:'),
            #html.P('Features to be Predicted (comma separated): ' + ','.join(var), style = {'font-size': '16px'}),
            #dbc.Input(id="knn-prediction-data", placeholder=','.join(var), type="text"),
            #html.Br(),
            #dbc.Button("Predict", color="primary", id = 'knn-predict'),
            #html.Div([], id = "knn-prediction"),
            #html.Div([],id = "knn-predicted-scatter-plot")
        ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div

@app.callback(
    Output('knn-prediction-data-do-nothing' , "children"),
    [Input('knn-prediction-data', 'value')]
)
def knn_model_prediction_data(value):
    if not value is None:
        db.put("knn.model_prediction_data", value)
    return None

@app.callback(
    [Output('knn-prediction' , "children"),
    Output('knn-predicted-scatter-plot' , "children")],
    [Input('knn-predict', 'n_clicks')]
)
def knn_model_predict(n_clicks):
    predict_data = db.get("knn.model_prediction_data")
    summary = db.get('knn.model_summary')
    lr_instance = db.get('knn.model_instance')
    n_var = summary['Total Number of Features in Dataset']
    if predict_data is None:
        return ("" , "")
    if len(predict_data.split(',')) != n_var:
        return (common.error_msg('Enter Valid Prediction Data!!'), "")
    try:
        feature_vector = get_predict_data_list(predict_data)
        feature_vector = np.array(feature_vector)
        prediction = lr_instance.predict(feature_vector)
        db.put('knn.prediction', prediction)
    except Exception as e:
        return (common.error_msg("Exception during prediction: " + str(e)), "")
    df = db.get('knn.data_train')
    df = df.iloc[:, :-1]
    div = html.Div([
        html.Div([html.H2("Predicted & Testing Data Scatter Plot")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'knn-x-axis-predict',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'knn-y-axis-predict',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'knn-predict-scatter-plot-button'),
                html.Div([], id = "knn-x-axis-predict-do-nothing"),
                html.Div([], id = "knn-y-axis-predict-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="knn-scatter-plot-predict")
        ]),

    ])
    return (common.success_msg('Predicted/Classified Class = ' + prediction), div)

@app.callback(
    Output('knn-x-axis-predict-do-nothing' , "children"),
    [Input('knn-x-axis-predict', 'value')]
)
def knn_x_axis(value):
    if not value is None:
        db.put("knn.x_axis_predict", value)
    return None

@app.callback(
    Output('knn-y-axis-predict-do-nothing' , "children"),
    [Input('knn-y-axis-predict', 'value')]
)
def knn_y_axis(value):
    if not value is None:
        db.put("knn.y_axis_predict", value)
    return None

@app.callback(
    Output("knn-scatter-plot-predict", "children"),
    [Input('knn-predict-scatter-plot-button', 'n_clicks')]
)
def knn_scatter_plot(n):
    df = db.get("knn.data_test")
    clazz_col = db.get('knn.model_class')
    x_col = db.get("knn.x_axis_predict")
    y_col = db.get("knn.y_axis_predict")
    predict_data = db.get("knn.model_prediction_data")
    prediction = db.get('knn.prediction')

    feature_vector = get_predict_data_list(predict_data)
    feature_vector.append('Predicted-'+prediction)
    df.loc[len(df)] = feature_vector

    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='knn-x-vs-y-predict',
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
