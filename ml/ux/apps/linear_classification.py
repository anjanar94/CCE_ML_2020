import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils

from ml.linear_classification import linearClassifier
from ml.linear_classification import LogisticRegression

layout = html.Div([
    common.navbar("Classification - Linearly Separable"),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'cl-load-cleaned-files')
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'cl-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([],id = "linear-classification-selected-scatter-plot")
])

@app.callback(
    Output("cl-selected-cleaned-file", "options"),
    [Input('cl-load-cleaned-files', 'n_clicks')]
)
def selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("linear-classification-selected-scatter-plot", "children"),
    [Input('cl-selected-cleaned-file', 'value')]
)
def cl_display_selected_file_scatter_plot(value):
    db_value = db.get("cl.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("cl.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("cl.data", df)

    stats = df.describe().head(3).round(5)
    stats.insert(loc=0, column='Statistics', value=['Count', 'Mean', 'Standard Deviation'])

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
        ]),
        html.Br(),
        get_cl_model_properties_div(df),
        html.Div([], id = "cl-trained-model", style = {'margin': '10px'}),
    ])

    return div

@app.callback(
    Output("cl-scatter-plot", "children"),
    [Input('cl-scatter-plot-button', 'n_clicks')]
)
def cl_scatter_plot(n):
    df = db.get("cl.data")
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

def get_cl_model_properties_div(df):
    cl_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train Linear Classification Model"),
            dbc.Label("Class"),
            dcc.Dropdown(
                id="cl-model-class",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=False),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="cl-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            dbc.Label("Train Data %"),
            dbc.Input(id="cl-train-data", placeholder="70,75,80,85,90", type="number"),
            dbc.Label("Learning Rate"),
            dbc.Input(id="cl-learning-rate", placeholder="0.005, 0.01, 0.05", type="number"),
            dbc.Label("Epoch"),
            dbc.Input(id="cl-epoch", placeholder="100,200,300,400,500..", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'cl-train-model'),

            html.Div([], id = "cl-model-class-do-nothing"),
            html.Div([], id = "cl-model-variables-do-nothing"),
            html.Div([], id = "cl-train-data-do-nothing"),
            html.Div([], id = "cl-learning-rate-do-nothing"),
            html.Div([], id = "cl-epoch-do-nothing"),
            html.Div([], id = "cl-prediction-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    cl_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(cl_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return cl_model_properties_div

@app.callback(
    Output('cl-model-class-do-nothing' , "children"),
    [Input('cl-model-class', 'value')]
)
def cl_model_class(value):
    if not value is None:
        db.put("cl.model_class", value)
    return None

@app.callback(
    Output('cl-model-variables-do-nothing' , "children"),
    [Input('cl-model-variables', 'value')]
)
def cl_model_variables(value):
    if not value is None:
        db.put("cl.model_variables", value)
    return None

@app.callback(
    Output('cl-train-data-do-nothing' , "children"),
    [Input('cl-train-data', 'value')]
)
def cl_model_train(value):
    if not value is None:
        db.put("cl.model_train", value)
    return None

@app.callback(
    Output('cl-learning-rate-do-nothing' , "children"),
    [Input('cl-learning-rate', 'value')]
)
def cl_model_lr(value):
    if not value is None:
        db.put("cl.model_lr", value)
    return None

@app.callback(
    Output('cl-epoch-do-nothing' , "children"),
    [Input('cl-epoch', 'value')]
)
def cl_model_epoch(value):
    if not value is None:
        db.put("cl.model_epoch", value)
    return None


@app.callback(
    Output('cl-trained-model' , "children"),
    [Input('cl-train-model', 'n_clicks')]
)
def cl_model_train(n_clicks):
    c = db.get('cl.model_class')
    var = db.get('cl.model_variables')
    train = db.get('cl.model_train')
    #test = db.get('cl.model_test')
    lr = db.get('cl.model_lr')
    epoch = db.get('cl.model_epoch')
    if c is None and var is None and train is None and lr is None and epoch is None:
        div = ""
    elif train is None or train < 0 or train > 100:
        div = common.error_msg('Training % should be between 0 - 100 !!')
    elif (not c is None) and (not var is None) and (not train is None) and (not lr is None) and (not epoch is None):
        #parameters = "Training Data = " + str(train) + " % Testing Data = " + str(100 - train) + " % Learning rate = " + str(lr) + " Epoch = " + str(epoch)

        try:
            cols = [] + var
            cols.append(c)
            df = db.get('cl.data')
            df = df[cols]

            msk = np.random.rand(len(df)) < (train / 100)
            train_df = df[msk]
            test_df = df[~msk]

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')

            instanceOfLR, summary = linearClassifier(train_df, test_df, len(var), lr, epoch)
            summary['Features'] = str(var)
            summary_df = pd.DataFrame(summary.items(), columns=['Parameters', 'Value'])
            db.put('cl.model_summary', summary)
            db.put('cl.model_instance', instanceOfLR)
        except Exception as e:
            return common.error_msg("Exception during training model: " + str(e))

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Model Parameters & Summary:'),
            dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Prediction/Classification:'),
            dbc.Label('Features to be Predicted (Comma separated)'),
            dbc.Input(id="cl-prediction-data", placeholder="f1,f2,f3,f4...", type="text"),
            html.Br(),
            dbc.Button("Predict", color="primary", id = 'cl-predict'),
            html.Div([], id = "cl-prediction")
        ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div

@app.callback(
    Output('cl-prediction-data-do-nothing' , "children"),
    [Input('cl-prediction-data', 'value')]
)
def cl_model_prediction_data(value):
    if not value is None:
        db.put("cl.model_prediction_data", value)
    return None

@app.callback(
    Output('cl-prediction' , "children"),
    [Input('cl-predict', 'n_clicks')]
)
def cl_model_predict(n_clicks):
    predict_data = db.get("cl.model_prediction_data")
    summary = db.get('cl.model_summary')
    lr_instance = db.get('cl.model_instance')
    n_var = summary['Total Number of Features in Dataset']
    if predict_data is None:
        return ""
    if len(predict_data.split(',')) != n_var:
        return common.error_msg('Enter Valid Prediction Data!!')
    try:
        predict_data = predict_data.split(',')
        feature_vector = []
        for d in predict_data:
            feature_vector.append(float(d))
        feature_vector = np.array(feature_vector)
        prediction = lr_instance.predict(feature_vector)
    except Exception as e:
        return common.error_msg("Exception during prediction: " + str(e))

    return common.success_msg('Predicted/Classified Class = ' + prediction)

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
