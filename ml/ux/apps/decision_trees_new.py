import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import traceback

import pandas as pd
import numpy as np

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils

from ml.decision_trees import DecisionTree

layout = html.Div([
    common.navbar("Decision Trees New"),
    html.Div([], style = {'padding': '30px'}),
    html.Br(),
    html.H2('Decision Tree API Integration for Data Set banknote.csv'),
    html.Div([],id = "decision-trees-new-selected-div")
])

@app.callback(
    Output("decision-trees-new-selected-div", "children"),
    [Input('decision-trees-new', 'value')]
)
def dtn_display_selected_file_scatter_plot(value):
    value = "banknote"
    db.put("dtn.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("dtn.data", df)

    db.put('dtn.model_class', 'class')
    db.put('dtn.model_variables', ['variance','skewness','curtosis','entropy'])

    call_path = FileUtils.path('nets', 'dt_banknote_call1.csv')
    cdf = DataUtils.read_csv(call_path)

    trace_1 = go.Scatter(x = cdf['max_depth'], y = cdf['avg_train_score'], name = 'Average Train Score')
    trace_2 = go.Scatter(x = cdf['max_depth'], y = cdf['avg_test_score'], name = 'Average Test Score')
    title = go.Layout(title = 'Depth of Tree Vs Performance Plot', hovermode = 'closest', xaxis={'title': 'Depth of Tree'}, yaxis={'title': 'Performance'})
    fig = go.Figure(data = [trace_1, trace_2], layout = title)

    div = html.Div([
        common.msg("Selected cleaned file: "+ file),
        dbc.Table.from_dataframe(df.head(10).round(5).astype(str), striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        html.H2('Using Default parameters for both max_depth and min_size.'),
        html.H2('Max Depth = 2 to 15'),
        html.H2('Min Size = 10'),
        dbc.Table.from_dataframe(cdf.round(4), striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        dcc.Graph(id='dtn-plot', figure=fig),
        html.Br(),
        get_dtn_model_properties_div(df),
        html.Div([], id = "dtn-trained-model", style = {'margin': '10px'}),
    ])

    return div

def get_dtn_model_properties_div(df):
    dtn_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train Decision Tree Model"),
            dbc.Label("Max Depth"),
            dcc.Dropdown(
                id="dtn-max-depth",
                options=[{'label':col, 'value':col} for col in range(2,15)],
                value=None,
                multi=False),
            dbc.Label("Min Size"),
            dbc.Input(id="dtn-min-size", placeholder="10,20,30...", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'dtn-train-model'),

            html.Div([], id = "dtn-max-depth-do-nothing"),
            html.Div([], id = "dtn-min-size-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    dtn_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(dtn_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return dtn_model_properties_div

@app.callback(
    Output('dtn-max-depth-do-nothing' , "children"),
    [Input('dtn-max-depth', 'value')]
)
def dtn_model_class(value):
    if not value is None:
        db.put("dtn.max_depth", value)
    return None

@app.callback(
    Output('dtn-min-size-do-nothing' , "children"),
    [Input('dtn-min-size', 'value')]
)
def dtn_model_variables(value):
    if not value is None:
        db.put("dtn.min_size", value)
    return None

@app.callback(
    Output('dtn-trained-model' , "children"),
    [Input('dtn-train-model', 'n_clicks')]
)
def dtn_model_train(n_clicks):
    c = db.get('dtn.model_class')
    var = db.get('dtn.model_variables')
    max_depth = db.get('dtn.max_depth')
    min_size = db.get('dtn.min_size')
    if c is None or var is None or max_depth is None or min_size is None:
        div = ""
    elif (not c is None) and (not var is None) and (not max_depth is None) and (not min_size is None):
        try:
            cols = [] + var
            cols.append(c)
            df = db.get('dtn.data')
            df = df[cols].astype(str)
            train_df, test_df = common.split_df(df, c, train)

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')

            training_set = train_df.values.tolist()
            model = DecisionTree()
            tree = model.learn(training_set, cols, c)
            print(tree)

            test_set = test_df.values.tolist()
            y_predict = model.predict(test_set)
            cc_percentage = model.score(test_set, y_predict) * 100

            summary = {}
            summary['Total Training Data'] = len(train_df)
            summary['Total Testing Data'] = len(test_df)
            summary['Total Number of Features in Dataset'] = len(var)
            summary['Model Accuracy %'] = round(cc_percentage, 2)
            summary['Features'] = str(var)
            summary_df = pd.DataFrame(summary.items(), columns=['Parameters', 'Value'])

            db.put('dtn.data_train', train_df)
            db.put('dtn.data_test', test_df)
            db.put('dtn.model_summary', summary)
            db.put('dtn.model_instance', model)
            #confusion_df = get_confusion_matrix(test_df, c, var, instanceOfLR)
        except Exception as e:
            traceback.print_exc()
            return common.error_msg("Exception during training model: " + str(e))

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Tree:'),
            html.H2(str(tree)),
            html.Br(),
            html.H2('Model Parameters & Summary:'),
            dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            #html.H2('Confusion Matrix (Precision & Recall):'),
            #dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            html.H2('Prediction/Classification:'),
            html.P('Features to be Predicted (comma separated): ' + ','.join(var), style = {'font-size': '16px'}),
            dbc.Input(id="dtn-prediction-data", placeholder=','.join(var), type="text"),
            html.Br(),
            dbc.Button("Predict", color="primary", id = 'dtn-predict'),
            html.Div([], id = "dtn-prediction")
            ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div
