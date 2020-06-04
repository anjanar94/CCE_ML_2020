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

from ml.decision_trees import DecisionTree

layout = html.Div([
    common.navbar("Decision Trees"),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'dt-load-cleaned-files', className="mr-2", style={'display': 'inline-block'}),
        dbc.Button("Clear", color="secondary", id = 'dt-clear-db', className="mr-2", style={'display': 'inline-block'})
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'dt-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "dt-clear-db-do-nothing"),
    html.Div([],id = "decision-trees-selected-div")
])

@app.callback(
    Output("dt-selected-cleaned-file", "options"),
    [Input('dt-load-cleaned-files', 'n_clicks')]
)
def dt_selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("dt-clear-db-do-nothing", "options"),
    [Input('dt-clear-db', 'n_clicks')]
)
def dt_selected_file(n_clicks):
    return db.clear('dt.')

@app.callback(
    Output("decision-trees-selected-div", "children"),
    [Input('dt-selected-cleaned-file', 'value')]
)
def dt_display_selected_file_scatter_plot(value):
    db_value = db.get("dt.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("dt.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("dt.data", df)

    div = html.Div([
        common.msg("Selected cleaned file: "+ file),
        dbc.Table.from_dataframe(df.head(10).astype(str), striped=True, bordered=True, hover=True, style = common.table_style),
        #html.Div([html.H3("Data Statistics")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        #dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        get_dt_model_properties_div(df),
        html.Div([], id = "dt-trained-model", style = {'margin': '10px'}),
    ])

    return div

def get_dt_model_properties_div(df):
    dt_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train Decision Tree Model"),
            dbc.Label("Class"),
            dcc.Dropdown(
                id="dt-model-class",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=False),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="dt-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            dbc.Label("Train Data %"),
            dbc.Input(id="dt-train-data", placeholder="70,75,80,85,90", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'dt-train-model'),

            html.Div([], id = "dt-model-class-do-nothing"),
            html.Div([], id = "dt-model-variables-do-nothing"),
            html.Div([], id = "dt-train-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    dt_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(dt_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return dt_model_properties_div

@app.callback(
    Output('dt-model-class-do-nothing' , "children"),
    [Input('dt-model-class', 'value')]
)
def dt_model_class(value):
    if not value is None:
        db.put("dt.model_class", value)
    return None

@app.callback(
    Output('dt-model-variables-do-nothing' , "children"),
    [Input('dt-model-variables', 'value')]
)
def dt_model_variables(value):
    if not value is None:
        db.put("dt.model_variables", value)
    return None

@app.callback(
    Output('dt-train-data-do-nothing' , "children"),
    [Input('dt-train-data', 'value')]
)
def dt_model_train(value):
    if not value is None:
        db.put("dt.model_train", value)
    return None

@app.callback(
    Output('dt-trained-model' , "children"),
    [Input('dt-train-model', 'n_clicks')]
)
def dt_model_train(n_clicks):
    c = db.get('dt.model_class')
    var = db.get('dt.model_variables')
    train = db.get('dt.model_train')
    if c is None and var is None and train is None:
        div = ""
    elif train is None or train < 0 or train > 100:
        div = common.error_msg('Training % should be between 0 - 100 !!')
    elif (not c is None) and (not var is None) and (not train is None):

        try:
            cols = [] + var
            cols.append(c)
            df = db.get('dt.data')
            df = df[cols].astype(str)
            train_df, test_df = train_test_split(df, test_size=(100-train)/100)

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')

            training_set = train_df.values.tolist()
            model = DecisionTree()
            tree = model.learn(training_set, cols, c)
            print(tree)

            #db.put('cl.data_train', train_df)
            #db.put('cl.data_test', test_df)
            #db.put('cl.model_summary', summary)
            #db.put('cl.model_instance', instanceOfLR)
            #confusion_df = get_confusion_matrix(test_df, c, var, instanceOfLR)
        except Exception as e:
            traceback.print_exc()
            return common.error_msg("Exception during training model: " + str(e))

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Tree:'),
            html.H2(str(tree)),
            html.H2('Model Parameters & Summary:'),
            #dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            #html.H2('Confusion Matrix (Precision & Recall):'),
            #dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            html.Br()
            ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div

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
