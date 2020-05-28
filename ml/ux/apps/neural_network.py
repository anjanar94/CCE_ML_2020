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

from ml.neural_net.digit_recog_1_layer import DigitNeuralNet1HiddenLayer
from ml.neural_net.digit_recog_2_layer import DigitNeuralNet2HiddenLayer
import base64

layout = html.Div([
    common.navbar("Neural Network"),
    html.Br(),
    html.Div([
        html.P("A Digit Recognition Neural Network is Trained on the MNIST Handwritten Digit Images of 28*28 Pixels."),
        html.H2("Select a Trained Digit Recognition Neural Network")
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'nn-select-neural-network',
        options = [{'label':'One Hidden Layer', 'value':'one'}, {'label':'Two Hidden Layer', 'value':'two'}],
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([],id = "nn-selected-neural-network")
])


@app.callback(
    Output("nn-selected-neural-network", "children"),
    [Input('nn-select-neural-network', 'value')]
)
def nn_select_neural_network(value):
    if value is None:
        return ""
    net = None
    if value == 'one':
        value = 'One Hidden Layer'
        net = DigitNeuralNet1HiddenLayer(784, 100, 10)
    elif value == 'two':
        value = 'Two Hidden Layer'
        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
    db.put('nn.net', net)
    net.load()
    params, confusion_matrix = net.parameters()
    params['Accuracy'] = round(params['Accuracy'], 2)

    summary_df = pd.DataFrame(params.items(), columns=['Parameters', 'Value'])

    distinct_count_df = pd.DataFrame(columns=['Class', 'Total Count', 'Training Count', 'Testing Count'])
    distinct_count_df.loc[0] = ['0',6903, 5923, 980]
    distinct_count_df.loc[1] = ['1',7877, 6742, 1135]
    distinct_count_df.loc[2] = ['2',6990, 5958, 1032]
    distinct_count_df.loc[3] = ['3',7141, 6131, 1010]
    distinct_count_df.loc[4] = ['4',6824, 5842, 982]
    distinct_count_df.loc[5] = ['5',6313, 5421, 892]
    distinct_count_df.loc[6] = ['6',6876, 5918, 958]
    distinct_count_df.loc[7] = ['7',7293, 6265, 1028]
    distinct_count_df.loc[8] = ['8',6825, 5851, 974]
    distinct_count_df.loc[9] = ['9',6958, 5949, 1009]
    distinct_count_df.loc[10] = ['Gross Total = ',70000, 60000, 10000]

    confusion_df = pd.DataFrame(columns=['Class', 'Total Retrieved Records', 'Total Relevant Records', 'Retrieved & Relevant', 'Precision', 'Recall'])
    i = 0
    for k, v in confusion_matrix.items():
        confusion_df.loc[i] = [k, v['t_ret'],v['t_rel'], v['rr'], round(v['rr']/v['t_ret'], 4), round(v['rr']/v['t_rel'], 4)]
        i = i+1

    div = html.Div([
        html.H2(value + ' Digit Recognition Neural Network'),
        html.Br(),
        html.H2('Image Recognition/Classification:'),
        html.Div([
            dcc.Dropdown(
            id = 'nn-select-image',
            options=[{'label':file, 'value':file} for file in FileUtils.files('images')],
            value = None,
            multi = False
        )],style = {'width': '50%'}),
        html.Div([],id = "nn-selected-image"),
        html.Br(),
        html.Hr(),
        html.H2('Class Grouping in Data:'),
        dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
        html.H2('Model Parameters & Summary:'),
        dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
        html.H2('Confusion Matrix (Precision & Recall):'),
        dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br()
        ],style = {'margin': '10px'})
    return div

@app.callback(
    Output("nn-selected-image", "children"),
    [Input('nn-select-image', 'value')]
)
def nn_selected_image(value):
    if value is None:
        return ""
    path = FileUtils.path('images', value)
    encoded_image = base64.b64encode(open(path, 'rb').read())

    net = db.get('nn.net')
    clazz = net.predict(path)

    div = html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
            style={'height' : '5%',
                'width' : '5%',
                'float' : 'left',
                'position' : 'relative',
                'padding-top' : 0,
                'padding-right' : 0
                }),
        html.H2('Prediction: ' + str(clazz))
        ], style = {'margin': '50px'})
    return div
