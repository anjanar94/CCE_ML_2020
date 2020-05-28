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
    common.navbar("Neural Network"),
    html.Br(),
    html.Div([
        html.H2("Select a Trained Digit Recognition Neural Network")
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'nn-selected-neural-network',
        options = [{'label':'One Hidden Layer', 'value':'one'}, {'label':'Two Hidden Layer', 'value':'two'}],
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "cl-clear-db-do-nothing"),
    html.Div([],id = "linear-classification-selected-scatter-plot")
])
