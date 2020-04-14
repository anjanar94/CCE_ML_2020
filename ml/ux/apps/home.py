import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import common


home_layout = [
    common.navbar("Home"),
    html.Br(),
    html.H3(
        children='A tool developed as part of IISc CCE Machine Learning Course, 2020',
        style={'textAlign': 'center'}),
    html.Hr()

]

layout = html.Div(home_layout)
