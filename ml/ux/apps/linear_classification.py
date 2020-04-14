import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import common

layout = html.Div([
    common.navbar("Classification - Linearly Separable"),
    html.Br(),
])
