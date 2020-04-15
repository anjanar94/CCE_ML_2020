import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.app import db
from ml.ux.apps import common

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
    return common.selected_file(db["file"])
