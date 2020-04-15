import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db

layout = html.Div([
    common.navbar("Classification - Linearly Non-Separable"),
    html.Br(),
    html.Div([],id = "non-linear-classification-selected")
])

@app.callback(
    Output("non-linear-classification-selected", "children"),
    [Input('non-linear-classification', 'href')]
)
def selected_file(href):
    return common.selected_file(db.get("file"))
