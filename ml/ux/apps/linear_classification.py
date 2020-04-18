import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pandas as pd

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.database import db
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils

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
    file = db.get("file")
    format = db.get("format")
    df = db.get("data")
    div = None
    if file is None:
        div =  ""
    elif df is None:
        div  = [common.msg("Selected File: " + file + " Selected Format: " + format), common.error_msg("Please apply file properties!!")]
    else:
        msg = "Selected File: " + file + " Selected Format: " + format +" Following Properties Applied. Separator=" + sep + " Header="+ str(header)
        div = [common.msg(msg)]
    return div
