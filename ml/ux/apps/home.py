import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils
from ml.framework.database import db

home_layout = [
    common.navbar("Home"),
    html.Br(),
    html.H3(
        children='A tool developed as part of IISc CCE Machine Learning Course, 2020',
        style={'textAlign': 'center'}),
    html.Hr(),
    html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([html.A('Drag and Drop or Select Files')],
        style = {'font-size': '16px'}),
        style={
            'width': '50%',
            'height': '50px',
            'lineHeight': '50px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True),
        html.Br(),
        html.Div([
            html.H2("Select a file from all the uploaded files:"),
            dcc.Dropdown(
                id = 'selected-file',
                options=[{'label':file, 'value':file} for file in FileUtils.files('raw')],
                value=None,
                multi=False,
                #style = {}
            ),
            html.Br(),
        ],
        style = {'margin': '10px', 'width': '50%'}),
        html.Div(id = "display-file")
]

layout = html.Div(home_layout)

@app.callback(
    Output("selected-file", "options"),
    [Input('upload-data', 'contents'),
    Input('upload-data', 'filename')]
)
def upload_data(contents, filename):
    """Upload Files and Regenerate the file list."""
    if contents:
        for i in range(len(filename)):
            FileUtils.upload(filename[i], contents[i])
    files = FileUtils.files('raw')
    if len(files) == 0:
        options=[{'label':'No files uploaded yet!', 'value':'None'}]
        return options
    else:
        options=[{'label':file, 'value':file} for file in files]
        return options

@app.callback(
    Output("display-file", "children"),
    [Input('selected-file', 'value')]
)
def display_data(value):
    """Displaying the head for the selected file."""
    db_value = db.get("file")
    if value is None and db_value is None:
        return ""
    elif value is None and not db_value is None:
        value = db_value
    div = None
    format = FileUtils.file_format(value)
    if format == 'csv' or format == 'txt':
        head = DataUtils.read_text_head('raw', value)
        table_col = [html.Col(style = {'width':"10%"}), html.Col(style = {'width':"90%"})]
        table_header = [html.Thead(html.Tr([html.Th("Row No"), html.Th("Data")]))]
        rows = []
        for i in range(len(head)):
            row = html.Tr([html.Td(i+1), html.Td(head[i])])
            rows.append(row)
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_col+ table_header + table_body, bordered=True,
        style = {'margin': '10px', 'font-size':'16px', 'padding': '20px'})
        div =  [common.selected_file(value), table]
    elif format == 'jpeg' or format == 'jpg' or format == 'gif':
        div =  [common.selected_file(value)]
    else:
        div = "Format Not Supported!!"
    db.put("file", value)
    return div
