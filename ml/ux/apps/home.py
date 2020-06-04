import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import common
from ml.framework.file_utils import FileUtils
from ml.framework.data_utils import DataUtils
from ml.framework.database import db

layout = html.Div([
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
                multi=False
            ),
            html.Br(),
        ],
        style = {'margin': '10px', 'width': '50%'}),
        html.Div([], id = "display-file"),
        html.Div([], id = "file-properties"),
        html.Div([], id = "file-separator-do-nothing"),
        html.Div([], id = "file-header-do-nothing")
])

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
    return common.get_options('raw')

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
    elif not value == db_value:
        db.reset()
    format = FileUtils.file_format(value)
    if format == 'csv' or format == 'txt':
        path = FileUtils.path('raw', value)
        head = DataUtils.read_text_head(path)
        table_col = [html.Col(style = {'width':"10%"}), html.Col(style = {'width':"90%"})]
        table_header = [html.Thead(html.Tr([html.Th("Row No"), html.Th("Data")]))]
        rows = []
        for i in range(len(head)):
            row = html.Tr([html.Td(i+1), html.Td(head[i])])
            rows.append(row)
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_col+ table_header + table_body, bordered=True, style = common.table_style)
        div =  [common.msg("Selected File: " + value),
                common.msg("Selected Format: " + format),
                table,
                html.Br(),
                csv_properties_div]
    elif format == 'jpeg' or format == 'jpg' or format == 'gif':
        div =  [common.msg("Selected File: " + value),
                common.msg("Selected Format: " + format)]
    else:
        div = "Format Not Supported!!"
    db.put("file", value)
    db.put("format", format)
    return div

csv_properties = dbc.Card([
    dbc.FormGroup([
        html.H2("Apply File Properties"),
        dbc.Label("Header"),
        dcc.Dropdown(
            id="file-header",
            options=[{'label':"True", 'value':1}, {"label": "False", "value": 0}],
            value=None,
            multi=False),
        dbc.Label("Separator"),
        dbc.Input(id="file-separator", placeholder="Separator", type="text", step=1),
        html.Br(),
        dbc.Button("Apply", color="primary", id = 'file-apply-properties'),
        ],
        style = {'padding': '10px'})
    ])

csv_properties_div = html.Div([
    dbc.Row([
        dbc.Col(csv_properties, md=4)
    ],
    align="center")
],
style = {'margin': '10px', 'font-size': '16px'})

@app.callback(
    Output("file-properties", "children"),
    [Input('file-apply-properties', 'n_clicks')]
)
def apply_file_properties(n):
    file = db.get("file")
    format = db.get("format")
    sep = db.get("file_separator")
    header = db.get("file_header")
    div = None
    if format is None:
        div = None
    elif (format == 'csv' or format == 'txt') and header is None:
        div= common.error_msg('Please Select Header!!')
    elif format == 'csv' or format == 'txt':
        if sep is None:
            sep = ','
            db.put("file_separator", sep)
        path = FileUtils.path('raw', file)
        df = DataUtils.read_csv(path, sep, header)
        ### Save Clean DataFrame ###
        path = FileUtils.path('clean', file.split('.')[0])
        df.to_csv(path, index=False)
        db.put("data", df)
        msg = "Following Properties Applied. Separator=" + sep + " Header="+ str(header)
        table = dbc.Table.from_dataframe(df.head(10).astype(str), striped=True, bordered=True, hover=True, style = common.table_style)
        div = [common.msg(msg), table]
    return div


@app.callback(
    Output('file-separator-do-nothing' , "children"),
    [Input('file-separator', 'value')]
)
def file_separator(value):
    if not value is None:
        db.put("file_separator", value)
    return None

@app.callback(
    Output('file-header-do-nothing' , "children"),
    [Input('file-header', 'value')]
)
def file_header_true(value):
    if value == 1:
        db.put("file_header", True)
    elif value == 0:
        db.put("file_header", False)
    return None
