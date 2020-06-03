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

from ml.stochastic_neural_net import ann_training
from ml.stochastic_neural_net import ann_testing
from ml.stochastic_neural_net import ann_predict

layout = html.Div([
    common.navbar("Stochastic Gradient Descent"),
    html.Br(),
    html.Div([
        html.H2("Load and Select a file from all the cleaned files:"),
        dbc.Button("Load Cleaned File", color="primary", id = 'sgd-load-cleaned-files', className="mr-2", style={'display': 'inline-block'}),
        dbc.Button("Clear", color="secondary", id = 'sgd-clear-db', className="mr-2", style={'display': 'inline-block'})
    ],style = {'margin': '10px'}),
    html.Div([
    dcc.Dropdown(
        id = 'sgd-selected-cleaned-file',
        options = common.get_options('clean'),
        value = None,
        multi = False
    )],
    style = {'margin': '10px', 'width': '50%'}),
    html.Div([], id = "sgd-clear-db-do-nothing"),
    html.Div([],id = "sgd-selected-scatter-plot")
])

@app.callback(
    Output("sgd-selected-cleaned-file", "options"),
    [Input('sgd-load-cleaned-files', 'n_clicks')]
)
def selected_file(n_clicks):
    return common.get_options('clean')

@app.callback(
    Output("sgd-clear-db-do-nothing", "options"),
    [Input('sgd-clear-db', 'n_clicks')]
)
def selected_file(n_clicks):
    return db.clear('sgd.')

@app.callback(
    Output("sgd-selected-scatter-plot", "children"),
    [Input('sgd-selected-cleaned-file', 'value')]
)
def sgd_display_selected_file_scatter_plot(value):
    db_value = db.get("sgd.file")
    if value is None and db_value is None:
        return common.msg("Please select a cleaned file to proceed!!")
    elif value is None and not db_value is None:
        value = db_value

    db.put("sgd.file", value)
    file = value
    path = FileUtils.path('clean', file)
    df = DataUtils.read_csv(path)
    db.put("sgd.data", df)

    stats = df.describe().head(3).round(5)
    stats.insert(loc=0, column='Statistics', value=['Count', 'Mean', 'Standard Deviation'])

    div = html.Div([
        common.msg("Selected cleaned file: "+ file),
        dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True, style = common.table_style),
        html.Div([html.H3("Data Statistics")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True, style = common.table_style),
        html.Br(),
        html.Div([html.H2("Scatter Plot")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Class"),
                dcc.Dropdown(
                    id = 'sgd-class',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'sgd-x-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'sgd-y-axis',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'sgd-scatter-plot-button'),
                html.Div([], id = "sgd-class-do-nothing"),
                html.Div([], id = "sgd-x-axis-do-nothing"),
                html.Div([], id = "sgd-y-axis-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="sgd-scatter-plot")
        ]),
        html.Br(),
        get_sgd_model_properties_div(df),
        dcc.Loading(id="sgd-model-training",
            children=[html.Div([], id = "sgd-trained-model", style = {'margin': '10px'})],
            type="default"),
        #html.Div([], id = "sgd-trained-model", style = {'margin': '10px'}),
    ])

    return div

@app.callback(
    Output("sgd-scatter-plot", "children"),
    [Input('sgd-scatter-plot-button', 'n_clicks')]
)
def sgd_scatter_plot(n):
    df = db.get("sgd.data")
    clazz_col = db.get("sgd.class")
    x_col = db.get("sgd.x_axis")
    y_col = db.get("sgd.y_axis")
    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='sgd-x-vs-y',
        figure={
            'data': [
                go.Scatter(
                    x=df[df[clazz_col] == clazz][x_col],
                    y=df[df[clazz_col] == clazz][y_col],
                    text=df[df[clazz_col] == clazz][clazz_col],
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=clazz
                ) for clazz in df[clazz_col].unique()
            ],
            'layout': dict(
                #title='Scatter Plot',
                xaxis={'title': x_col},
                yaxis={'title': y_col},
                margin={'l': 40, 'b': 40},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
    return graph

@app.callback(
    Output('sgd-class-do-nothing' , "children"),
    [Input('sgd-class', 'value')]
)
def sgd_class(value):
    if not value is None:
        db.put("sgd.class", value)
    return None

@app.callback(
    Output('sgd-x-axis-do-nothing' , "children"),
    [Input('sgd-x-axis', 'value')]
)
def sgd_x_axis(value):
    if not value is None:
        db.put("sgd.x_axis", value)
    return None

@app.callback(
    Output('sgd-y-axis-do-nothing' , "children"),
    [Input('sgd-y-axis', 'value')]
)
def sgd_y_axis(value):
    if not value is None:
        db.put("sgd.y_axis", value)
    return None

def get_sgd_model_properties_div(df):
    sgd_model_properties = dbc.Card([
        dbc.FormGroup([
            html.H2("Train Neural Network - Stochastic Gradient Descent"),
            dbc.Label("Class"),
            dcc.Dropdown(
                id="sgd-model-class",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=False),
            dbc.Label("Features"),
            dcc.Dropdown(
                id="sgd-model-variables",
                options=[{'label':col, 'value':col} for col in [*df]],
                value=None,
                multi=True),
            dbc.Label("Train Data %"),
            dbc.Input(id="sgd-train-data", placeholder="70,75,80,85,90", type="number"),
            dbc.Label("No of Neuron in each Hidden Layer"),
            dbc.Input(id="sgd-no-of-neuron", placeholder="5,10,15", type="number"),
            dbc.Label("Learning Rate"),
            dbc.Input(id="sgd-learning-rate", placeholder="0.005, 0.01, 0.05", type="number"),
            dbc.Label("Epoch"),
            dbc.Input(id="sgd-epoch", placeholder="100,200,300,400,500..", type="number"),
            html.Br(),
            dbc.Button("Train", color="primary", id = 'sgd-train-model'),

            html.Div([], id = "sgd-model-class-do-nothing"),
            html.Div([], id = "sgd-model-variables-do-nothing"),
            html.Div([], id = "sgd-train-data-do-nothing"),
            html.Div([], id = "sgd-no-of-neuron-do-nothing"),
            html.Div([], id = "sgd-learning-rate-do-nothing"),
            html.Div([], id = "sgd-epoch-do-nothing"),
            html.Div([], id = "sgd-prediction-data-do-nothing")
            ],
            style = {'padding': '10px'})
        ])

    sgd_model_properties_div = html.Div([
        dbc.Row([
            dbc.Col(sgd_model_properties, md=6)
        ],
        align="center")
    ],
    style = {'margin': '10px', 'font-size': '16px'})

    return sgd_model_properties_div

@app.callback(
    Output('sgd-model-class-do-nothing' , "children"),
    [Input('sgd-model-class', 'value')]
)
def sgd_model_class(value):
    if not value is None:
        db.put("sgd.model_class", value)
    return None

@app.callback(
    Output('sgd-model-variables-do-nothing' , "children"),
    [Input('sgd-model-variables', 'value')]
)
def sgd_model_variables(value):
    if not value is None:
        db.put("sgd.model_variables", value)
    return None

@app.callback(
    Output('sgd-train-data-do-nothing' , "children"),
    [Input('sgd-train-data', 'value')]
)
def sgd_model_train(value):
    if not value is None:
        db.put("sgd.model_train", value)
    return None

@app.callback(
    Output('sgd-no-of-neuron-do-nothing' , "children"),
    [Input('sgd-no-of-neuron', 'value')]
)
def sgd_model_epoch(value):
    if not value is None:
        db.put("sgd.no_of_neuron", value)
    return None

@app.callback(
    Output('sgd-learning-rate-do-nothing' , "children"),
    [Input('sgd-learning-rate', 'value')]
)
def sgd_model_lr(value):
    if not value is None:
        db.put("sgd.model_lr", value)
    return None

@app.callback(
    Output('sgd-epoch-do-nothing' , "children"),
    [Input('sgd-epoch', 'value')]
)
def sgd_model_epoch(value):
    if not value is None:
        db.put("sgd.model_epoch", value)
    return None


@app.callback(
    Output('sgd-trained-model' , "children"),
    [Input('sgd-train-model', 'n_clicks')]
)
def sgd_model_train(n_clicks):
    c = db.get('sgd.model_class')
    var = db.get('sgd.model_variables')
    train = db.get('sgd.model_train')
    #test = db.get('sgd.model_test')
    lr = db.get('sgd.model_lr')
    epoch = db.get('sgd.model_epoch')
    #no_of_hidden_layer = db.get('sgd.no_of_hidden_layer')
    no_of_neuron = db.get('sgd.no_of_neuron')
    if c is None and var is None and train is None and lr is None and epoch is None:
        div = ""
    elif train is None or train < 0 or train > 100:
        div = common.error_msg('Training % should be between 0 - 100 !!')
    elif (not c is None) and (not var is None) and (not train is None) and (not lr is None) and (not epoch is None):
        #parameters = "Training Data = " + str(train) + " % Testing Data = " + str(100 - train) + " % Learning rate = " + str(lr) + " Epoch = " + str(epoch)

        try:
            cols = [] + var
            cols.append(c)
            df = db.get('sgd.data')
            df = df[cols]
            ## Make DataFrame compatible for SGD API ##
            df, quantized_classes, reverse_quantized_classes = quantized_class(df, c)

            msk = np.random.rand(len(df)) < (train / 100)
            train_df = df[msk]
            test_df = df[~msk]

            distinct_count_df_total = get_distinct_count_df(df, c, 'Total Count')
            distinct_count_df_train = get_distinct_count_df(train_df, c, 'Training Count')
            distinct_count_df_test = get_distinct_count_df(test_df, c, 'Testing Count')

            distinct_count_df = distinct_count_df_total.join(distinct_count_df_train.set_index('Class'), on='Class')
            distinct_count_df = distinct_count_df.join(distinct_count_df_test.set_index('Class'), on='Class')
            distinct_count_df['Class'] = distinct_count_df['Class'].map(reverse_quantized_classes)

            ycap, loss_dict, cc_percentage, wc_percentage, model, yu = ann_training(train_df[var], train_df[c], no_of_neuron, lr, epoch)
            ycap, cc_percentage, wc_percentage = ann_testing(test_df[var], test_df[c], model, yu)

            summary = {}
            summary['Total Training Data'] = len(train_df)
            summary['Total Testing Data'] = len(test_df)
            summary['Total Number of Features in Dataset'] = len(var)
            summary['No of Hidden Layer'] = 1
            summary['No of Neuron in each Hidden Layer'] = no_of_neuron
            summary['Activation Function'] = 'Sigmoid'
            summary['Learning rate'] = lr
            summary['Epochs'] = epoch
            summary['Model Accuracy'] = round(cc_percentage, 2)
            summary['Features'] = str(var)
            summary_df = pd.DataFrame(summary.items(), columns=['Parameters', 'Value'])

            db.put('sgd.data_train', train_df)
            db.put('sgd.data_test', test_df)
            db.put('sgd.quantized_classes', quantized_classes)
            db.put('sgd.reverse_quantized_classes', reverse_quantized_classes)
            db.put('sgd.model', model)
            db.put('sgd.model_yu', yu)
            db.put('sgd.summary', summary)

            confusion_df = get_confusion_matrix(test_df, c, var, model, yu, reverse_quantized_classes)
        except Exception as e:
            return common.error_msg("Exception during training model: " + str(e))

        trace = go.Scatter(x = loss_dict['Epoch_no'], y = loss_dict['Loss'],
                        line = dict(width = 2, color = 'rgb(106, 181, 135)'))
        convergence_title = go.Layout(title = 'Convergence Plot', hovermode = 'closest', xaxis={'title': 'Epoch'}, yaxis={'title': 'Loss Function'})
        convergence_fig = go.Figure(data = [trace], layout = convergence_title)

        div = html.Div([
            html.H2('Class Grouping in Data:'),
            dbc.Table.from_dataframe(distinct_count_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.H2('Model Parameters & Summary:'),
            dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            dcc.Graph(id='sgd-convergence-plot', figure=convergence_fig),
            html.H2('Confusion Matrix (Precision & Recall):'),
            dbc.Table.from_dataframe(confusion_df, striped=True, bordered=True, hover=True, style = common.table_style),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H2('Prediction/Classification:'),
            html.P('Features to be Predicted (comma separated): ' + ','.join(var), style = {'font-size': '16px'}),
            dbc.Input(id="sgd-prediction-data", placeholder=','.join(var), type="text"),
            html.Br(),
            dbc.Button("Predict", color="primary", id = 'sgd-predict'),
            html.Div([], id = "sgd-prediction"),
            html.Div([],id = "sgd-predicted-scatter-plot")
            ])
    else:
        div = common.error_msg('Select Proper Model Parameters!!')
    return div

@app.callback(
    Output('sgd-prediction-data-do-nothing' , "children"),
    [Input('sgd-prediction-data', 'value')]
)
def sgd_model_prediction_data(value):
    if not value is None:
        db.put("sgd.model_prediction_data", value)
    return None

@app.callback(
    [Output('sgd-prediction' , "children"),
    Output('sgd-predicted-scatter-plot' , "children")],
    [Input('sgd-predict', 'n_clicks')]
)
def sgd_model_predict(n_clicks):
    predict_data = db.get("sgd.model_prediction_data")
    summary = db.get('sgd.model_summary')
    model = db.get('sgd.model')
    yu = db.get('sgd.model_yu')
    n_var = summary['Total Number of Features in Dataset']
    var = db.get('sgd.model_variables')
    test_df = db.get('sgd.data_test')
    if predict_data is None:
        return ("" , "")
    if len(predict_data.split(',')) != n_var:
        return (common.error_msg('Enter Valid Prediction Data!!'), "")
    try:
        feature_vector = get_predict_data_list(predict_data)
        #feature_vector = np.array(feature_vector)
        l = len(test_df)
        test_df.loc[l] = feature_vector
        feature_vector = test_df[var].iloc[l:l+1]
        print(l)
        prediction = ann_predict(feature_vector, model, yu)
        db.put('sgd.prediction', prediction)
    except Exception as e:
        return (common.error_msg("Exception during prediction: " + str(e)), "")
    df = db.get('sgd.data_train')
    df = df.iloc[:, :-1]
    div = html.Div([
        html.Div([html.H2("Predicted & Testing Data Scatter Plot")], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select X Axis"),
                dcc.Dropdown(
                    id = 'sgd-x-axis-predict',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Label("Select Y Axis"),
                dcc.Dropdown(
                    id = 'sgd-y-axis-predict',
                    options=[{'label':col, 'value':col} for col in [*df]],
                    value=None,
                    multi=False
                ),
                html.Br(),
                dbc.Button("Plot", color="primary", id = 'sgd-predict-scatter-plot-button'),
                html.Div([], id = "sgd-x-axis-predict-do-nothing"),
                html.Div([], id = "sgd-y-axis-predict-do-nothing")
            ], md=2,
            style = {'margin': '10px', 'font-size': '16px'}),
            dbc.Col([], md=9, id="sgd-scatter-plot-predict")
        ]),

    ])
    return (common.success_msg('Predicted/Classified Class = ' + prediction), div)

@app.callback(
    Output('sgd-x-axis-predict-do-nothing' , "children"),
    [Input('sgd-x-axis-predict', 'value')]
)
def sgd_x_axis(value):
    if not value is None:
        db.put("sgd.x_axis_predict", value)
    return None

@app.callback(
    Output('sgd-y-axis-predict-do-nothing' , "children"),
    [Input('sgd-y-axis-predict', 'value')]
)
def sgd_y_axis(value):
    if not value is None:
        db.put("sgd.y_axis_predict", value)
    return None

@app.callback(
    Output("sgd-scatter-plot-predict", "children"),
    [Input('sgd-predict-scatter-plot-button', 'n_clicks')]
)
def sgd_scatter_plot(n):
    df = db.get("sgd.data_test")
    clazz_col = db.get('sgd.model_class')
    x_col = db.get("sgd.x_axis_predict")
    y_col = db.get("sgd.y_axis_predict")
    predict_data = db.get("sgd.model_prediction_data")
    prediction = db.get('sgd.prediction')
    reverse_quantized_classes = db.get('sgd.reverse_quantized_classes')

    feature_vector = get_predict_data_list(predict_data)
    feature_vector.append('Predicted-'+prediction)
    df.loc[len(df)] = feature_vector
    df[c] = df[c].map(reverse_quantized_classes)

    if clazz_col is None or x_col is None or y_col is None:
        return None
    graph = dcc.Graph(
        id='sgd-x-vs-y-predict',
        figure={
            'data': [
                go.Scatter(
                    x=df[df[clazz_col] == clazz][x_col],
                    y=df[df[clazz_col] == clazz][y_col],
                    text=df[df[clazz_col] == clazz][clazz_col],
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=clazz
                ) for clazz in df[clazz_col].unique()
            ],
            'layout': dict(
                #title='Scatter Plot',
                xaxis={'title': x_col},
                yaxis={'title': y_col},
                margin={'l': 40, 'b': 40},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
    return graph

def get_predict_data_list(predict_data: str) -> []:
    predict_data = predict_data.split(',')
    feature_vector = []
    for d in predict_data:
        feature_vector.append(float(d))
    return feature_vector

def quantized_class(df, c):
    classes = df[c].unique()
    quantized_classes = {}
    reverse_quantized_classes = {}
    i = 0
    for clazz in classes:
        quantized_classes[clazz] = i
        reverse_quantized_classes[i] = clazz
        #df.replace(to_replace = clazz, value = i)
        i = i + 1
    df[c] = df[c].map(quantized_classes)
    return df, quantized_classes, reverse_quantized_classes

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

def get_confusion_matrix(df, c, var, model, yu, reverse_quantized_classes):
    classes = df[c].unique()
    i = 0
    d = {}
    for clazz in classes:
        clazz = str(int(clazz))
        d[clazz] = {'t_rel':0, 't_ret':0, 'rr':0}
    for index, row in df.iterrows():
        clazz = str(int(row[c]))
        feature_vector = df[var].iloc[i:i+1]
        i = i + 1
        prediction = ann_predict(feature_vector, model, yu)
        prediction = str(int(prediction))
        d[clazz]['t_rel'] = d[clazz]['t_rel'] + 1
        d[prediction]['t_ret'] = d[prediction]['t_ret'] + 1
        if clazz == prediction:
            d[clazz]['rr'] = d[clazz]['rr'] + 1
    df = pd.DataFrame(columns=['Class', 'Total Retrieved Records', 'Total Relevant Records', 'Retrieved & Relevant', 'Precision', 'Recall'])
    i = 0
    for k, v in d.items():
        key = reverse_quantized_classes[int(k)]
        df.loc[i] = [key, v['t_ret'],v['t_rel'], v['rr'], round(v['rr']/v['t_ret'], 4), round(v['rr']/v['t_rel'], 4)]
        i = i+1
    return df
