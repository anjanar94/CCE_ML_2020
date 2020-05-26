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
    lr_instance = db.get('sgd.model_instance')
    n_var = summary['Total Number of Features in Dataset']
    if predict_data is None:
        return ("" , "")
    if len(predict_data.split(',')) != n_var:
        return (common.error_msg('Enter Valid Prediction Data!!'), "")
    try:
        feature_vector = get_predict_data_list(predict_data)
        feature_vector = np.array(feature_vector)
        prediction = lr_instance.predict(feature_vector)
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

    feature_vector = get_predict_data_list(predict_data)
    feature_vector.append('Predicted-'+prediction)
    df.loc[len(df)] = feature_vector

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

def get_confusion_matrix(df, c, var, model):
    classes = df[c].unique()
    d = {}
    for clazz in classes:
        d[clazz] = {'t_rel':0, 't_ret':0, 'rr':0}
    for index, row in df.iterrows():
        feature_vector = []
        for v in var:
            feature_vector.append(row[v])
        feature_vector = np.array(feature_vector)
        clazz = row[c]
        prediction = model.predict(feature_vector)
        d[clazz]['t_rel'] = d[clazz]['t_rel'] + 1
        d[prediction]['t_ret'] = d[prediction]['t_ret'] + 1
        if clazz == prediction:
            d[clazz]['rr'] = d[clazz]['rr'] + 1
    df = pd.DataFrame(columns=['Class', 'Total Retrieved Records', 'Total Relevant Records', 'Retrieved & Relevant', 'Precision', 'Recall'])
    i = 0
    for k, v in d.items():
        df.loc[i] = [k, v['t_ret'],v['t_rel'], v['rr'], round(v['rr']/v['t_ret'], 4), round(v['rr']/v['t_rel'], 4)]
        i = i+1
    return df
