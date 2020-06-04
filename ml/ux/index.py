import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from ml.ux.app import app
from ml.ux.apps import home, linear_classification, non_linear_classification, neural_network, stochastic_gradient_descent, decision_trees

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/apps/linear-classification':
        return linear_classification.layout
    elif pathname == '/apps/non-linear-classification':
        return non_linear_classification.layout
    elif pathname == '/apps/neural-network':
        return neural_network.layout
    elif pathname == '/apps/stochastic-gradient-descent':
        return stochastic_gradient_descent.layout
    elif pathname == '/apps/decision-trees':
        return decision_trees.layout
    else:
        return '404'


app.run_server(debug=True)
