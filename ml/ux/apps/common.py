import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.framework.file_utils import FileUtils

def navbar(page_name: str):
    nav = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_name)),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Linear Classification", href="/apps/linear-classification", id = "linear-classification", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Non Linear Classification", href="/apps/non-linear-classification", id = "non-linear-classification", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Neural Network", href="/apps/neural-network", id = "neural-network", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Stochastic Gradient Descent", href="/apps/stochastic-gradient-descent", id = "stochastic-gradient-descent", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Decision Trees", href="/apps/decision-trees", id = "decision-trees", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Principle Component Analysis", href="/apps/pca", id = "pca", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("K Nearest Neighbors", href="/apps/knn", id = "knn", style = {'font-size': '16px'})
                ],
                nav=True,
                in_navbar=True,
                label="Projects",
                style = {'padding-left': 20, 'padding-right': 50},
            ),
        ],
        brand="Machine Learning Tool",
        brand_href="/",
        color="#25383C",
        dark=True,
        fixed='top',
        style = {'font-size': '16px'},
        brand_style = {'font-size': '16px'}
    )
    return nav

table_style = {'margin': '10px', 'font-size':'16px', 'padding': '20px'}

def msg(msg: str):
    if msg is None:
        return None
    return html.Div([
        html.H2(children = msg,
            style = {'margin': '10px', 'font-size': '16px'}),
            html.Br()])

def error_msg(msg: str):
    if msg is None:
        return None
    return html.Div([
        html.H2(children = msg,
            style = {'margin': '10px', 'font-size': '16px', 'color': 'red'}),
            html.Br()])

def success_msg(msg: str):
    if msg is None:
        return None
    return html.Div([
        html.H2(children = msg,
            style = {'margin': '10px', 'font-size': '16px', 'color': 'green'}),
            html.Br()])

def get_options(dir: str):
    files = FileUtils.files(dir)
    if len(files) == 0:
        return [{'label':'No files yet!', 'value':'None'}]
    else:
        return [{'label':file, 'value':file} for file in files]

def split_df(df, c: str, train: float):
    train_frames = []
    test_frames = []
    for clazz in df[c].unique():
        clazz_df = df[df[c] == clazz]
        train_df, test_df = train_test_split(clazz_df, test_size=(100-train)/100)
        train_frames.append(train_df)
        test_frames.append(test_df)
    train_df = pd.concat(train_frames)
    test_df = pd.concat(test_frames)
    return train_df, test_df
