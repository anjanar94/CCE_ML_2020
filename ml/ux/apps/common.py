import dash_bootstrap_components as dbc
import dash_html_components as html

def navbar(page_name: str):
    nav = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_name)),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Linear Classification", href="/apps/linear-classification", id = "linear-classification", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Non Linear Classification", href="/apps/non-linear-classification", id = "non-linear-classification", style = {'font-size': '16px'}),
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
        style = {'font-size': '16px'},
        brand_style = {'font-size': '16px'}
    )
    return nav

def selected_file(file: str):
    if file is None:
        return None
    return html.Div([
        html.H2(children = "Selected File: " + file,
            style = {'margin': '10px', 'font-size': '16px'}),
            html.Br()])
