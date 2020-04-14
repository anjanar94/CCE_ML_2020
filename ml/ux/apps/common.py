import dash_bootstrap_components as dbc

def navbar(page_name: str):
    nav = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_name)),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Linear Classification", href="/apps/linear-classification", style = {'font-size': '16px'}),
                    dbc.DropdownMenuItem("Non Linear Classification", href="/apps/non-linear-classification", style = {'font-size': '16px'}),
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
