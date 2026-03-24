import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import dash_table
import plotly.express as px
import plotly.graph_objects as go

from .hedging import compute_hedging_strategy

DISPLAY = {
    "Brent": "Brent",
    "WTI": "WTI",
    "GO": "Gas Oil",
    "RBOB": "RBOB",
    "HO": "Heating Oil",
}

TICKERS = ["Brent", "WTI", "GO", "RBOB", "HO"]


def build_r2_heatmap(r2_mat):
    # Create the R2 heatmap figure
    return px.imshow(
        r2_mat,
        text_auto=".3f",
        aspect="auto",
        title="R2 Heatmap",
    )

def build_inventory_table(initial_records):
    # Initialise inventory table
    inventory_table = dash_table.DataTable(
        id='inventory-table',
        columns=[
            {"name": "Product", "id": "Product"},
            {"name": "Quantity", "id": "Quantity"},
        ],
        data=initial_records,
    )
    return inventory_table

def build_trade_entry_panel():
    # Enter trade: Dropdown + input + button
    return html.Div(
        children=[
            html.Label("Enter your trade:"),
            dcc.Dropdown(
                id="commodity-dropdown",
                options=[{"label": DISPLAY[t], "value": t} for t in TICKERS],
                placeholder="Select a Commodity",
            ),
            dcc.Input(
                id="quantity-input",
                type="number",
                placeholder="Quantity",
            ),
            html.Button("Submit Trade", id="submit-trade"),
        ],
        style={"display": "grid", "gap": "8px", "maxWidth": "360px"},
    )

def build_dialogs():
    return html.Div(
        children=[
            dcc.ConfirmDialog(
                id="success-dialog",
                message="Trade Submitted Successfully!",
                displayed=False,
            ),
            dcc.ConfirmDialog(
                id="fail-dialog",
                message="Invalid Input",
                displayed=False,
            ),
        ]
    )

def build_hedge_table():
    # Hedge suggestion table
    columns = [
        {"name": "Hedge Asset", "id": "hedge_asset"},
        {"name": "Action", "id": "action"},
        {"name": "Hedge Qty", "id": "hedge_qty"},
        {"name": "Beta", "id": "beta"},
        {"name": "R2", "id": "r2"},
        {"name": "var_before", "id": "var_before"},
        {"name": "var_after", "id": "var_after"},
        {"name": "var_change", "id": "var_change"},
        {"name": "cost", "id": "cost"},
        {"name": "score", "id": "score"},
    ]

    return dash_table.DataTable(
        id="hedge-table",
        columns=columns,
        data=[],
        style_cell={"padding": "6px"},
    )

def build_hedge_summary():
    # Text for hedge suggestion
    return html.Div(
        id="hedge-summary",
        children="",
        style={
            "padding": "10px",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "marginBottom": "10px",
        },
    )

def init_regression_plot():
    # Initialise the regression plot
    return go.Figure().update_layout(
        title="Click a heatmap cell to show regression"
    )


def create_app(r2_mat, results_df, returns):

    app = Dash(__name__)

    r2_fig = build_r2_heatmap(r2_mat)

    inventory_df = pd.DataFrame({
        "Product": TICKERS,
        "Quantity": [0]*len(TICKERS),
    })
    initial_records = inventory_df.to_dict("records")

    inventory_table = build_inventory_table(initial_records)

    app.layout = html.Div(
        children=[
            html.H1("Commodity Hedging Strategy"),

            html.H2("Trade Entry"),
            build_trade_entry_panel(),

            html.H2("Your Inventory"),
            inventory_table,

            build_dialogs(),

            html.H2("Hedge Suggestion"),
            build_hedge_summary(),
            build_hedge_table(),

            html.H2("Hedging Heatmap"),
            dcc.Graph(id="r2-heatmap", figure=r2_fig),

            html.H2("Regression Plot"),
            dcc.Graph(id="regression-plot", figure=init_regression_plot()),
        ],
        style={"padding": "16px", "maxWidth": "1100px"},
    )
        

    # Hedge Callback
    @app.callback(
        Output('inventory-table', 'data'),
        Output('hedge-summary', 'children'),
        Output('hedge-table', 'data'),
        Output('commodity-dropdown', 'value'),
        Output('quantity-input', 'value'),
        Output('success-dialog', 'displayed'),
        Output('fail-dialog', 'displayed'),

        Input('submit-trade', 'n_clicks'),
        State('commodity-dropdown', 'value'),
        State('quantity-input', 'value'),

        prevent_initial_call=True,
    )
    def update_inventory_hedge(n_clicks, exposure_asset, exposure_qty):

        # Check for commodity and quatity input
        if exposure_asset is None or exposure_qty is None or exposure_qty == 0:
            return inventory_df.to_dict('records'), "Invalid input!", [], None, None, False, True
        
        # update inventory
        inventory_df.loc[inventory_df['Product']==exposure_asset, 'Quantity'] += exposure_qty

        # compute hedge table
        rows, hedge_text = compute_hedging_strategy(results_df, returns, inventory_df, exposure_asset, exposure_qty)

        return inventory_df.to_dict("records"), hedge_text, rows, None, None, True, False


    # Callback for Heatmap to show Regression
    @app.callback(
        Output("regression-plot", "figure"),
        Input("r2-heatmap", "clickData"),
        prevent_initial_call = True,
    )
    def show_regression_plot(clickData):
        # clickData is a JSON (Python dict)
        
        pt = clickData["points"][0]
        x_col = pt["x"]
        y_col = pt["y"]

        x_product = DISPLAY[x_col]
        y_product = DISPLAY[y_col]

        if x_col == y_col:
            return go.Figure().update_layout(title=f"Please click an off-diagonal cell")
    
        row = results_df[(results_df["x"]==x_col) & (results_df["y"]==y_col)].iloc[0]
        alpha = row["alpha"]
        beta = row["beta"]
        r2 = row["r2"]

        x = returns[x_col].to_numpy()
        y = returns[y_col].to_numpy()
        y_hat = alpha + beta * x

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y_hat = y_hat[order]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Returns"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                mode="lines",
                name="Fit Line"
            )
        )
        fig.update_layout(
            title=f"Linear Regression: {y_product} vs {x_product} "
                  f"(beta={beta:.5g}, intercep={alpha:.5g}, R^2={r2:.5g})",
            xaxis_title=f"{x_product} returns",
            yaxis_title=f"{y_product} returns",
            height=450,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    return app
