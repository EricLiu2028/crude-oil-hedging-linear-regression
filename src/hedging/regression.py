from itertools import permutations
import pandas as pd
from sklearn.linear_model import LinearRegression

def build_merged_prices(brent_df, wti_df, ho_df, go_df, rb_df):
    # Extract close price for each product
    brent_price = brent_df[["Date", "Close"]].rename(columns={"Close": "Brent"})
    wti_price   = wti_df[["Date", "Close"]].rename(columns={"Close": "WTI"})
    ho_price    = ho_df[["Date", "Close"]].rename(columns={"Close": "HO"})
    go_price    = go_df[["Date", "Close"]].rename(columns={"Close": "GO"})
    rb_price    = rb_df[["Date", "Close"]].rename(columns={"Close": "RBOB"})

    # Merge the data
    merged = (
        brent_price.merge(wti_price, on="Date")
                   .merge(ho_price, on="Date")
                   .merge(go_price, on="Date")
                   .merge(rb_price, on="Date")
    ).set_index("Date")

    return merged

def compute_returns(merged):
    # Compute % returns
    returns = merged.pct_change().dropna()
    return returns

def compute_corr(returns):
    # Compute Correlation matrix
    corr = returns.corr()
    return corr

def pairwise_regression(returns):
    # Initialise results & model
    results = []
    model = LinearRegression()

    # Build the pairs
    cols = list(returns.columns)
    pairs = permutations(cols, 2)

    # Pairwise linear regression
    for x_col, y_col in pairs:
        # Implement the model
        X = returns[[x_col]].values
        y = returns[y_col].values
        model.fit(X, y)

        # Store parameters
        alpha = model.intercept_
        beta = model.coef_[0]
        r2 = model.score(X, y)
        results.append({
            "y": y_col, "x": x_col,
            "alpha": alpha, "beta": beta, "r2": r2
        })

    results_df = (
        pd.DataFrame(results)
          .sort_values("r2", ascending=False)
          .reset_index(drop=True)
    )
    return results_df

# Prepare Data for Webpage to plot R2 map
# convert to a matrix df using pivot
def pivot_matrices(results_df):

    r2_mat = results_df.pivot(index="y", columns="x", values="r2")

    # Add diagonal (self vs self) so heatmap looks complete
    for c in r2_mat.index:
        r2_mat.loc[c, c] = 1.0

    return r2_mat