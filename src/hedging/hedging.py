# Compute hedge suggestions
'''
return turn a text describing the optimum hedge suggestion
return a record for hedge_table to show the best 3 hedging data
'''

import pandas as pd
import numpy as np
from .portfolio_var import portfolio_variance

# Transaction Cost
SPREAD = {
    "Brent": 0.01,
    "WTI": 0.02,
    "GO": 0.05,
    "HO": 0.10,
    "RBOB": 0.05
}



def get_current_position(inventory_df):
    '''
    convert inventory dataframe to a dictionary
    '''
    return inventory_df.set_index("Product")["Quantity"].astype(float).to_dict()



def calculate_position_metrics(returns, position, hedge_asset, hedge_qty):
    '''
    Calculte position metrics for a single hedge asset

    parameters:
    - returns: Dataframe of future returns for five assets
    - position: Dictionary of current inventory/position
    - hedge_asset: asset used for hedging
    - hedge_qty: quantity of hedge_asset for heding

    returns:
    - var_before: portfolio variance before hedging
    - var_after: portfolio variance after hedging
    - var_change: change in portfolio variance
    - cost: transaction cost
    '''

    # Calculate var_before
    var_before = portfolio_variance(returns, position)

    # Position after hedging
    new_pos = position.copy()
    new_pos[hedge_asset] += hedge_qty

    # Calculate var_after
    var_after = portfolio_variance(returns, new_pos)

    # Calculate var_change & total cost
    var_change = var_after - var_before
    cost = SPREAD[hedge_asset] * abs(hedge_qty)

    return var_before, var_after, var_change, cost



def calculate_hedging_score(var_change, cost):
    '''
    Variance reduction per cost

    score = -var_change / cost
    '''

    score = -var_change / cost

    return score



def build_hedge_summary(
    hedge_asset,
    hedge_qty,
    beta,
    r2,
    var_before,
    var_after,
    var_change,
    cost,
):
    '''
    summarising the hedging data for the hedge_asset
    into a dictionary, used for dashboard's hedge table construction
    '''

    score = calculate_hedging_score(var_change, cost)

    return {
        "hedge_asset": hedge_asset,
        "hedge_qty": hedge_qty,
        "beta": beta,
        "r2": r2,
        "var_before": var_before,
        "var_after": var_after,
        "var_change": var_change,
        "cost": cost,
        "score": score,
    }



def get_regression_params(results_df, y_col, x_col):
    '''
    get regression parameters for two assets

    returns:
    - beta: beta between two assets
    - r2: model fitness for the assets
    '''

    # store the target row into a Series by iloc[0]
    # so row["beta"] and row["r2"] are scalars rather than Series
    row = results_df[
        (results_df["y"]==y_col) &
        (results_df["x"]==x_col)
    ].iloc[0]

    beta = row["beta"]
    r2 = row["r2"]

    return beta, r2



def calculate_brent_hedge_qty(exposure_asset, exposure_qty, results_df):
    '''
    calculate hedge data when using Brent for hedging

    returns:
    - brent_hedge_qty: hedging quantity when using Brent
    - beta_to_brent: beta for exposure_asset to Brent
    - r2_to_brent: r2 for exposure_asset to Brent
    '''
    if exposure_asset == "Brent":
        brent_hedge_qty = -1.0 * exposure_qty
        beta_to_brent = 1.0
        r2_to_brent = 1.0
    else:
        beta_to_brent, r2_to_brent = get_regression_params(
            results_df, y_col=exposure_asset, x_col="Brent"
        )
        brent_hedge_qty = -beta_to_brent * exposure_qty

    return brent_hedge_qty, beta_to_brent, r2_to_brent



def generate_brent_hedge(
    returns,
    position,
    exposure_asset,
    exposure_qty,
    results_df,
):
    '''
    generate the hedging strategy when using Brent as hedge asset

    returns:
    - brent_summary: dictionary of hedging data
    - brent_hedge_qty: hedging quantity when using Brent
    - beta_to_brent: beta for exposure asset to Brent
    '''

    # Calculate hedging quantity and regression params
    brent_hedge_qty, beta_to_brent, r2_to_brent = calculate_brent_hedge_qty(
        exposure_asset, exposure_qty, results_df
    )

    # Calculate metrics for hedging
    var_before, var_after, var_change, cost = calculate_position_metrics(
        returns, position, "Brent", brent_hedge_qty
    )

    # Generate the summary
    brent_summary = build_hedge_summary(
        hedge_asset="Brent",
        hedge_qty=brent_hedge_qty,
        beta=beta_to_brent,
        r2=r2_to_brent,
        var_before=var_before,
        var_after=var_after,
        var_change=var_change,
        cost=cost,
    )

    return brent_summary, brent_hedge_qty, beta_to_brent



def get_candidate_assets(returns, exposure_asset):
    '''
    the list of assets except exposure_asset and Brent
    so they can be used as hedging candidates
    '''

    candidates = [c for c in returns.columns if c not in [exposure_asset, "Brent"]]

    return candidates



def calculate_candidate_asset_hedge_qty(candidate_asset, brent_hedge_qty, results_df):
    '''
    candidate's hedge quantity = brent_hedge_quantity * beta(Brent to candidate)
    '''

    beta_brent_to_candidate, r2_brent_to_candidate = get_regression_params(
        results_df, y_col="Brent", x_col=candidate_asset
    )

    candidate_hedge_qty = brent_hedge_qty * beta_brent_to_candidate

    return candidate_hedge_qty, beta_brent_to_candidate, r2_brent_to_candidate



def generate_candidate_asset_summary(
    returns,
    position,
    candidate_asset,
    brent_hedge_qty,
    beta_to_brent,
    results_df,
):
    # Calculate hedge quantity and regression params
    candidate_hedge_qty, beta_brent_to_candidate, r2_brent_to_candidate = calculate_candidate_asset_hedge_qty(
        candidate_asset, brent_hedge_qty, results_df
    )

    # Calculate effective beta
    effective_beta = beta_to_brent * beta_brent_to_candidate

    # Calculate metrics for hedging
    var_before, var_after, var_change, cost = calculate_position_metrics(
        returns, position, candidate_asset, candidate_hedge_qty
    )

    # Generate the hedge summary
    hedge_summary = build_hedge_summary(
        hedge_asset=candidate_asset,
        hedge_qty=candidate_hedge_qty,
        beta=effective_beta,
        r2=r2_brent_to_candidate,
        var_before=var_before,
        var_after=var_after,
        var_change=var_change,
        cost=cost,
    )

    return hedge_summary



def combined_candidate_hedge_summaries(
    returns,
    position,
    exposure_asset,
    brent_hedge_qty,
    beta_to_brent,
    results_df,
):
    '''
    store the hedge summaries for candidates into a list
    '''

    candidates = get_candidate_assets(returns, exposure_asset)

    combined_summaries = []
    for candidate in candidates:
        hedge_summary = generate_candidate_asset_summary(
            returns,
            position,
            candidate,
            brent_hedge_qty,
            beta_to_brent,
            results_df,
        )
        combined_summaries.append(hedge_summary)

    return combined_summaries



def generate_hedge_table_row(hedge_summary):
    '''
    format a single hedge summary into data for hedge table
    e.g., convert quantity into action (buy or sell)
    '''

    qty = hedge_summary["hedge_qty"]
    action = "BUY" if qty > 0 else "SELL"
    
    return {
        "hedge_asset": hedge_summary["hedge_asset"],
        "action": action,
        "hedge_qty": round(abs(qty), 4),
        "beta": round(hedge_summary["beta"], 5),
        "r2": round(hedge_summary["r2"], 5),
        "var_before": round(hedge_summary["var_before"], 5),
        "var_after": round(hedge_summary["var_after"], 5),
        "var_change": round(hedge_summary["var_change"], 5),
        "cost": round(hedge_summary["cost"], 5),
        "score": round(hedge_summary["score"], 5),
    }



def generate_hedge_table_data(scored_df):
    '''
    store all hedge table rows into a list

    parameters:
    - scored_df: a dataframe storing all hedge summaries
                 that has been sorted based on score
    '''

    hedge_table_data = []
    for _, record in scored_df.iterrows():
        row = generate_hedge_table_row(record)
        hedge_table_data.append(row)
    
    return hedge_table_data


def generate_summary_text(best_hedge, exposure_asset, exposure_qty):
    '''
    generate the text for optimal hedging strategy
    '''

    best_asset = best_hedge["hedge_asset"]
    best_qty = best_hedge["hedge_qty"]
    best_action = "BUY" if best_qty > 0 else "SELL"
    
    summary_text = (
        f"Exposure {exposure_qty:+g} units of {exposure_asset}. "
        f"Best hedge: {best_action} {abs(best_qty):.4f} units of {best_asset}. "
        f"Var before={best_hedge['var_before']:.5g}, "
        f"Var change={best_hedge['var_change']:.5g}, "
        f"cost={best_hedge['cost']:.5g}, "
        f"score={best_hedge['score']:.5g}"
    )
    
    return summary_text



def compute_hedging_strategy(
    results_df,
    returns,
    inventory_df,
    exposure_asset,
    exposure_qty,
):
    '''
    Generate hedge_table_data & summary_text for dashboard

    returns:
    - hedge_table_data: a record (list of dictionaries) of hedge summaries
    - summary_text: text for optimal hedging strategy
    '''

    # Get current position
    position = get_current_position(inventory_df)

    # Generate hedging strategy for Brent
    brent_summary, brent_hedge_qty, beta_to_brent = generate_brent_hedge(
        returns,
        position,
        exposure_asset,
        exposure_qty,
        results_df,
    )

    all_hedge_summaries = [] if (exposure_asset=="Brent") else [brent_summary]

    # Generate candidate assets' hedging strategy
    # benmarked on Brent
    candidate_hedge_summaries = combined_candidate_hedge_summaries(
        returns,
        position,
        exposure_asset,
        brent_hedge_qty,
        beta_to_brent,
        results_df,
    )

    all_hedge_summaries.extend(candidate_hedge_summaries)

    # Sorting on score in decreasing order
    scored_df = pd.DataFrame(all_hedge_summaries).sort_values("score", ascending=False)

    # Convert scored_df to hedge_table_data
    hedge_table_data = generate_hedge_table_data(scored_df)

    # Generate optimal hedging strategy prompt
    best_hedge = scored_df.iloc[0]
    summary_text = generate_summary_text(best_hedge, exposure_asset, exposure_qty)

    return hedge_table_data, summary_text

