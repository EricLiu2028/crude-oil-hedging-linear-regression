# app.py
from hedging.data_loader import load_raw_data
from hedging.data_cleaner import align_dates
from hedging.regression import (
    build_merged_prices, compute_returns, compute_corr,
    pairwise_regression, pivot_matrices
)
from hedging.dashboard import create_app

def main():
    # Load Data
    brent_df, wti_df, ho_df, go_df, rb_df = load_raw_data()

    # Align Dates
    brent_df, wti_df, ho_df, go_df, rb_df = align_dates(
        brent_df, wti_df, ho_df, go_df, rb_df
    )

    # Merged + Returns + Corr
    merged = build_merged_prices(brent_df, wti_df, ho_df, go_df, rb_df)
    returns = compute_returns(merged)
    corr = compute_corr(returns)
    print("Correlation matrix:\n", corr)

    # Regression
    results_df = pairwise_regression(returns)
    print("Top 5 highest R^2:")
    print(results_df.head())

    r2_mat = pivot_matrices(results_df)

    # Dash app
    app = create_app(r2_mat, results_df, returns)
    app.run(debug=True)

if __name__ == "__main__":
    main()


# Presentation slides (pdf)
# formula (hedging efficiency) & logic, 
# asset using, screen shot of dashboard