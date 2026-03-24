import pandas as pd

def align_dates(brent_df, wti_df, ho_df, go_df, rb_df):
    # Change type to datetime
    brent_df["Date"] = pd.to_datetime(brent_df["Date"])
    wti_df["Date"]   = pd.to_datetime(wti_df["Date"])
    ho_df["Date"]    = pd.to_datetime(ho_df["Date"])
    go_df["Date"]    = pd.to_datetime(go_df["Date"])
    rb_df["Date"]    = pd.to_datetime(rb_df["Date"])

    # Extract which dates are in common
    common_date = (
        set(brent_df["Date"]) &
        set(wti_df["Date"]) &
        set(ho_df["Date"]) &
        set(go_df["Date"]) &
        set(rb_df["Date"])
    )

    # Update the original dataframes
    dfs = [brent_df, wti_df, ho_df, go_df, rb_df]
    dfs = [
        df[df["Date"].isin(common_date)]
            .sort_values("Date")
            .reset_index(drop=True)
        for df in dfs
    ]
    brent_df, wti_df, ho_df, go_df, rb_df = dfs

    return brent_df, wti_df, ho_df, go_df, rb_df
