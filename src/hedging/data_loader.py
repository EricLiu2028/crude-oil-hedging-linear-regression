import pandas as pd

def load_raw_data():
    brent_df = pd.read_csv("data/FUTURE_UK_IFEU_BRNF26.csv")
    wti_df = pd.read_csv("data/FUTURE_US_XNYM_CLF26.csv")
    ho_df = pd.read_csv("data/FUTURE_US_XNYM_HOF26.csv")
    go_df = pd.read_csv("data/FUTURE_UK_IFEU_GASF26.csv")
    rb_df = pd.read_csv("data/FUTURE_US_XNYM_RBF26.csv")

    return brent_df, wti_df, ho_df, go_df, rb_df

