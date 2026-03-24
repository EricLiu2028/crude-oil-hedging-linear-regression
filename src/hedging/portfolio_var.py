# Calculate portfolio return & variance

import numpy as np
import pandas as pd

def portfolio_return(returns_df, position):
    '''
    position = {"Brent": 0, "WTI": 0, ...}
    r_port = sum(quantity_i * return_i)
    '''
    q = np.array(list(position.values()), dtype=float) # dim = (5, )
    R = returns_df.to_numpy() # (T, 5)

    return pd.Series(R @ q, index=returns_df.index)


def portfolio_variance(returns_df, position):

    r_port = portfolio_return(returns_df, position)
    return float(r_port.var())