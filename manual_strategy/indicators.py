"""  Generate Indicators

Copyright 2018, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data


def indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'], sv=100000):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    price = prices_all[syms]           # only portfolio symbols
   
    #calc bb value
    window_size = 20
    k = 2
    sma = price.rolling(window_size).mean()
    std_bb = price.rolling(window_size).std()
    bb = (price - sma) / (2*std_bb)

    # calc momentum
    delta = 5
    momentum = price/price.shift(delta) - 1
    
    
    #get sddr = volatility
    daily_r = price / price.shift(1) - 1.00
    sddr = daily_r.std()
    vol = sddr.values()
    
    return bb, std_bb, momentum, vol


def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['JPM']
    sv = 100000
    bb, std_bb, momentum, vol = indicators(sd = start_date, ed = end_date, syms = symbols, sv = sv)

    df_temp = pd.concat([port_val / port_val[0],\
                         prices_SPY / prices_SPY.iloc[0]],\
                        keys = ['Portfolio', 'SPY'],\
                        axis = 1)
    plot_data(df_temp)

   
if __name__ == "__main__":
    test_code()
