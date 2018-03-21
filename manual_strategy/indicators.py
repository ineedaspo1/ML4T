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


def indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['JPM'], \
    sv=100000, rfr=0.0, sf=252.0):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]           # only portfolio symbols
   
    # Get daily portfolio value
    normed = prices.divide(prices.ix[0])
    #allocated = normed * allocs 
    pos_vals = normed * sv 
    port_val = pos_vals.sum(axis = 1)
    
    # get cumulative return
    cr = (port_val[-1] -port_val[0]) / port_val[0] 

    # get daily return
    daily_r = port_val / port_val.shift(1) - 1.00

    # average daily return
    adr = daily_r.mean()

    # calc. std for daily return
    sddr = daily_r.std()

    # get sharp ratio
    sr = np.sqrt(sf) * (adr - rfr) / sddr         
   
    return cr, adr, sddr, sr


def sd_daily_r(allocs, normed):
    """ This function calculates standart deviation for daily return """
    allocated = normed * allocs 
    pos_vals = allocated  
    port_val = pos_vals.sum(axis = 1)
    daily_r = port_val / port_val.shift(1) - 1.00000
    sddr = daily_r.std()
    return sddr


def test_code():
    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM','X','GLD']
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
