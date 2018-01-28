"""MC1-P2: Optimize a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo


def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]           # only portfolio symbols
    prices_SPY = prices_all['SPY']      # only SPY, for comparison later

    # Get daily portfolio value
    normed = prices.divide(prices.ix[0])
    allocated = normed * allocs 
    pos_vals = allocated * sv 
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


# This is the function that will be tested by the autograder
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    #Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  #automatically adds SPY
    prices = prices_all[syms]           #only portfolio symbols
    prices_SPY = prices_all['SPY']      #only SPY, for comparison later

    #uniformly initilize allocations 
    n_allocs = len(syms)
    allocs = np.ones(n_allocs) / n_allocs
    normed = prices.divide(prices.ix[0])
    bnds = ((0.,1.),) * n_allocs         #make sure all allocations are between 0 and 1
  
    #minimize sddr that gives optimal allocation values 
    results = spo.minimize(sd_daily_r, allocs, args=(normed,), method='SLSQP', bounds = bnds, \
                 constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}))
    opt_allocs = results.x 

    #Get daily portfolio value
    allocated = normed * opt_allocs 
    pos_vals = allocated * 1.0  # sv=1.0 
    port_val = pos_vals.sum(axis = 1)

    
    #call assess portfolio to get stats
    cr, adr, sddr, sr = assess_portfolio(sd = sd, ed = ed,\
                            syms = syms, allocs = opt_allocs, sv = 1, rfr=0.0, sf=252.0)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val/port_val[0],\
                            prices_SPY / prices_SPY.iloc[0]],\
                            keys=['Portfolio', 'SPY'],\
                            axis=1)
        plot_data(df_temp)
        pass

    return opt_allocs, cr, adr, sddr, sr


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
