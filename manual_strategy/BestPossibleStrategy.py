""" Best Possible Strategy

Copyright 2018, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import marketsim as ms
from util import get_data, plot_data


def author():
    return 'truzmetov3'

def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),\
               sv = 100000, rolling_days = 2):
    """
    This function generates orders data frame for best possible strategy.
    """
    
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_normed = normalize_stocks(prices)
   
   
    #calc 'future price - current price'
    rd = rolling_days
    shift = look_into_future(prices_normed, rolling_days = rd)

    df_sha = pd.DataFrame(0, index = prices_normed.index, columns = ['Shares'])
    df_ord = pd.DataFrame('BUY', index = prices_normed.index, columns = ['Order'])
    df_sym = pd.DataFrame(symbol, index = prices_normed.index, columns = ['Symbol'])
    df_shift = pd.DataFrame(shift, index = prices_normed.index, columns = ['Shift'])
    net_holdings = 0

    #loop over rows to generate signals for trading
    for i, row in prices_normed.iterrows():
        val_shift = df_shift.loc[i]['Shift']
        price = row[symbol]

        if (val_shift > 0.0) and (net_holdings < 1000):
            df_ord.loc[i] = 'BUY'
            if net_holdings == 0:
                df_sha.loc[i] = 1000
                net_holdings += 1000
            else:
                df_sha.loc[i] = 2000
                net_holdings += 2000
                
        elif (val_shift < 0.0) and (net_holdings > -1000):
            df_ord.loc[i] = 'SELL'
            if net_holdings == 0:
                df_sha.loc[i] = 1000
                net_holdings -= 1000
            else:
                df_sha.loc[i] = 2000
                net_holdings -= 2000
                

    df = pd.concat([df_sym, df_ord, df_sha], axis=1)
    df.columns = ['Symbol', 'Order', 'Shares']
    df = df[df.Shares != 0] # drop rows with zero Shares

    return df


def look_into_future(prices_normed, rolling_days):
    price_shift = pd.DataFrame(0, index = prices_normed.index, columns = ['Shift'])
    price_shift['Shift'] = prices_normed.shift(periods=-rolling_days) - prices_normed
    return price_shift

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)


def optimize_in_sample_port(n_trials = 50, period = 1):
    """
    This function returns number of rooling days that gives max finale port_value
    """
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000

    fin_val = []
    rd_vals = []
    rd = 0
    for i in range(0,n_trials):
        rd = rd + period
        df_trades = testPolicy(symbol = "JPM", sd = sd, ed = ed, sv = sv, rolling_days = rd)
        portvals = ms.compute_portvals(df_trades, start_val = sv)
        fin_val.append(portvals[-1])
        rd_vals.append(rd)

    index = np.argmax(fin_val)
    opt_rd = rd_vals[index]    

    return opt_rd

def test_code():
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    opt_rd = optimize_in_sample_port(n_trials = 20, period = 1)
    rd = opt_rd
    
    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = sv, rolling_days = rd)
    portvals = ms.compute_portvals(df_trades, start_val = sv)

    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  
    prices_portval_normalized = normalize_stocks(portvals)
    prices =  prices_all[syms]
    prices_JPM_normalized = normalize_stocks(prices)
    
    chart_df = pd.concat([prices_portval_normalized, prices_JPM_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(title='BPS and Benchmark Comparison for In Sample', use_index=True, color=['Black', 'Blue'],lw=2)
           
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    cum_ret_Bench, avg_daily_ret_Bench, std_daily_ret_Bench, sharpe_ratio_Bench = ms.compute_portfolio_stats(prices)

    print('In Sample stats:')
    print "Optimal Rolling_Days Value: {}".format(rd)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_Bench)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    
    plt.show()

if __name__ == "__main__":
    test_code()
