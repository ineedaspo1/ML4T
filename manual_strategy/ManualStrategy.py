"""  Manual Strategy by T. Ruzmetov

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
               sv = 100000, rolling_days = 20, k = 2 ):
    """
    This function uses indicators to generate orders data frame based on manual strategy.
    --------------------------
    Parameters
    k -  rolling_std prefactor for bb calculation
    rolling_days - number of rolling days for rolling_mean and r_std calculation
    """
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_normalized = normalize_stocks(prices)

    sma = compute_sma(prices_normalized, rolling_days)
    bb = compute_bollinger_bands(prices_normalized, rolling_days, sma, k)

    m_rd = 5
    momentum = compute_momentum(prices_normalized, rd = m_rd)

    df_sha = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
    df_ord = pd.DataFrame('BUY', index = prices_normalized.index, columns = ['Order'])
    df_sym = pd.DataFrame(symbol, index = prices_normalized.index, columns = ['Symbol'])
    df_mom = pd.DataFrame(momentum, index = prices_normalized.index, columns = ['Momentum'])
    net_holdings = 0
    #loop over rows to generate signals for trading
    for i, row in prices_normalized.iterrows():
        sma_value = sma.loc[i]['SMA']
        upper_b = bb.loc[i]['upper']
        lower_b = bb.loc[i]['lower']
        mom = df_mom.loc[i]['Momentum']
        price = row[symbol]

        if (price < lower_b) and (net_holdings < 1000):
            df_ord.loc[i] = 'BUY'
            if net_holdings == 0:
                df_sha.loc[i] = 1000
                net_holdings += 1000
            else:
                df_sha.loc[i] = 2000
                net_holdings += 2000
        elif (price > upper_b) and (net_holdings > -1000):
            df_ord.loc[i] = 'SELL'
            if net_holdings == 0:
                df_sha.loc[i] = 1000
                net_holdings -= 1000
            else:
                df_sha.loc[i] = 2000
                net_holdings -= 2000

    df = pd.concat([df_sym, df_ord, df_sha], axis=1) # merge all and make orders
    df.columns = ['Symbol', 'Order', 'Shares']
    df = df[df.Shares != 0] # drop rows with zero Shares
    return df


def compute_sma(price, rd):
    """ Compute rolling mean given normalized price and  #of_rolling_days=rd"""
    sma = pd.DataFrame(0, index = price.index, columns = ['SMA'])
    sma['SMA'] = price.rolling(window=rd, min_periods = rd).mean()
    return sma

def compute_bollinger_bands(price, rd, sma, k=2):
    """Calculated lower and upper Bollinger Ban """
    bb = pd.DataFrame(0, index = price.index, columns = ['lower','upper'])
    roll_std = pd.DataFrame(0, index = price.index, columns = ['band'])
    roll_std['band'] = price.rolling(window = rd, min_periods = rd).std()
    bb['upper'] = sma['SMA'] + roll_std['band'] * k
    bb['lower'] = sma['SMA'] - roll_std['band'] * k
    
    return bb

def compute_momentum(price, rd):
    """ Compute Momentum given normalized prices and  #of_rolling_days=rd"""
    m = pd.DataFrame(0, index = price.index, columns = ['Momentum'])
    m['Momentum'] = price.diff(rd)/price.shift(rd)
    return m

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)


def optimize_in_sample_port():
    """
    This function performs grid search to find optimum values for
    rolling_days and k
    !!!!!!!!!!!!!!!!! not yet finale!
    """
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000

    fin_val = [[]]
    rd = 0
    for i in range(0,20):
        k = 0.0
        rd = rd + 2
        for j in range(0,6):
            k = k + 0.5 
            df_trades = testPolicy(symbol = "JPM", sd = sd, ed = ed, sv = sv, rolling_days = rd, k = k)
            portvals = ms.compute_portvals(df_trades, start_val = sv)
            fin_val[i,j] = portvals[-1]
    

def test_code():
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    rd = 10
    k = 2.0
    
    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = sv, rolling_days = rd, k = k)
    portvals = ms.compute_portvals(df_trades, start_val = sv)

    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  
    prices_portval_normalized = normalize_stocks(portvals)
    prices =  prices_all[syms]
    prices_JPM_normalized = normalize_stocks(prices)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    cum_ret_Bench, avg_daily_ret_Bench, std_daily_ret_Bench, sharpe_ratio_Bench = ms.compute_portfolio_stats(prices)

    print('In Sample stats:')
    print "Average Daily Return for Portfolio: {}".format(avg_daily_ret)
    print "Average Daily Return for Benchmark: {}".format(avg_daily_ret_Bench)
    print "Cumulative Return of Portfolio : {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_Bench)
    print "Std for Daily Return of Portfolio : {}".format(std_daily_ret)
    print "Std for Daily Return of Benchmark : {}".format(std_daily_ret_Bench)
    
    ###########################################################################################
    #plot
    chart_df = pd.concat([prices_portval_normalized, prices_JPM_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(title='Manual Strategy and Benchmark Comparison for In Sample',\
                  use_index=True,\
                  color=['Black', 'Blue'],\
                  figsize=(10,5))

    for index, row in df_trades.iterrows():
        if df_trades.loc[index]['Order'] == 'BUY':
            plt.axvline(x=index, color='g', linestyle='-')
        elif df_trades.loc[index]['Order'] == 'SELL':
            plt.axvline(x=index, color='r', linestyle='-')

    #plt.set_tight_layout(True)
    plt.show()
    #plt.savefig('plots/MS_InSample.pdf')
    #############################################################################################
     

    sd=dt.datetime(2010,1,1)
    ed=dt.datetime(2011,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = sv, rolling_days = rd, k = k)
    portvals = ms.compute_portvals(df_trades, start_val = sv)

    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_portval_normalized = normalize_stocks(portvals)
    prices =  prices_all[syms]
    prices_JPM_normalized = normalize_stocks(prices)
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    cum_ret_Bench, avg_daily_ret_Bench, std_daily_ret_Bench, sharpe_ratio_Bench = ms.compute_portfolio_stats(prices)

    print('#################################################################################')
    print('Out of Sample Stats:')
    print "Average Daily Return for Portfolio : {}".format(avg_daily_ret)
    print "Average Daily Return for Benchmark: {}".format(avg_daily_ret_Bench)
    print "Cumulative Return of Portfolio : {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_Bench)
    print "Std for Daily Return of Portfolio : {}".format(std_daily_ret)
    print "Std for Daily Return of Benchmark : {}".format(std_daily_ret_Bench)

    ###########################################################################################
    #plot
    chart_df = pd.concat([prices_portval_normalized, prices_JPM_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(title='Manual Strategy and Benchmark Comparison for Out of Sample',\
                  use_index=True,\
                  color=['Black', 'Blue'],\
                  figsize=(10,5))
    #plt.set_tight_layout(True)
    plt.show()
    #plt.savefig('plots/MS_OutSample.png')
    ###########################################################################################
    
if __name__ == "__main__":
    test_code()
