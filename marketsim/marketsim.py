"""MC2-P1: Market simulator.

Copyright 2018, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'truzmetov3'


def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0):
    daily_rets = ((port_val/port_val.shift(1)) - 1).ix[1:] # daily return
    cr = (port_val.ix[-1]/port_val.ix[0]) - 1 # cumulative return
    adr = daily_rets.mean() # avg. daily return
    sdr = daily_rets.std() # std daily return
    sr = np.sqrt(sf)*(daily_rets-rfr).mean()/sdr # sharpe ratio

    start_date = port_val.index[0].to_datetime()
    end_date = port_val.index[-1].to_datetime()

    return cr, adr, sdr, sr, port_val, start_date, end_date

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # Add orders to dataframe
    orders = pd.read_csv(orders_file, index_col=0, parse_dates=True, sep=',')
    orders = orders.sort_index()
    
    start_date = orders.index[0].to_datetime()
    end_date = orders.index[-1].to_datetime() 

    dates = pd.date_range(start_date, end_date)
    dates = get_data(['SPY'], dates).index.get_values() 
    
    symbols = orders.get('Symbol').unique().tolist()
    
    # Read in adjusted closing prices for given symbols + dates
    prices_all = get_data(symbols, dates) # automatically adds SPY
    prices = prices_all[symbols] # only portfolio symbols
    prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1) # all dates in prices
    prices = prices.fillna(method='ffill') 
    prices_SPY = prices_all['SPY'] 

    # Create leverage dataframe
    leverage =  pd.DataFrame(columns=['leverage'], index=[dates])
    leverage.ix[:,['leverage']] = 0

    # Create cash-orders dataframe
    cashOrders = pd.DataFrame(columns=['cash_order'], index=[dates])
    cashOrders.ix[0, ['cash_order']] = start_val
    cashOrders = cashOrders.fillna(value=0)

    # Create share-orders dataframe
    shareOrders = pd.DataFrame(columns=symbols, index=[dates])
    shareOrders = shareOrders.fillna(value=0)

    # Create positions dataframe
    columns = ['cash']
    positions = pd.DataFrame(columns=columns, index=[dates])  
    positions.ix[0] = 0

    # Iterate through the rows in orders
    for i in range(len(orders)):
        if orders.ix[i]['Order'] == 'SELL':
            orderTypeConst = -1.0

        else:
            orderTypeConst = 1.0

        shareOrders.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] = shareOrders.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] + (orders.ix[i]['Shares'] * orderTypeConst) 
        orderCost =  prices.ix[orders.index[i].to_datetime()][orders.ix[i]['Symbol']] * orders.ix[i]['Shares'] * orderTypeConst 
        
        print "Stock:",orders.ix[i]['Symbol']
        print "Order type:",orders.ix[i]['Order']
        print "Cost of order:",orderCost
        print "Date order placed:",orders.index[i].to_datetime(),"\n"

        cashOrders.ix[orders.index[i].to_datetime(), ['cash_order']] = cashOrders.ix[orders.index[i].to_datetime(), ['cash_order']] - orderCost

    positions.cash = cashOrders.cash_order.cumsum() # tot. cash in portfolio
    shares = shareOrders.cumsum() # tot. # shares in portfolio
    positions = pd.concat([positions,shares * prices],axis=1)
    portvals = positions.sum(axis=1)

    return portvals

def test_code():
    # Define input parameters
    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    
    else:
        "Warning: code did not return a DataFrame"
    
    cr, adr, sdr, sr, port_val, start_dates, end_dates = compute_portfolio_stats(portvals)
    cr_SPY, adr_SPY, sdr_SPY, sr_SPY, spy_value, start_dates, end_dates = compute_portfolio_stats(get_data(['SPY'], dates=pd.date_range(start_dates, end_dates)))

    # Compare portfolio against $SPX
    print
    print "Date Range: {} to {}".format(start_dates, end_dates)
    print
    print "Sharpe Ratio of Fund: {}".format(sr)
    print "Sharpe Ratio of SPY : {}".format(sr_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cr)
    print "Cumulative Return of SPY : {}".format(cr_SPY)
    print
    print "Standard Deviation of Fund: {}".format(sdr)
    print "Standard Deviation of SPY : {}".format(sdr_SPY)
    print
    print "Average Daily Return of Fund: {}".format(adr)
    print "Average Daily Return of SPY : {}".format(adr_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
