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


def indicators(sd = dt.datetime(2008,1,1),\
               ed = dt.datetime(2009,12,31),\
               syms = ['JPM'],\
               window_size = 20,\
               k = 2):
    """
    This function generatos set of indicators along with helpers, that can be used
    to create trading signals.
    """
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    price = prices_all[syms]            # only portfolio symbols
    #price = price / price.iloc[0]
    price = normalize_and_fill(price)


    #get SMA
    sma = price.rolling(window_size).mean()

    #get Bolinger Bands 
    rstd = price.rolling(window_size).std()
    upper_b = sma + k*rstd
    lower_b = sma - k*rstd
   
    #get momentum
    roll_days = 5
    momentum = calc_momentum(price, roll_days)
    
    #get sddr = volatility
    daily_r = price / price.shift(1) - 1.00
    sddr = daily_r.std()
    vol = sddr
    
    return price, sma, upper_b, lower_b, momentum, vol

def calc_momentum(price, roll_days):
    """ Calculates Momentum given normalized price data frame"""
    momentum = price/price.shift(roll_days) - 1
    return momentum

def normalize_and_fill(price):
    price.fillna(method='ffill', inplace=True)
    price.fillna(method='bfill', inplace=True)
    return price / price.ix[0, :]


def test_code():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['JPM']

    price,sma,upper_b,lower_b,momentum,vol = indicators(sd = start_date, ed = end_date, syms = symbols)

    df_sma = pd.concat([price, sma, price/sma],\
                       keys = ['Price', 'SMA', 'Price/SMA'],\
                       axis = 1)
    #SMA
    df = df_sma
    df.dropna()
    fig = plt.figure(figsize=(10,5))
    plt.title('Simple Moving Average', size = 14)
    plt.plot(df.index,df.Price, linestyle='-',color='blue')
    plt.plot(df.index,df.SMA, linestyle='-',color='orange')
    plt.plot(df.index,df.Price/df.SMA, linestyle='-',color='green')
    plt.legend(["Price","SMA","Price/SMA"], loc='best')
    plt.xticks(rotation=45, size=12) 
    plt.yticks(size=12)          
    plt.xlabel('Date', size=14)
    plt.ylabel('')
    fig.set_tight_layout(True)
    plt.show()
    fig.savefig('plots/SMA.pdf')

    #Bollinger Bands
    df_bb = pd.concat([price, sma, upper_b, lower_b],\
                      keys = ['Price', 'SMA', 'Upper_BB', 'Lower_BB'],\
                      axis = 1)
    df_bb.dropna()
    fig = plt.figure(figsize=(10,5))
    plt.plot(df_bb.index,df_bb.Price, linestyle='-',color='blue')
    plt.plot(df_bb.index,df_bb.SMA, linestyle='-',color='orange')
    plt.plot(df_bb.index,df_bb.Upper_BB, linestyle='-',color='green')
    plt.plot(df_bb.index,df_bb.Lower_BB, linestyle='-',color='black')
    plt.title('Bollinger Bands', size=14)
    plt.legend(["Price","SMA","Upper Band", 'Lower Band'], loc='best')
    plt.xticks(rotation=45, size=12) 
    plt.yticks(size=12)          
    plt.xlabel('Date', size=14)
    plt.ylabel('')
    fig.set_tight_layout(True)
    plt.show()
    fig.savefig('plots/BB.pdf')

    
    df_m = pd.concat([price,sma,momentum],\
                     keys = ['Price','SMA','Momentum'],\
                     axis = 1)
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(df_m.index,df_m.Price, linestyle='-',color='blue')
    #plt.plot(df_m.index,df_m.SMA, linestyle='-',color='orange')
    plt.plot(df_m.index,df_m.Momentum, linestyle='-',color='green')
    plt.axhline(y=0.0, color='black', linestyle='--')
    plt.title('Momentum', size=14)
    plt.legend(["Price","Momentum"], loc='best')
    plt.xticks(rotation=45, size=12) 
    plt.yticks(size=12)          
    plt.xlabel('Date', size=14)
    plt.ylabel('')
    fig.set_tight_layout(True)
    plt.show()
    fig.savefig('plots/Momentum.pdf')

if __name__ == "__main__":
    test_code()
