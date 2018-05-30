# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:50:36 2018

@author: Prakash.Tiwari

Functions -->

check_summary : Basic summary stats
check_distribution : Plot histogram
check_stationarity : Check stationarity
check_plots : Basic line plots

"""
import os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller

#Summary Statistics
def check_summary(series):
    """
    Split the series into two and check summary statistics for both the series
    """
    
    X = series.values
    split = len(X) / 2
    X1, X2 = X[0:split], X[split:]
    meanoverall, mean1, mean2 = X.mean(), X1.mean(), X2.mean()
    varoverall, var1, var2 = X.var(), X1.var(), X2.var()
    print('mean overall = %f, \t mean1=%f, \t mean2=%f' % (meanoverall, mean1, mean2))
    print('variance overall= %f, \t variance1=%f, \t variance2=%f' % (varoverall, var1, var2))


#Distribution by plotting Histogram
def check_distribution(series):
    """
    Distribution by plotting Histogram
    """
    series.hist()
    plt.show()        


#Stationarity Check:
def check_stationarity(series):   
    """
    Check stationarity using ADF test
    Ho : The series is non-stationary and has a unit root --> p-value > 0.05
    H1 : The series is stationary and does not have a unit root --> p-value < 0.05
    """
    
    X = series.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
    
    if result[4]['5%'] > result[0]:
        print 'Series is stationary at 95% CI'
    elif result[4]['10%'] > result[0]:
        print 'Series is stationary at 90% CI but non-stationary at 95% CI'
    else:
        print 'Series is non-stationary'
        
#Series Plots:
def check_plots(series):
    """
    Plots the series as a line chart
    """
    plt.plot(series)
    plt.show()        
