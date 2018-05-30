# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:56:08 2018

@author: Prakash.Tiwari

Purpose: Vector Auto regressive models

Transformations - Only on level variables
    Log Diff  
    Percentage
   
Variables - 
    Rate variables - unemp, infl
    level variables - GDP, CPI, HPI, CRE

Tests:
    Stationarity of series - Done
    Cointegration of the series - Pending

Function --> 

def make_var_model:
    To do:
    
    1. Add OLS Residual tests:
        Auto-correlation of residuals - Pending
        Homoscedasiticty of residuals - Pending
        Normally distributed - Pending
    
    2. Add model performance measures for different equations - R_square, ..      
    
    3. Store results in a dataframe for viewing different models - indexed by Lags       
        
        
"""
from TimeSeries_Tests import *
from statsmodels.tsa.api import VAR, DynamicVAR
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str

#########
# Load pre-loaded macroeconomic data from PANDAS
#########

mdata = sm.datasets.macrodata.load_pandas().data

 # prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)

mdata1 = mdata[['realgdp', 'cpi', 'unemp', 'infl']]
mdata1.index = pd.DatetimeIndex(quarterly)

"""
a. Take Log difference of the level variables
b. Take differences of the rate variables
"""
mdata1['realgdp_logdiff'] = pd.Series(np.log(mdata1['realgdp']).diff().dropna())
mdata1['cpi_logdiff'] = pd.Series(np.log(mdata1['cpi']).diff().dropna())

mdata1['unemp_diff'] = pd.Series(mdata1['unemp'].diff().dropna())
mdata1['infl_diff'] = pd.Series(mdata1['infl'].diff().dropna())

"""
a. Drop NA
"""
mdata1.dropna(inplace = True)


"""
a. Basic checks
"""
#Normal line plots
check_plots(mdata1['realgdp'])    
check_plots(mdata1['unemp'])    
check_plots(mdata1['unemp'].diff().dropna())
    
#Summary Statistics
check_summary(mdata1['logdiff_cpi'])    
check_summary(mdata1['unemp'])

#Distribution by plotting Histogram
check_distribution(mdata1['logdiff_cpi'])
check_distribution(mdata1['unemp'])
check_distribution(mdata1['unemp'].diff().dropna())

#Stationarity Check:  
check_stationarity(mdata1['realgdp'])
check_stationarity(mdata1['logdiff_realgdp'])
check_stationarity(mdata1['infl'])
check_stationarity(mdata1['unemp'].diff().dropna())


"""
a. Prepare modeling dataset
"""
temp_data = pd.DataFrame(mdata1).dropna()
model_data = temp_data[['unemp_diff', 'realgdp_logdiff', 'cpi_logdiff']]


"""
a. Create VAR models -
    To be updated:
        
"""

def make_var_model(data, lags = 1, actual_plot = False ):
 
    # make a VAR model
    model = VAR(data)
    
    result_dict = {}
    for lag in range(1, lags+1):
        
        results = model.fit(maxlags = lag)
        
        print 'Exogenous Variables for the model with Lag: %d \n '%lag+ str(results.exog_names)
        print results.summary()
        
        if actual_plot ==True:
            results.plot()
            
        fitted_values = results.fittedvalues
        
        lag_order = results.k_ar
        
        forecast_values = pd.DataFrame(data = results.forecast(y= model_data.values[-lag_order:], steps=  5), columns = results.names)
        
        results.forecast_interval(y= model_data.values[-lag_order:], steps = 5)
        results.plot_forecast(steps = 5, plot_stderr = False)
        
        result_dict['Lag_Order_{}'.format(lag)] = results

    return result_dict

make_var_model(data = model_data, lags = 1, actual_plot = False )
    
