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
        Auto-correlation of residuals - Pending - acf done 
        Homoscedasiticty of residuals - Pending 
        Normally distributed - Pending - Done
    
    2. Add model performance measures for different equations - R_square, ..      
    
    3. Store results in a dataframe for viewing different models - indexed by Lags -Done      
        
        
"""
from TimeSeries_Tests import *
from statsmodels.tsa.api import VAR, DynamicVAR
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.vector_ar import plotting
from statsmodels.tsa.stattools import acf, pacf

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
check_summary(mdata1['cpi_logdiff'])    
check_summary(mdata1['unemp'])

#Distribution by plotting Histogram
check_distribution(mdata1['cpi_logdiff'])
check_distribution(mdata1['unemp'])
check_distribution(mdata1['unemp'].diff().dropna())

#Stationarity Check:  
check_stationarity(mdata1['realgdp'])
check_stationarity(mdata1['realgdp_logdiff'])
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
    normality_dict = {}
    normal_var = [None] * len(data.columns)
    
    for lag in range(1, lags+1):
        
        results = model.fit(maxlags = lag)
        
        print ('Exogenous Variables for the model with Lag: %d \n '%lag+ str(results.exog_names))
        print (results.summary())
        
        if actual_plot ==True:
            results.plot()
            
        fitted_values = results.fittedvalues
        
        lag_order = results.k_ar
        
        forecast_values = pd.DataFrame(data = results.forecast(y= data.values[-lag_order:], steps=  5), columns = results.names)
        
        results.forecast_interval(y= data.values[-lag_order:], steps = 5)
        results.plot_forecast(steps = 5, plot_stderr = False)
        
        #test for normality of residuals
        for var in range(0, len(data.columns)):
            normal_var[var] = check_normality(results.resid[data.columns[var]])
        
        print("acf plot \n")
        #acf plots of residuals of each variable for 10 lags
        for var in range(0, len(data.columns)):
            acf_data = pd.DataFrame(acf(results.resid[data.columns[var]])[1:10])
            acf_data.columns = [data.columns[var]]
            acf_data.plot(kind = "bar")
            
            
            
        
        equations_lag = lag
        equation_model_data = pd.DataFrame(index = results.tvalues.index, columns = ["coefs", "std err", "tvalues", "pvalues"])
        
        writer = pd.ExcelWriter('Models.xlsx', engine='xlsxwriter')
        
        #writing results of equations to different excel sheets
        for var in range(0,len(data.columns)):
            
            sheet_name =  data.columns[var] + "lag" + str((equations_lag))
            intercept = np.asarray(results.intercept[var])
            coefs = results.coefs[0][var]
            equation_model_data["coefs"] = np.append(intercept, coefs)
            equation_model_data["std err"] = results.stderr
            equation_model_data["tvalues"] = results.tvalues
            equation_model_data["pvalues"] = results.pvalues
            equation_model_data.to_excel(writer,sheet_name)
            
                    
        result_dict['Lag_Order_{}'.format(lag)] = results
        normality_dict['Lag_Order_{}'.format(lag)] = normal_var

        
    return result_dict, normality_dict

make_var_model(data = mdata1, lags = 1, actual_plot = False )
    
