
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

"""# Upload the data"""

def get_stock_data(ticker):
    return yf.Ticker(ticker)

def plot_stock(data,start,end,use_log=False):
    tickerData = data
    tickerDf = tickerData.history(period='1d', start=start,end=end)
    if use_log == True:
        df = np.log(tickerDf['Close'])
        title = tickerData.info["longName"]+" Closing log-Prices"
    if use_log==False:
        df=tickerDf['Close']
        title = tickerData.info["longName"]+" Closing Prices"
    fig=plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.plot(df)
    plt.title(title)
    st.pyplot(fig)
    

def find_arima(data,start,end,use_log=False):
    tickerData = data
    tickerDf = tickerData.history(period='1d', start=start,end=end)
    if use_log == True:
        df = np.log(tickerDf['Close'])
    else:
        df=tickerDf['Close']

    train_data, test_data = df[0:int(len(df)*0.9)], df[int(len(df)*0.9):]
    #plt.figure(figsize=(10,6))
    #plt.grid(True)
    #plt.xlabel('Dates')
    #plt.ylabel('Closing Prices')
    #plt.plot(df_log, 'green', label='Train data')
    #plt.plot(test_data, 'blue', label='Test data')
    #plt.legend()

    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    
    pred,conf_int=model_autoARIMA.predict(n_periods=len(test_data),return_conf_int=True,alpha=0.05)
    pred_series = pd.Series(pred, index=test_data.index)

    lower_series = pd.Series(conf_int[:, 0], index=test_data.index)
    upper_series = pd.Series(conf_int[:, 1], index=test_data.index)

    # Plot
    fig=plt.figure(figsize=(10,5), dpi=100)
    plt.plot(train_data, label='training data')
    plt.plot(test_data, color = 'blue', label='Actual Stock Price')
    plt.plot(pred_series, color = 'orange',label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.10)
    plt.title(tickerData.info["longName"]+" Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(tickerData.info["longName"]+" Closing Price")
    plt.legend(loc='upper left', fontsize=8)

    return st.write(model_autoARIMA.summary()),st.pyplot(model_autoARIMA.plot_diagnostics(figsize=(15,8))),st.pyplot(fig)

'''

# report performance
mse = mean_squared_error(test_data, pred_series)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, pred_series)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, pred_series))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(pred_series - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

'''