import streamlit as st
from stock_modeling import get_stock_data,plot_stock,find_arima
import datetime

#set_page_title("Stock Forecasting with Times Series Modeling") 



ticker = st.text_input('Stock Ticker', 'JNJ')
stock_data = get_stock_data(ticker)
st.write('The current modeled stock is', stock_data.info['longName'],".")

start = st.date_input("Start date",datetime.date(1980, 1, 1))
st.write('The start date is:', start)

end = st.date_input("End date",datetime.date(2021, 11, 15))
st.write('The end date is:', end)

use_log = st.checkbox('Use log')

if st.button(label="Plot stock"):
    plot_stock(stock_data,start,end)

if st.button(label="Fit ARIMA model"):
    find_arima(ticker,start,end,use_log)
