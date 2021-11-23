import streamlit as st
from stock_modeling import plot_stock,find_arima
import datetime

#set_page_title("Stock Forecasting with Times Series Modeling") 

ticker = st.text_input('Stock Ticker', 'JNJ')
st.write('The current stock modeled is', ticker)

start = st.date_input("Start date",datetime.date(1980, 1, 1))
st.write('The start date is:', start)

end = st.date_input("End date",datetime.date(2021, 11, 15))
st.write('The end date is:', end)

if st.button(label="Plot stock"):
    plot_stock(ticker,start,end)

if st.button(label="Fit ARIMA model"):
    find_arima(ticker,start,end)

