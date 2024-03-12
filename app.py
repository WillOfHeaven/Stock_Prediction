import numpy as np
import csv as csv
import json as js
import matplotlib.pyplot as plt
import pandas_datareader as pd_data
import pickle
import requests
import keras
from keras.models import load_model
import streamlit as st
import datetime
import sklearn
# Enter your Alpha Vantage API key
#api_key = st.secrets["api_key"]
api_key = 'XO2YT4XMHPUERBBP'

import datetime
days_val = 720
# get today's date
today = datetime.date.today()

# calculate the date 365 days before today
days_before = datetime.timedelta(days=days_val)
date_before = today - days_before

st.title('Stock Price Prediction')
user_input = st.text_input("Enter the stock symbol", 'IBM')
# Specify the API endpoint and query parameters
url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': user_input,
    'apikey': api_key,
    'outputsize': 'full'
}

# Send the API request and parse the JSON response
response = requests.get(url, params=params)
data_IBM = js.loads(response.text)
#st.write(data_IBM)
try:
  daily_data = data_IBM['Time Series (Daily)']
except KeyError:
  print("Error: 'Time Series (Daily)' key not found in data")
  # Handle the missing key here (e.g., display an error message, try a different source)
  # You can exit the script or use a default value depending on your application logic
  exit()  # Example of exiting if the key is missing
dates = list(daily_data.keys())[:days_val][::-1]  # Slice the most recent 600 dates and reverse the order
header = ['Date', '1. open', '2. high', '3. low', '4. close', '5. volume']
rows = []
for date in dates:
    row = [date] + [float(daily_data[date][col]) for col in header[1:]]
    rows.append(row)
# Write the data to a CSV file
with open('data_IBM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
import pandas as pd
# Read CSV file into a pandas DataFrame
df = pd.read_csv('data_IBM.csv')
df = df.rename(columns={'4. close':'close','5. volume':'volume'})
df = df.rename(columns={'1. open':'open','2. high':'high','3. low':'low'})
#df.drop([dividend amount', '8. split coefficient'], axis=1, inplace=True)

#Describing Data

st.subheader('Raw Data for the stock symbol: '+user_input+'\n Time Period : '+str(date_before)+' to '+str(today))
st.write(df.describe())

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)
st.download_button(
    label="Download data as CSV",
    data= csv ,
    file_name=user_input+'_data.csv',
    mime='text/csv',
)


#Visualizing Data

st.subheader('Closing Price vs Time Chart')
# fig =  plt.figure(figsize=(12,6))
# plt.plot(df.close)
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# read in the stock data from a CSV file
#df = pd.read_csv('stock_data.csv')

# create a trace for the stock price

# create a figure with two y-axes
# fig =  go.Figure(figsize=(12,6))
fig = make_subplots(specs=[[{"secondary_y": True}]])
#fig =  make_subplots(specs=[[{"secondary_y": True}]])
fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)

# add the trace for the stock price
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['close'], mode='lines', name='stock price'),
    secondary_y=False
)
fig1.add_trace(go.Scatter(x=df['Date'], y=df['close'], mode='lines', name='stock price', line_color='firebrick'))
ma100 = df.close.rolling(100).mean()
fig1.add_trace(go.Scatter(x=df['Date'], y = ma100, mode='lines', name='100 day moving average', line_color='royalblue'))
ma200 = df.close.rolling(200).mean()
fig1.add_trace(go.Scatter(x=df['Date'], y = ma200, mode='lines', name='200 day moving average', line_color='green'))
# add the trace for the volume
fig.add_trace(
    go.Bar(x=df['Date'], y=df['volume'], name='volume'),
    secondary_y=True
)

# customize the layout
fig.update_layout(
    title='Stock price and volume',
    xaxis_title='Date',
    yaxis_title='Price'
)

# label the second y-axis
fig.update_yaxes(title_text="Volume", secondary_y=True)

# display the plot
#fig.show()

# display the plot
st.plotly_chart(fig, use_container_width=True)

# customize the layout
# set the layout
# data_plot = [trace1, trace2, trace3]
fig1.update_layout( 
    #title = 'Closing Price with Moving Averages',
    xaxis = dict(title = 'Date'),
    yaxis = dict(title = 'Price'),
)
# fig1.update_layout(
#     #
#     #width=int(st.beta_get_query_params()['w'][0]),
#     # height=int(st.beta_get_query_params()['h'][0])    
# )
#fig1.go.Figure(data=data_plot, layout=layout)
#st.write("")
st.subheader('Closing Price vs Time Chart with Moving Averages')
st.plotly_chart(fig1, use_container_width=True)


data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Splitting the data into x_train and y_train

# x_train = []
# y_train = []
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i,0])

#Loading the model

model = load_model('keras_mode.h5')

#feeding the data to the model

#testing the model

past_100_days = data_training.tail(100) 
final_df = pd.concat([past_100_days,data_testing],ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
    
x_test,y_test = np.array(x_test),np.array(y_test)
print(x_test.shape)
# Making Predicstion
y_predicted =  model.predict(x_test)
y_predicted =  y_predicted.reshape(-1)
#plt.plot(y_predicted)
print(y_predicted.shape)
#st.plotly_chart(y_predicted, use_container_width=True)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
print(scale_factor)
y_predicted  = y_predicted * scale_factor
y_test = y_test * scale_factor
#print(y_predicted)
#plt.plot(y_predicted)
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig2.add_trace(go.Scatter(x = df['Date'], y = y_test, mode = 'lines', name = 'Actual Price', line_color='firebrick'))
fig2.add_trace(go.Scatter(x = df['Date'], y = y_predicted, mode = 'lines', name = 'Predicted Price', line_color='royalblue'))
fig2.update_layout( 
    #title = 'Closing Price with Moving Averages',
    xaxis = dict(title = 'Date'),
    yaxis = dict(title = 'Price'),
)

st.subheader('Prediction using LSTM model')

st.plotly_chart(fig2, use_container_width=True)
st.write("Made by Sammy ---- urf Sambuddha Chatterjee, yes! and I did follow the geek for geeks tutorial ")
st.write("  a Stock Prediction App built using Streamlit, Keras, and Plotly! The app predicts stock prices based on Moving Averages of 100 and 200 days and also uses LSTM for more accurate predictions. I added a personal touch by using Plotly to create an interactive graph instead of images of graphs. As the Yahoo API was deprecated, I utilized Alpha Vantage for stock data. It was a challenging and rewarding project, and I learned a lot in the process! ")
