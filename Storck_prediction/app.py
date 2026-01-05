import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = "2015-01-01"
end = "2024-12-31"

st.title('Stock Trend Prediction 📈')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)

if df.empty:
    st.error("❌ Invalid ticker or no data found.")
    st.stop()

st.subheader(f'Data for {user_input} from {start} to {end}')
st.write(df.describe())

st.subheader("First 5 rows of the dataset")
st.write(df.head())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

BASE_DIR = os.path.dirname(__file__)
model = load_model(os.path.join(BASE_DIR, "keras_model.h5"))

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted *= scale_factor
y_test *= scale_factor

st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original')
plt.plot(y_predicted, 'r', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
