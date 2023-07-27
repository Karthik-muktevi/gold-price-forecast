import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from PIL import Image
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title("Gold Price Forecasting")
df = pd.read_csv("Gold_data.csv")

st.sidebar.header('Select Forecasting Period')
n = st.sidebar.slider('Days', 1, 365, 30)



df["t"] = np.arange(1,2183)

df["t_squared"] = df["t"]*df["t"]

df["log_price"] = np.log(df["price"])

import calendar
import datetime as dt
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df['month'] = df['month'].apply(lambda x: calendar.month_name[x])

month_dummies = pd.get_dummies(df['month'])
df = pd.concat([df,month_dummies],axis=1)
img=Image.open("Gold-price.jpg")

model=load(open('hwe_model_add_add.pkl','rb'))
st.image(img, caption='Gold ', use_column_width=True)


submit= st.button(label="get prediction")

if submit:
    prediction_2 = pd.DataFrame(model.forecast(n),columns=['Forecasted Price'])
    st.write(prediction_2)
    plt.figure(figsize = (10, 6))
    plt.plot(prediction_2)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Trend')
    st.pyplot(plt)
    



