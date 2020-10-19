#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:52:55 2020

@author: psardin
"""

from matplotlib import pyplot
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




# Für das SARIMAX Modell wurde nur ein univariate Dataset verwendet, als "prices"
# Die Vorhersage ist nicht gut gefallen, wahrscheinlich weil die Preise im Jahr 2018 im vergleich zum restlichen Datensatz höher sind.
 
def split_dataset(data):
    train_data, test_data = data[:train_split], data[train_split:]
    return train_data, test_data 

def plotting_dalyPrices(train_data, test_data):
    plt.figure(figsize=(16,6))
    plt.plot(train_data.index, train_data, label = "Train")
    plt.plot(test_data.index, test_data, label = "Test")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def the_modell(train_data):
    modell = sm.tsa.statespace.SARIMAX(train_data, order=(7,0,1), seasonal_order=(0, 1, 1,7)).fit()
    #modell = sm.tsa.statespace.SARIMAX(train_data, order=(7,0,1)).fit()
    return modell






data_rw = pd.read_pickle('data').reset_index()
data = data_rw["prices"]

# plt.matshow(data.corr())
# plt.xticks(range(data.shape[1]), data, fontsize=14, rotation=90)
# plt.gca().xaxis.tick_bottom()
# plt.yticks(range(data.shape[1]), data, fontsize=14)

# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title("Feature Correlation Heatmap", fontsize=14)
# plt.show()




scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data_.values.reshape(-1))


train_split = round(0.8569*len(data_rw))
#val_split = round(0.18*train_split)
test_split = round(0.1430*len(data_rw))


plot_acf(data)


train_data, test_data = split_dataset(data)
plotting_dalyPrices(train_data, test_data)

modell = the_modell(train_data)

price_pred = modell.predict(start = 1090, end = 1271, dynamic = False)
price_pred.index = test_data.index


plt.figure(figsize=(20,6))
plt.plot(train_data.index, train_data, label = "Train")
plt.plot(test_data.index, test_data, label = "Test")
plt.plot(price_pred, label = "perd")
plt.legend()
plt.grid(True)
plt.show()
plt.close()




