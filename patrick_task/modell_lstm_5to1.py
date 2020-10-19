# -*- coding: utf-8 -*-
"""
Created on Fri Oct  16 22:34:59 2020

@author: Tchuente
"""

from pandas import read_csv, concat
from matplotlib import pyplot
from numpy import nan, isnan
import numpy as np
import pandas as pd
import math

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, RepeatVector, Activation, Dropout, Bidirectional, TimeDistributed
from tensorflow. keras.utils import plot_model
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Hier geht man ähnlich wie bei dem "modell_lstm_2to1" vor.
# Allerding hat den Dataset hier 5 Features. Man nutzt also 5 Informationen
# aus der vergangenheit , um eine information in der Zukunft zu prädizieren.
# Die Länge der Sequenzen bliebt unverändert. Es ändern sich also nur den Dataset (2->% Features)
# und die Variable "index_prices_sum".



# 14 days
t_step = 14

# normalize features
def normalize_data(data):
    prices = data["prices"]
    data = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_prices = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    prices = scaler_prices.fit_transform(prices.values.reshape(-1,1))
    return data, prices, scaler, scaler_prices


# split a multivariate dataset into train/test sets for Training and Test
def split_dataset(data):
    train_data, test_data = data[:train_split], data[train_split:]
    return train_data, test_data 


""" Supervised Problem"""
#convert train data into inputs and aoutputs
def to_supervised(train_data, n_input, n_out = t_step):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(train_data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(train_data):
            X.append(train_data[in_start:in_end, :])
            y.append(train_data[in_end:out_end, index_prices_sum])
        # move along one time step
        in_start += 1
        
    X_Train= np.array(X)
    Y_Train= np.array(y)
       
    return X_Train, Y_Train


#convert train data into inputs and aoutputs
def from_supervised(test_data, n_input, n_out = t_step):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(test_data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(test_data):
            X.append(test_data[in_start:in_end, :])
            y.append(test_data[in_end:out_end, index_prices_sum])
        # move along one time step
        in_start += 1
        
    X_Test= np.array(X)
    Y_Test= np.array(y)
    
    
    return X_Test, Y_Test #, X_Test_tr, Y_Test_tr, x_test_scaler, y_test_scaler


""" Building the model"""
# train the model
def build_model( X_Train, Y_Train, n_input, X_Test, Y_Test):
    # define parameters
    verbose, epochs, batch_size = 1, 3500, 28
    n_daysteps, n_features, n_outputs = X_Train.shape[1], X_Train.shape[2], Y_Train.shape[1]
    # reshape output into [samples, timesteps, features]
    Y_Train = Y_Train.reshape((Y_Train.shape[0], Y_Train.shape[1], 1))
    #define model
    model = Sequential()
    model.add(LSTM(272, activation='relu', input_shape=(n_daysteps, n_features)))
    #model.add(BatchNormalization())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(272, activation='relu', return_sequences=True))
    #model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(160, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', optimizer='adam', metrics=['acc', 'mae'])
    # fit network
    #X_Val, Y_Val = X_Train[-val_split:], Y_Train[-val_split:]
    history = model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, validation_data=(X_Test, Y_Test), verbose=verbose)
    
    # Modell Architektur
    print(model.summary())
    acc = history.history['acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    #plotting training and validation loss
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    #plotting training and validation accuracy
    plt.clf()

    #val_acc = history.history['val_acc']
    #plt.plot(epochs, acc, label='Training acc')
    #plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    
    model.save('LSTM_Model_14_4.h5') 
    return model






















data_rw = pd.read_pickle('data')
n= data_rw.isnull().sum()



train_split = round(0.9*len(data_rw))
#val_split = round(0.18*train_split)
test_split = round(0.1*len(data_rw))

t_step = 14
n_input = 14
index_prices_sum = 4


data, prices, scaler, scaler_prices = normalize_data(data_rw)
np.save('scaler_prices', scaler_prices)


#split into train and test
train_data, test_data = split_dataset(data)

# supervised (input an output)
X_Train, Y_Train = to_supervised(train_data, n_input, n_out = t_step)

#test data
X_Test, Y_Test = from_supervised(test_data, n_input, n_out = t_step)
np.save('X_Test', X_Test)
np.save('Y_Test', Y_Test)

#the lstm model
model = build_model(X_Train, Y_Train, n_input, X_Test, Y_Test)

    
                
    
    



