#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:25:11 2020

@author: psardin
"""


from pandas import read_csv, concat
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

# 14 days
t_step = 14

# Dataset normalisieren
def normalize_data(data):
    prices = data["prices"]
    data = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_prices = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    prices = scaler_prices.fit_transform(prices.values.reshape(-1,1))
    return data, prices, scaler, scaler_prices


# Teilung in train/test sets for Training and Test
def split_dataset(data):
    train_data, test_data = data[:train_split], data[train_split:]
    return train_data, test_data 

# Erkundung der Korrelation zwieschen den Features 
def show_Corr_map(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns,  rotation =90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns)
    
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsixe= 14)
    plt.title("Correlation zwischen Features", fontsize = 14)
    plt.show()


""" Supervised Problem"""
#Berechnung der Inputs und Outputs
# Die Daten für das Modell werden in der folgenden Form gebracht: (z.B. Siehe Abbildung "XY_Train_Struktur" in dem Arbeitsordner)
# Mit dem Schema Many-To-One. Das heißt aus mehreren Feature wird eine Feature prädiziert.
# In dem aktuellen Fall habe ich eine 2-to-1. Aus "cancelation" und "prices" der Vergangenheit, wird  "prices" der Zukunft prädiziert.
# In einem anderen Skript (modell_lstm_5to1)  wurden mehr Features berücksichtigt.

def to_supervise(train_data, n_input, n_out = t_step):
    X, y = list(), list()
    in_start = 0

    for _ in range(len(train_data)):
        #Ende der Inputsequenze definieren
        in_end = in_start + n_input
        out_end = in_end + n_out
        # sicher stellen , dass ich eine komplete Sequenz aus den restlichen Werten bilden kann.
        if out_end <= len(train_data):
            X.append(train_data[in_start:in_end, :])
            # Nur die prices soll vorhersagt werden
            y.append(train_data[in_end:out_end, index_prices_sum])
        in_start += 1
        
    X_Train= np.array(X)
    Y_Train= np.array(y)
       
    return X_Train, Y_Train


#Inputs und Outputs für Test
def from_supervised(test_data, n_input, n_out = t_step):
    X, y = list(), list()
    in_start = 0
    for _ in range(len(test_data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(test_data):
            X.append(test_data[in_start:in_end, :])
            y.append(test_data[in_end:out_end, index_prices_sum])
        in_start += 1
        
    X_Test= np.array(X)
    Y_Test= np.array(y)
    
    
    return X_Test, Y_Test #, X_Test_tr, Y_Test_tr, x_test_scaler, y_test_scaler


""" Aufbau und Training des Modells"""
#Mit dem foldenden Modell wird man in der Lage sein aus Daten von zwei Wochen der 
#Vergangenheit zwei Wochen in der Zukunft zu Prädizieren, was vorteilhaft ist.
#Damit kann man z.B. im Jahr 2019 Vorhersagen durchführen, sobald man weißt, was zwei Wochen früher passiert ist. 
#Wie die Tatsächliche Prädiktion durchgeführt wird, zeigt den Skript  "prediction_01_to_14_07_2018". 

def build_model( X_Train, Y_Train, n_input, X_Test, Y_Test):

    # mini-bacht gradient descent als Trainingsalgorithmen ausgewählt 
    #(Sequenzgroße=14 <Batch-Size < Große des Trainingsdatensatzes )
	# So wurde die Batch-Size als vielfacheit der Eingangs- und Ausgangssequenzlänge gewählt
	
    verbose, epochs, batch_size = 1, 3000, 28
    n_daysteps, n_features, n_outputs = X_Train.shape[1], X_Train.shape[2], Y_Train.shape[1]
   
    Y_Train = Y_Train.reshape((Y_Train.shape[0], Y_Train.shape[1], 1))
    #Neues leeres Modell anlegen
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(n_daysteps, n_features)))
    model.add(BatchNormalization())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', optimizer='adam', metrics=['acc', 'mae'])
    # Modell trainieren
    #X_Val, Y_Val = X_Train[-val_split:], Y_Train[-val_split:]
    history = model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, validation_data=(X_Test, Y_Test), verbose=verbose)
    
    # Model Architektur
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
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('acc')
    # plt.legend()
    # plt.show()
    
    model.save('LSTM_Model_14_2to1.h5') 
    return model





data_rw = pd.read_pickle('data')
#Suche nach fehlenden Werten
n= data_rw.isnull().sum()

# Weil eine "cancellation" als einziger feature, der gut mit "prices" korreliert (Aufruf der Fuktion show_Corr_map(data_rw)), wird 
# er als einzigefeature weiter berücksichtigt
data = data_rw[["cancellation", "prices"]]



train_split = round(0.8569*len(data_rw))
#val_split = round(0.18*train_split)
test_split = round(0.1430*len(data_rw))

#Eingang- und Ausgangseqzenz haben die Länge 14 (Zwei Wochen)
t_step = 14
n_input = 14
index_prices_sum = 1

#show_Corr_map(data_rw)


data, prices, scaler, scaler_prices = normalize_data(data)
np.save('scaler_prices', scaler_prices)


#split into train and test
train_data, test_data = split_dataset(data)

# supervised (Input und Output)
X_Train, Y_Train = to_supervise(train_data, n_input, n_out = t_step)

#test data
X_Test, Y_Test = from_supervised(test_data, n_input, n_out = t_step)
np.save('X_Test', X_Test)
np.save('Y_Test', Y_Test)

# Das Modell
model = build_model(X_Train, Y_Train, n_input, X_Test, Y_Test)

    
                
    
    



