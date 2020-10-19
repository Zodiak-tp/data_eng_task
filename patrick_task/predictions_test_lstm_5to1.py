# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:50:22 2020

@author: Tchuente
"""

import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
# from build_modell import test_data
import math


#Hier geht man wie in dem Skript "prediction_test_lstm_2to1
# Allerdings erwarte hier das Modell einnen Eingangsarray mit 5 Features.



""""-------------- Modell laden und Test - Prädiktion durchführen--------------------"""


###load model
model = tensorflow.keras.models.load_model('LSTM_Model_14_4.h5')


X_Test = np.load('X_Test.npy')
Y_Test = np.load('Y_Test.npy')



#Prediktion seq2seq
predictions = list()
for i in range(X_Test.shape[0]):
    # predict the next 3 seconds
    output_sequence = model.predict(X_Test[i].reshape((1, X_Test.shape[1], X_Test.shape[2])), verbose=1)
    # store the predictions
    predictions.append(output_sequence[0])
Y_pred = np.array(predictions)

Y_pred = Y_pred.reshape((Y_pred.shape[0], Y_pred.shape[1]))



Y_pred_13 = Y_pred[:,13]
Y_Test_13 = Y_Test[:,13]

Y_pred_12 = Y_pred[:,12]
Y_Test_12 = Y_Test[:,12]

Y_pred_11 = Y_pred[:,11]
Y_Test_11 = Y_Test[:,11]

Y_pred_10 = Y_pred[:,10]
Y_Test_10 = Y_Test[:,10]

Y_pred_9 = Y_pred[:,9]
Y_Test_9 = Y_Test[:,9]


ig, ax = plt.subplots(figsize=(20,5))
ax.plot(Y_pred_13, label='predicted')
ax.plot(Y_Test_13, label='actual')
plt.grid(True)
plt.legend()
plt.show()


ig, ax = plt.subplots(figsize=(20,5))
ax.plot(Y_pred_12, label='predicted')
ax.plot(Y_Test_12, label='actual')
plt.grid(True)
plt.legend()
plt.show()


ig, ax = plt.subplots(figsize=(20,5))
ax.plot(Y_pred_11, label='predicted')
ax.plot(Y_Test_11, label='actual')
plt.legend()
plt.grid(True)
plt.show()

ig, ax = plt.subplots(figsize=(20,5))
ax.plot(Y_pred_10, label='predicted')
ax.plot(Y_Test_10, label='actual')
plt.legend()
plt.grid(True)
plt.show()

ig, ax = plt.subplots(figsize=(20,5))
ax.plot(Y_pred_9, label='predicted')
ax.plot(Y_Test_9, label='actual')
plt.legend()
plt.grid(True)
plt.show()