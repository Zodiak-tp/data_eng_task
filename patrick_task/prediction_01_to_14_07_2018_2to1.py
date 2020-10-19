#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:04:47 2020

@author: psardin
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow
import numpy as np


data_rw = pd.read_pickle('data')

# Aus dem Dataset die letzte 2 Wochen wählen und für die Eingabe in dem 
#trainierten Modell vorbereiten (Einstellung dern Dimension)


data_for_task_pred = data_rw[["cancellation", "prices"]].tail(15)

# Weil es von dem 01.07 vorhegesagt werden soll lasse ich letzte 
#Zeile weg, weil sie infos vom 01.07 beinhaltet
data_for_task_pred = data_for_task_pred.iloc[:-1]

#Nur um die Skalierung zu bilden
prices_ = data_for_task_pred["prices"]


scaler = MinMaxScaler(feature_range=(0, 1))
scaler_prices = MinMaxScaler(feature_range=(0, 1))


data_for_task_pred_scaled = scaler.fit_transform(data_for_task_pred.values)
prices_scaled = scaler_prices.fit_transform(prices_.values.reshape(1, -1))



model = tensorflow.keras.models.load_model('LSTM_Model_14_2to1.h5')






""""-------------- --------------Prädiktion----------------------------------"""
output_sequence = model.predict(data_for_task_pred_scaled.reshape((1, 14, 2)), verbose=1)






pred_sequence = scaler_prices.inverse_transform(output_sequence.reshape(1,-1))
np.save('pred_sequence_2to1', pred_sequence)
