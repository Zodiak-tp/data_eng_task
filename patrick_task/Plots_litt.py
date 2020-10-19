# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 06:16:07 2020

@author: Tchuente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






future = list()

for i in range (1, 15):
    d = '2018-07-%02d' % i
    future.append(d)
    
    
   
    
pred_sequence_5to1 = np.load('pred_sequence_5to1.npy')[0]
pred_sequence_2to1 = np.load('pred_sequence_2to1.npy')[0]

prices_pred = np.load('prices_pred.npy')
prices_pred_lower = np.load('prices_pred_lower.npy')
prices_pred_upper = np.load('prices_pred_upper.npy')



#df = pd.DataFrame([future, pred_sequence_2to1, pred_sequence_5to1],  columns=["date", "2to1", "5to1"])

# plt.figure(figsize=(20,6))
# plt.plot( np.array(future), pred_sequence_5to1, label = "5to1")
# plt.plot( np.array(future), pred_sequence_2to1, label = "2to1")
# plt.legend()
# plt.title("Prädizierte Werte(prices) mit verschiedenen Modelle", fontsize=14)
# plt.grid(True)
# plt.show()
# plt.close()

plt.figure(figsize=(20,6))
plt.plot( np.array(future), pred_sequence_5to1, label = "LSTM_5to1")
plt.plot( np.array(future), pred_sequence_2to1, label = "LSTM_2to1")
plt.plot( np.array(future), prices_pred, label = "Prophet_pred")
plt.plot( np.array(future), prices_pred_lower, label = "Prophet_pred_lower")
plt.plot( np.array(future), prices_pred_upper, label = "Prophet_pred_upper")
plt.legend()
plt.title("Model Comparaison", fontsize=14)
plt.grid(True)
plt.show()
plt.close()


#
# plt.matshow(data.corr())
# plt.xticks(range(data.shape[1]), data.columns, fontsize = 14, rotation =90)
# plt.gca().xaxis.tick_bottom()
# plt.yticks(range(data.shape[1]), data.columns, fontsize = 14)


# cb = plt.colorbar()
# cb.ax.tick_params(labelsixe= 14)
# plt.title("Correlation zwischen Features", fontsize = 14)
# plt.show()