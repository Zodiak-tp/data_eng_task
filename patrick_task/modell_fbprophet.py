#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 00:26:15 2020

@author: psardin
"""

from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Modelle mit der Bibliothek Prophet sind nur für univariate Prädiktionsprobleme geeignet
# Aus diesem Grun werde ich mich hier nur auf die Spalte "prices" konzentrieren.

# Laden des Datasets
data_rw = pd.read_pickle('data').reset_index()
data = data_rw[["date", "prices"]]

#Intern für schnellere Berechnungen müssen die Spalten umbennant werden
data.columns = ["ds", "y"]
#data["ds"] = pd.to_datetime (data["ds"] ).date()

data.plot(figsize=(30,5))
plt.grid(True)
plt.show()

# Neues Prophet-Modell wird erstellt
#model = Prophet(daily_seasonality=True)
model = Prophet()

#Modell wird trainiert
model.fit(data)



#Die Vorhersage wird durch Übergabe eines Datensets, der nur eine Spalte namens "ds" hat
#und Zeile mit gewünschte Zeitspanne für den zu vorhersagenden Intervall.

future = list()

for i in range (1, 15):
    d = '2018-07-%02d' % i
    future.append([d])
   
future = pd.DataFrame(future)
future.columns = ['ds']

prices_pred = model.predict(future)

#Es werden viele Informationen bei der Prädiktion ausgegeben. Und die tragen schon Namen. 
#Was uns aber interressiert ist eigentlicht nur die prädizietten Preise, die sich in de Spalte"yhat" stecken.
#Maximale und Minimale Werte davon könnten aber auch von Interesse sein.
print(prices_pred[['ds','yhat', 'yhat_lower', 'yhat_upper']])

plt.figure(figsize=(20,6))
plt.plot(data.index, data['y'], label = "past")
plt.plot(prices_pred['yhat'], label = "predicted")
plt.legend()
plt.grid(True)
plt.show()
plt.close()
