# -*- coding: utf-8 -*-
"""
Created on Thu Oct  16 17:27:39 2020

@author: Tchuente
"""

from pandas import read_csv, concat
import matplotlib.pyplot as plt
from numpy import nan, isnan
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing 
import  math 


#Augrund der Größe der Daten, fand ich logischer sie separat zu behandeln.
#Es tauchen also ein paar Befehle, die sich wiederholen.


# Dateien Laden
data_15 = read_csv('POS_data_2015.csv')
data_16 = read_csv('POS_data_2016.csv')
data_17 = read_csv('POS_data_2017.csv')
data_18 = read_csv('POS_data_2018.csv')
data_18 = data_18.rename({'invoice_closed': 'invoice_close'}, axis=1)
data_a = read_csv('articles.csv')

# Die "Unnamed" Spalten in jeder Datei entfernen
data_a = data_a.loc[:, ~data_a.columns.str.contains('^Unnamed')]
data_15 = data_15.loc[:, ~data_15.columns.str.contains('^Unnamed')]
data_16 = data_16.loc[:, ~data_16.columns.str.contains('^Unnamed')]
data_17 = data_17.loc[:, ~data_17.columns.str.contains('^Unnamed')]
data_18 = data_18.loc[:, ~data_18.columns.str.contains('^Unnamed')]


# Weil die Datei sehr lang sind , werden separat die Spalten, die sich wiederholenden Elementen kodiert 
le = preprocessing.LabelEncoder()

data_15['kind'] = le.fit_transform(data_15['kind'])
data_15['group'] = le.fit_transform(data_15['group'])

data_16['kind'] = le.fit_transform(data_16['kind'])
data_16['group'] = le.fit_transform(data_16['group'])

data_17['kind'] = le.fit_transform(data_17['kind'])
data_17['group'] = le.fit_transform(data_17['group'])

data_18['kind'] = le.fit_transform(data_18['kind'])
data_18['group'] = le.fit_transform(data_18['group'])


#Teilung  der Spalte "table" in zwei Spalten
data_15[['table_1','table_2']] = data_15.table.str.split("/", expand=True)
data_16[['table_1','table_2']] = data_16.table.str.split("/", expand=True)
data_17[['table_1','table_2']] = data_17.table.str.split("/", expand=True)
data_18[['table_1','table_2']] = data_18.table.str.split("/", expand=True)

# Teilung der Datumeingabe in Tagen und Zeit
data_15['time'] = pd.to_datetime(data_15['time'])
data_15['date'] = [d.date() for d in data_15['time']]
data_15['year'] = data_15['time'].dt.year

data_16['time'] = pd.to_datetime(data_16['time'])
data_16['date'] = [d.date() for d in data_16['time']]
data_16['year'] = data_16['time'].dt.year

data_17['time'] = pd.to_datetime(data_17['time'])
data_17['date'] = [d.date() for d in data_17['time']]
data_17['year'] = data_17['time'].dt.year

data_18['time'] = pd.to_datetime(data_18['time'])
data_18['date'] = [d.date() for d in data_18['time']]
data_18['year'] = data_18['time'].dt.year



#Weil später die  Prädiktion pro Tag durchgeführt werden soll, ist von wenigen Interesse wie lange eine 
#Rechnung offen bleibt. Aus diesem Grund werden folgenden Spalten (Dauer einer rechnung) nicht mehr berücksichtigt.
columns = ['invoice_close', 'invoice_opened', "invoice", "ticket", "time", "table"]
data_15.drop(columns, inplace=True, axis=1)
data_16.drop(columns, inplace=True, axis=1)
data_17.drop(columns, inplace=True, axis=1)
data_18.drop(columns, inplace=True, axis=1)



#Die Funktion gibt für eine gegebene article_nummer abhängig vom Jahr den Preis in einem Dictionary zurück.
def getArtiNumbAndYear_df(article_number):
    keys = list()
    if article_number not in keys:
        k = article_number
        y = data_a["year"][data_a['article_number'] == article_number].values.tolist() 
        p = data_a["price"][data_a['article_number'] == article_number].values.tolist()
        keys.append(article_number)
    return (k, dict(zip(y,p)))

#Hier wird für jede article_number aus der Tabelle article das Jahr und den Preis sortiert
NumbYearPriceDict = dict(map(getArtiNumbAndYear_df, data_a["article_number"].values.tolist()))


# Mit einer gegebenen article_nummer und das entsprechende  Datum galangt man an den Preis.
# Weil Preise für ein paar article_nummer für die Jahre 2014 und 2016 nicht in der Tabelle 
#article vorhanden waren, ersetzte ich sie durch nan-Value.
def getPrice(article_number, year):
    if year in NumbYearPriceDict[article_number]:
        p = NumbYearPriceDict[article_number][year]
    else:
        p = math.nan
    return p
   

# Jeder Tabell wird mit einer neuen Spalte "Preis" versehen
prices_18 = list(map(getPrice, data_18["article_number"].values.tolist(), data_18["year"].values.tolist()))
prices_16 = list(map(getPrice, data_16["article_number"].values.tolist(), data_16["year"].values.tolist()))
prices_17 = list(map(getPrice, data_17["article_number"].values.tolist(), data_17["year"].values.tolist()))
prices_15 = list(map(getPrice, data_15["article_number"].values.tolist(), data_15["year"].values.tolist()))


data_15["prices"] = prices_15
data_16["prices"] = prices_16
data_17["prices"] = prices_17
data_18["prices"] = prices_18



#Ein Dataset aus allen Daten bilden und die nan Werte entfernen
data = [data_15, data_16, data_17, data_18]
data = pd.concat(data)

data = data[data['prices'].notna()]




# Hier werden die angaben aus der Tablle pro Tag sortiert. Dafür werden Operationen für die übrigen Spalten 
#durchgeführt
df1 = data[["kind", "date"]]
df1 =df1.groupby(['date']).mean()

df2 = data[["guests", "date"]]
df2 = df2.groupby(['date']).mean()

df3 = data[["group", "date"]]
df3 = df3.groupby(['date']).mean()

df4 = data[["cancellation", "date"]]
df4 = df4.groupby(['date']).sum()

df5 = data[["prices", "date"]]
df5 = df5.groupby(['date']).sum()

#Auswahl der Features für den Dataset. Ich habe mich zuers auf diesen Features beschänkt weil die andere Informationen
# nicht mit den Preisangaben korrelierten.
data = pd.concat([df1["kind"], df2["guests"], df3["group"], df4["cancellation"], df5["prices"]], axis=1)

# Dataset speichern. Er wird in den anderen Skripte wieder geladen und verwendet.
data.to_pickle("data") 























# data.groupby('date').aggregate({'kind':'mean', 
#                           'guests':'mean', 
#                           'group':'mean',
#                           'cancellation':'sum',
#                           'table_1':'mean',
#                           'table_2':'mean',
#                           'year':'max',
#                           'prices':'sum'})


















