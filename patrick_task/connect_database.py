# -*- coding: utf-8 -*-
"""
Created on Thu Oct  15 10:30:50 2020

@author: Tchuente
"""
import psycopg2
import numpy as np
import pandas as pd
from myfonctions import*

#Konnektion mit Postgres, um Zugang zu der Tabellen zu haben, die durch Lunch der Datenbank-Skripten aus  über Docker erstellt wurde.
#Falls man in seinem Host die Tabelle hat, kann man mit entsprechenden User, Passwort und Port Informationen sich einlogen und sich die Tabelle articles holen
conn = psycopg2.connect(database="postgres", user="postgres", password="fabiola", host="127.0.0.1", port="5432")
cursor = conn.cursor()

   
data = load_data("common", "articles", conn);
data.to_csv('articles.csv')
    

conn.commit() # <--- Änderung in der Database sichergestellt
conn.close()
cursor.close()

