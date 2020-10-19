# -*- coding: utf-8 -*-
"""
Created on Thu Oct  16 11:13:15 2020

@author: Tchuente
"""
import pandas as pd


# Lädt eine Tabelle aus einer Datenbank

def load_data(schema, table, connection):

    sql_command = "SELECT * FROM {}.{};".format(str(schema), str(table))
    print (sql_command)

    # Load the data
    data = pd.read_sql(sql_command, connection)

    print(data.shape)
    return (data)


# le = preprocessing.LabelEncoder()

# class MultiColumnLabelEncoder:
#     def __init__(self,columns = None):
#         self.columns = columns 

#     def fit(self,X,y=None):
#         return self 

#     def transform(self,X):
#         '''
#         Transforms columns of X specified in self.columns using
#         LabelEncoder(). If no columns specified, transforms all
#         columns in X.
#         '''
#         output = X.copy()
#         if self.columns is not None:
#             for col in self.columns:
#                 output[col] = le.fit_transform(output[col])
#         else:
#             for colname,col in output.iteritems():
#                 output[colname] = le.fit_transform(col)
#         return output

#     def fit_transform(self,X,y=None):
#         return self.fit(X,y).transform(X)