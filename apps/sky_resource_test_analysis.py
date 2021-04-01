#Common imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pickle import dump
from pickle import load



#Import the data set


#Import the data set

raw_data = pd.read_csv(r'E:\restapi_sky\apps\sky_test_input1.csv')
raw_data.columns

raw_data = raw_data[[' Construction_Scale ','Equipment1', 'Equipment2', 'Equipment3',
       'Equipment4', 'Equipment5', 'Equipment6', 'Equipment7', 'Equipment8',
       'Equipment9', 'Equipment10', 'Equipment11', 'Equipment12',
       'Equipment13', 'Equipment14', 'Equipment15', 'Equipment16',
       'Equipment17']]
raw_data_completion= raw_data[[' Construction_Scale ']]
print(raw_data)
#Import standardization functions from scikit-learn

from sklearn.preprocessing import StandardScaler

#Standardize the data set

scaler = StandardScaler()

scaler.fit(raw_data)

scaled_features = scaler.transform(raw_data)
print("scaled_features :", scaled_features)
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.columns)

#Standardize the data set

scaler = StandardScaler()

scaler.fit(raw_data_completion)

scaled_features = scaler.transform(raw_data_completion)

scaled_data_completion = pd.DataFrame(scaled_features, columns = raw_data_completion.columns)



#Split the data set into training data and test data

from sklearn.model_selection import train_test_split

x = scaled_data
y= scaled_data_completion
x_training_data, x_test_data,= train_test_split(y, test_size = 0.2)
print("x : ", x)
print("y : ", y)
model_basic = load(open(r'E:\restapi_sky\apps\sky_construction_Basic.pkl', 'rb'))
predictions_basic = model_basic.predict(x)
model_intermediate = load(open(r'E:\restapi_sky\apps\sky_construction_Intermediate.pkl', 'rb'))
predictions_intermediate = model_intermediate.predict(x)
model_advanced = load(open(r'E:\restapi_sky\apps\sky_construction_Advanced.pkl', 'rb'))
predictions_advanced = model_advanced.predict(x)
model_completion = load(open(r'E:\restapi_sky\apps\sky_Completion.pkl', 'rb'))
predictions_completion = model_completion.predict(y)
def completion(result):
    
    if (result == 1):
        return "True"
    else:
        return "False"
print ("No of resources required for Basic Level :", predictions_basic[-1])
print ("No of resources required for Intermediate Level :", predictions_intermediate[-1])
print ("No of resources required for Advanced Level :", predictions_advanced[-1])
print ("Can the construction be completed :", completion(predictions_completion[-1]))
