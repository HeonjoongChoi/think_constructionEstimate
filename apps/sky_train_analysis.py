#Common imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pickle import dump
from pickle import load



#Import the data set

raw_data = pd.read_csv('construction_resources_13.csv', index_col = 0)
raw_data.columns

raw_data = raw_data[[' Construction_Scale ','PersonCount_Intermediate','Equipment1', 'Equipment2', 'Equipment3',
       'Equipment4', 'Equipment5', 'Equipment6', 'Equipment7', 'Equipment8',
       'Equipment9', 'Equipment10', 'Equipment11', 'Equipment12',
       'Equipment13', 'Equipment14', 'Equipment15', 'Equipment16',
       'Equipment17']]

#Import standardization functions from scikit-learn

from sklearn.preprocessing import StandardScaler

#Standardize the data set

scaler = StandardScaler()

scaler.fit(raw_data.drop("PersonCount_Intermediate", axis=1))

scaled_features = scaler.transform(raw_data.drop("PersonCount_Intermediate", axis=1))

scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop("PersonCount_Intermediate", axis=1).columns)

#Split the data set into training data and test data

from sklearn.model_selection import train_test_split

x = scaled_data

y = raw_data["PersonCount_Intermediate"]

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)

#Train the model and make predictions

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_training_data, y_training_data)
# save the model
dump(model, open('sky_construction.pkl', 'wb'))


model = load(open('sky_construction.pkl', 'rb'))
predictions = model.predict(x_test_data)

#Performance measurement

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print(classification_report(y_test_data, predictions))

print(confusion_matrix(y_test_data, predictions))

