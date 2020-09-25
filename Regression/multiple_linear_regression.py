# IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATASET
dataset = pd.read_csv("/home/ankur/Udemy ML/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ENCODING CATEGORICAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# SPLITTING THE DATASET INTO THE TRAINING SET AND DATA SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# TRAINING THE MULTIPLE LINEAR REGRESSION MODEL ON THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST RESULTS
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)    # setting precision
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))