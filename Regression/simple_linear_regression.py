# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATASET
dataset = pd.read_csv("/home/ankur/Udemy ML/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# SPLITTING THE DATASET INTO THE TRAINING SET AND DATA SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

# VISUALISING THE TRAIN SET RESULTS
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# VISUALISING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction
print(regressor.predict([[12]]))

# Getting final linear regression equation with the values of coefficients
print(regressor.coef_)          # m
print(regressor.intercept_)     # c

# y = mx + c