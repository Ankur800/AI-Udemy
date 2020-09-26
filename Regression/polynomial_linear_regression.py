# IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATASET
dataset = pd.read_csv("/home/ankur/Udemy ML/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# -----NOTE-----
# here, we want to make our prediction good, so we'll not divide dataset as
# training and testing part, so no SPLITTING step

# TRAINING THE LINEAR REGRESSION MODEL ON WHOLE DATASET
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# TRAINING THE POLYNOMIAL REGRESSION ON THE WHOLE DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)     # degree means 'n' in b0 + b1*x1 + b2*(x1)^2 + ... + bn*(x1)^n
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# VISUALISING LINEAR REGRESSION RESULTS
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# VISUALISING POLYNOMIAL REGRESSION RESULTS
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# VISUALISING POLYNOMIAL REGRESSION RESULTS (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# PREDICTING A NEW RESULT WITH LINEAR REGRESSION
print(lin_reg.predict([[6.5]]))

# PREDICTING A NEW RESULT WITH LINEAR REGRESSION
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))