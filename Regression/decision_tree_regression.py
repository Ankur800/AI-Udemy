# IMPORTING THR LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING THE DATASET
dataset = pd.read_csv("/home/ankur/Udemy ML/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING THE DECISION TREE REGRESSION MODEL ON THE WHOLE DATASET
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# PREDICTING A NEW RESULT
print(regressor.predict([[6.5]]))

# VISUALISING THE DECISION TREE REGRESSION RESULTS
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()