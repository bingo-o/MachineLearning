import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../datasets/studentscores.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, lin_reg.predict(X_train), color="blue")
plt.show()

plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, lin_reg.predict(X_test), color="blue")
plt.show()