import numpy as np
import pandas as pd

dataset = pd.read_csv("../datasets/Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion="entropy", random_state=1)
dt_clf.fit(X_train, y_train)

y_predict = dt_clf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))