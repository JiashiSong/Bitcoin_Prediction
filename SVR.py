import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(12345)


label_data = pd.read_csv("data/output.csv", header=None).values
state_data = pd.read_csv("data/input.csv", header=None).values

X = label_data[1:10000, ]
Y = state_data[1:10000, ]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=max_depth,
                                                          random_state=0, n_jobs=8))

regr_multirf.fit(X_train, y_train)
y_multirf = regr_multirf.predict(X_test)


print("The SVR MSE is: {}".format(mean_squared_error(y_test, y_multirf)))


