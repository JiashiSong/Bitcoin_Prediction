import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import CSVLogger



np.random.seed(12345)


label_data = pd.read_csv("data/output.csv").values
state_data = pd.read_csv("data/input.csv").values

print("Data has been loaded")

X = label_data
Y = state_data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12345)

model = Sequential()
model.add(Dense(1000, input_dim=6, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(6))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])


history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=5000, verbose=2)
model.save('MLP_model.h5')

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


y_pred = model.predict(X_test)
print("MSE is %.2f", mean_squared_error(y_test, y_pred))

pyplot.figure(7)
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('MSE')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.savefig('history_Loss')




pyplot.figure(1)
pyplot.plot(y_test[1:300, 0])
pyplot.plot(y_pred[1:300, 0])
pyplot.savefig('input0_1.png')

pyplot.figure(2)
pyplot.plot(y_test[1:300, 1])
pyplot.plot(y_pred[1:300, 1])
pyplot.savefig('input1_1.png')

pyplot.figure(3)
pyplot.plot(y_test[1:300, 2])
pyplot.plot(y_pred[1:300, 2])
pyplot.savefig('input2_1.png')

pyplot.figure(4)
pyplot.plot(y_test[1:300, 3])
pyplot.plot(y_pred[1:300, 3])
pyplot.savefig('input3_1.png')

pyplot.figure(5)
pyplot.plot(y_test[1:300, 4])
pyplot.plot(y_pred[1:300, 4])
pyplot.savefig('input4_1.png')

pyplot.figure(6)
pyplot.plot(y_test[1:300, 5])
pyplot.plot(y_pred[1:300, 5])
pyplot.savefig('input5_1.png')


