# -*- coding: utf-8 -*-
import time
start = time.time()
# Part 1 - Building the Convolutional Neural Network
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)
# Part 1 - Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the training set
dataset_train = pd.read_csv('./datasets/google/Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('./datasets/google/Google_Stock_Price_Test.csv')

training_set = dataset_train.iloc[:, 1:2].values #.values method creates a numpy array

# Apply feature scaling, 
# Either Standardisation  =  xstand = (x - mean(x)) / st dev(x)
# or Normalisation = xnorm = (x - min(x)) / (max(x) - min(x)))
# We will use normalisation. 
from sklearn.preprocessing import MinMaxScaler 

scale = MinMaxScaler( feature_range = (0, 1))
# Apply scaler on data
training_set_scaled = scale.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
## Always use the reshape function when adding a dimension to the numpy array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# TODO: Add additional indicators
    
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential() # predicting a continuous output, therefore doing regression
# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

with tf.device('/gpu:0'):    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    

# Part 3 - Making the predictions and visualizing the results
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #training set plus test set
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)
inputs = scale.transform(inputs)

X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
#inverse scaling to get stock price
predicted_stock_price = scale.inverse_transform(predicted_stock_price)
# Visualizing the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()