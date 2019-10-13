# -*- coding: utf-8 -*-

# Part 1 - Building the Convolutional Neural Network
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
sess = tf.Session()
#sess = tf.compat.v1.Session()

from keras import backend as K
K.set_session(sess)
# Part 1 - Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


timesteps = 80
testing_records = 20 
feature_scaling = 'standardisation' #"standardization | normalization" 
optimizer = 'adam' #ex: adam, rmsprop (recommended for rnn), SFD, adadelta
batch_size = 32
epochs = 100
loss = '' # mean_squared_error
networkFileName = 'spy.h5'
loadNetwork = False
saveNetwork = True
dataset_csv = "SPY.csv"

def create_training_test(all_dataframe, test_record_count):
    ''' A function to split a dataframe with all data into training and test set dataframes '''
    df_training = all_dataframe.iloc[:all_dataframe.shape[0]-test_record_count, :]
    df_test = all_dataframe.iloc[all_dataframe.shape[0]-test_record_count:, :]
    return df_training, df_test

# Import data sets, split into training and test sets
dataset_all = pd.read_csv('./datasets/'+dataset_csv)
dataset_training, dataset_test = create_training_test(dataset_all, testing_records)

training_set = dataset_training.iloc[:,1:7].values
###training_set = dataset_training.iloc[:, 4:5].values
test_set = dataset_test.iloc[:,1:7].values
###test_set = dataset_test.iloc[:, 4:5].values

# Apply scaling 
from sklearn.preprocessing import MinMaxScaler 
scale = MinMaxScaler(feature_range = (0, 1))

# Apply scale object on data
training_set_scaled = scale.fit_transform(training_set)

X_train = [] 
y_train = []


for i in range(timesteps, training_set_scaled.shape[0]):
    ###X_train.append(training_set_scaled[i-timesteps:i, 0])
    ###y_train.append(training_set_scaled[i, 0])
    X_train.append(training_set_scaled[i-timesteps:i, 0:6])    
    y_train.append(training_set_scaled[i, 3:4]) #y_train.append(i, 3:4 ) or y_train.append(i, 0:4 ) ?, #79, 9:53

X_train, y_train = np.array(X_train), np.array(y_train)

# Add indicators use reshape #80
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, 6))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
import time
# Initialize the RNN
#def build_regressor():
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (timesteps, 6))) # #82. units = # lstm cells(neurons), 
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = False)) #last LSTM layer needs return_sequences = False
regressor.add(Dropout(0.2))
# Add output layer
regressor.add(Dense(units = 1))
regressor.compile(optimizer = optimizer, loss = 'mean_squared_error' )
# Fit RNN to training set or Load network
if loadNetwork == True:
    regressor = load_model(networkFileName)
    print("Loaded: ", networkFileName)
else:
    start = time.time()
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size )
    elapsed = time.time() - start
    print("Elapsed seconds: ", elapsed)        
    if saveNetwork == True:
        regressor.save(networkFileName)  

dataset_total = dataset_all[len(dataset_all) - (timesteps + testing_records):]
inputs = dataset_total.iloc[:, 1:7].values
#inputs = np.reshape(-1, 1)
inputs = scale.transform(inputs)

X_test = [] 
#y_test = []
for i in range(timesteps, inputs.shape[0]):
    X_test.append(inputs[i-timesteps:i, 0:6])    
    #y_test.append(inputs[i, 3:4])



y_scale = MinMaxScaler(feature_range = (0, 1))
y_training_set_scaled = y_scale.fit_transform(training_set[:, 3:4])
    
X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], timesteps, 6))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))
predicted_price = regressor.predict(X_test)
predicted_price = y_scale.inverse_transform(predicted_price)

real_stock_price = dataset_test.iloc[:, 3:4].values
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Stock Price' )
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


#with tf.device('/gpu:0'): 