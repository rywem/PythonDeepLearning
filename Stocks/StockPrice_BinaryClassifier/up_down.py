# -*- coding: utf-8 -*-

'''
A neural network that analyzes pricing trends and tries to predict
whether a stock's price will be up or down on a given day.
'''

# Setup Session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()

from keras import backend as K
K.set_session(sess)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_training_test(all_dataframe, test_record_count, timesteps):
    ''' A function to split a dataframe into a training and a test set'''
    df_training = all_dataframe.iloc[:all_dataframe.shape[0]-test_record_count, :]
    #df_test = all_dataframe.iloc[all_dataframe.shape[0]-test_record_count:, :]
    df_test= all_dataframe[all_dataframe.shape[0] - test_record_count - timesteps:]
    return df_training, df_test

timesteps = 60
batch_size = 32
testing_records = 20
epochs = 20
fileName = 'SPY'
dataset_csv = fileName + '.csv'
networkFileName = fileName + '.h5'
loadNetwork = False
saveNetwork = False
comparison_timestep = 1  # The timestep that is being compared to our target date. 1 would be the previous day
comparison_column = 4 # The column if comparison
selected_columns = ['Close','High', 'Low', 'Results']
lstm_units = 50
dataset_all = pd.read_csv('./datasets/'+dataset_csv)
dataset_all["Results"] = [0] * dataset_all.shape[0]


# Generate less than / greater than column, 0 = less than or equal,  1 = greater than
for i in range(comparison_timestep, dataset_all.shape[0]):
    first = dataset_all.iloc[i - comparison_timestep, comparison_column]
    second = dataset_all.iloc[i, comparison_column]
    result = 0    
    if first < second:
        result = 1    
    dataset_all.at[i, "Results"] = result

# Remove the rows without accurate comparison_timesteps
drop_array = []

for i in range(comparison_timestep):
    drop_array.append(i)

dataset_all_comparison_removed = dataset_all.drop( drop_array )
# Create a new dataframe selecting only the desired columns
dataset_all_final = dataset_all_comparison_removed[selected_columns].copy()

# Create our training and test sets
dataset_training, dataset_test = create_training_test(dataset_all_final, testing_records, timesteps)

training_set = dataset_training.iloc[:,:].values
test_set = dataset_test.iloc[:, :].values

# Scale our data
from sklearn.preprocessing import MinMaxScaler 
scale = MinMaxScaler(feature_range = (0, 1))


training_set_scaled = scale.fit_transform(training_set)

X_train = []
y_train = []

for i in range(timesteps, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i - timesteps : i, :])
    y_train.append(training_set_scaled[i, -1])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(selected_columns))) #possibly revisit


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model 
import time


# Initialising the RNN
classifier = Sequential() # predicting a continuous output, therefore doing regression
# Adding the first LSTM layer and some Dropout regularization
classifier.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (timesteps, len(selected_columns))))
classifier.add(Dropout(0.2))
# Adding a second LSTM layer with Dropout regularization
classifier.add(LSTM(units = lstm_units, return_sequences = True))
classifier.add(Dropout(0.2))
# Adding a third LSTM layer with Dropout regularization
classifier.add(LSTM(units = lstm_units, return_sequences = True))
classifier.add(Dropout(0.2))
# Adding a fourth LSTM layer with Dropout regularization
classifier.add(LSTM(units = lstm_units, return_sequences = False))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
# Fit RNN to training set or Load network
if loadNetwork == True:
    classifier = load_model(networkFileName)
    print("Loaded: ", networkFileName)
else:    
    with tf.device('/gpu:0'):    
        start = time.time()
        classifier.fit(X_train, y_train, epochs = epochs, batch_size = batch_size )
        elapsed = time.time() - start
        print("Elapsed seconds: ", elapsed)        
    if saveNetwork == True:
        classifier.save(networkFileName)  

test_set_scaled = scale.transform(test_set)


X_test = []
y_test = []
for i in range(timesteps, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-timesteps:i, :])
    y_test.append(test_set_scaled[i, -1])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(selected_columns)))

y_test = np.array(y_test)

scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred_values = classifier.predict(X_test)

y_pred = (y_pred_values > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
correct_predictions = cm[0][0] + cm[1][1]
incorrect_predictions = cm[1][0] + cm[0][1]
total = correct_predictions + incorrect_predictions
print("Pct Correct: ", correct_predictions / total)
print("Pct Incorrect: ", incorrect_predictions / total)



