# -*- coding: utf-8 -*-

# Data Source: https://jamesmccaffrey.wordpress.com/2018/03/14/datasets-for-binary-classification/

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('data_banknote_authentication.csv', delimiter=',')

# split into input (x) and output (y) variables
X = dataset[:, 0:4]
y = dataset[:, 4]

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu')) #this defines the input layer and the first hidden layer
model.add(Dense(4, activation='relu')) #defines the second hidden layer
model.add(Dense(1, activation='sigmoid')) #defines the output layer

# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=130, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

