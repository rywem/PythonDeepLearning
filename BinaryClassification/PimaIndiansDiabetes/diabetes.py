# -*- coding: utf-8 -*-


# Source: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input (x) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #this defines the input layer and the first hidden layer
model.add(Dense(8, activation='relu')) #defines the second hidden layer
model.add(Dense(1, activation='sigmoid')) #defines the output layer
# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make probabiliy predictions with the model
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]

# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
