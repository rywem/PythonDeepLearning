# -*- coding: utf-8 -*- 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown='ignore')
transformed = onehotencoder.fit_transform(X[:, [1,2]]).toarray()
X = np.concatenate([transformed, X[:, :1], X[:, 3:]], axis=1)
X = X[:, 1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#importing keras libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, input_dim = 12, activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu' ))

# Part 3 - Making the predictions and evaluating the model
classifier.add(Dense(1, activation = 'sigmoid' ))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Set threshold for true or false
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
correct_predictions = cm[0][0] + cm[1][1]
incorrect_predictions = cm[1][0] + cm[0][1]
total = correct_predictions + incorrect_predictions
print("Pct Correct: ", correct_predictions / total)
print("Pct Incorrect: ", incorrect_predictions / total)