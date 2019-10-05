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

X = np.concatenate([transformed[:, 1:3], X[:, :1], transformed[:, -2:-1], X[:, 3:]], axis=1)

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
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(6, input_dim = 11, activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(6, activation = 'relu' ))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, activation = 'sigmoid' ))    
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

print("----------")
print("Make prediction")

# Predicting a single new observation

single_prediction_array = np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]]) # Double square brackets to create horizontal row of values,
# Apply same scale as training set
scaled_single_prediction = sc.transform(single_prediction_array)
new_prediction = classifier.predict(scaled_single_prediction)

new_prediction = (new_prediction > 0.5)
print(new_prediction) # returns false, customer won't leave

'''
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():    
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, activation = 'relu'))
    classifier.add(Dense(6, activation = 'relu' ))
    classifier.add(Dense(1, activation = 'sigmoid' ))    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()
'''
# Dropout: https://www.udemy.com/deeplearning/learn/lecture/6743788#questions
# Solution to overfitting: Dropout regularization
# See dropout in classifier.add(Dropout()) 

# Parameter tuning: https://www.udemy.com/deeplearning/learn/lecture/6743798#questions