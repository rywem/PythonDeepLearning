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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
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
from keras.layers import Dropout
'''
# Initialising the ANN
classifier = Sequential();

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))
classifier.add(Dropout(rate = 0.1))

# Part 3 - Making the predictions and evaluating the model
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))

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

print("----------")
print("Make prediction")

# Predicting a single new observation

single_prediction_array = np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]]) # Double square brackets to create horizontal row of values,
# Apply same scale as training set
scaled_single_prediction = sc.transform(single_prediction_array)
new_prediction = classifier.predict(scaled_single_prediction)

new_prediction = (new_prediction > 0.5)
print(new_prediction) # returns false, customer won't leave


# Evaluating the ANN
# Implement k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():    
    classifier = Sequential();        
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 2)

mean = accuracies.mean()
variance = accuracies.std()
'''
# Dropout: https://www.udemy.com/deeplearning/learn/lecture/6743788#questions
# Solution to overfitting: Dropout regularization
# See dropout in classifier.add(Dropout()) 

# Parameter tuning: https://www.udemy.com/deeplearning/learn/lecture/6743798#questions
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
#alternative from sklearn.grid_search import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential();        
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#Grid search
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


