# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


data = pd.read_csv('OneHotEncoding2.csv')
X = data.iloc[:, :].values
y = data.iloc[:, 2].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(handle_unknown='ignore')

transformed = onehotencoder.fit_transform(X[:, [1]]).toarray()

#for i in X[:, :1]:
#    print(i)

X1 = np.concatenate([X[:, :1], transformed, X[:, 2:3]], axis=1)

