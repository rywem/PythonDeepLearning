# -*- coding: utf-8 -*-



# Source: https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944

import pandas as pd
import numpy as np

timesteps = 3
batch_size = 2
testing_records = 20
epochs = 20
fileName = 'GE'
dataset_csv = fileName + '.csv'
networkFileName = fileName + '.h5'
loadNetwork = False
saveNetwork = False
comparison_timestep = 1  # The timestep that is being compared to our target date. 1 would be the previous day
comparison_column = 4 # The column if comparison
selected_columns = ['Close','High', 'Low', 'Results']
lstm_units = 50


def build_timeseries(mat, y_col_index, TIME_STEPS):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

dataset_all = pd.read_csv('./datasets/'+dataset_csv)
#dataset_all["Results"] = [0] * dataset_all.shape[0]

dataset_all.tail()

from matplotlib import pyplot as plt
plt.figure()
plt.plot(dataset_all["Open"])
plt.plot(dataset_all["High"])
plt.plot(dataset_all["Low"])
plt.plot(dataset_all["Close"])
plt.title(fileName + ' stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()

plt.figure()
plt.plot(dataset_all["Volume"])
plt.title(fileName + ' stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()

print("checking if any null values are present\n", dataset_all.isna().sum())

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open","High","Low","Close","Volume"]
df_train, df_test = train_test_split(dataset_all, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])



x_t, y_t = build_timeseries(x_train, 3, timesteps)
x_t = trim_dataset(x_t, batch_size)
y_t = trim_dataset(y_t, batch_size)
x_temp, y_temp = build_timeseries(x_test, 3, timesteps)
x_val, x_test_t = np.split(trim_dataset(x_temp, batch_size),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, batch_size),2)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model 
#from keras import optimizers
#from keras.optimizers import lr
#from keras.optimizers import RMSprop

lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(batch_size, timesteps, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,     kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
#optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics = ['accuracy'])

history = lstm_model.fit(x_t, y_t, epochs=epochs, verbose=2, batch_size=batch_size,
                    shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                    trim_dataset(y_val, batch_size)))

predicted_stock_price = lstm_model.predict( x_test)
#inverse scaling to get stock price
predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)
# Visualizing the results

plt.plot(real_stock_price, color = 'red', label = 'Real ' + fileName +' Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ' + fileName +' Stock Price')
plt.title(fileName + ' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
