"""-----------------------------------------------------------------------------
---------------------------LSTM: Long Short Term Memory-------------------------
--------------------------------------------------------------------------------
"""
import re
import sys
import time
import math
import keras
import itertools
import pandas as pd
import numpy as np

from math import sqrt
from numpy.random import seed
from numpy import concatenate
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import Activation, Dropout
import keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

seed(16)

class Preprocess(object):
    """-------------------------------------------------------------------------
    This class is used to perform operations on data
    ----------------------------------------------------------------------------
    """
    def __init__(self):
        print ("initializing preprocess class")
        self.data_dim=24
        self.batch_size=1
        self.timesteps=8
        self.n_features=1
        self.bounds={"L1": [[1, 100], [5, 100], [0, 1]],
                     "L2":  [[1, 100], [1, 100], [5, 100], [0, 1]],
                     "L3":  [[1, 100], [1, 100], [1, 100], [5, 100], [0, 1]],
                     "L4":  [[1, 100], [1, 100], [1, 100], [1, 100], [5, 100], [0, 1]]
        }

    def get_data(self, link):
        """This function is used to get data"""
        # get data
        df = pd.read_csv('./data/Trans_atlantic.csv', header=None)
        df = df.iloc[1:]
        df = df.rename(columns = {0: "Time", int(link):"Out"})
        dates = df['Time'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M'))
	    # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        traffic_scaler = MinMaxScaler(feature_range=(0,1))
        return (df, scaler, traffic_scaler)

    def process_data(self, df, scaler,
                 traffic_scaler, print_shapes = True):
        """This function is used to process train and test data"""

        X = df.iloc[:,1].values
        Y = df.iloc[:,1].shift(1).fillna(0).values

        # normalize
        X = scaler.fit_transform(X.reshape(-1,1))
        Y = traffic_scaler.fit_transform(Y.reshape(-1,1))
        c=np.array(Y)
        np.savetxt("./data/real-Y1.csv", c, delimiter=',')
        X = X.reshape(X.shape[0], 1, 1)
        Y = Y.reshape(Y.shape[0], 1)

        # train-test split
        X_train = X[:2112]
        Y_train = Y[:2112]
        X_val = X[2112:2612]
        Y_val = Y[2112:2612]
        X_test=X[2612:]
        Y_test=Y[2612:]
        return (scaler, traffic_scaler, X_train,
               Y_train, X_val, Y_val,
               X_test, Y_test)


class Architecture(object):
    """-------------------------------------------------------------------------
    This class is used to construct LSTM architectures
    ----------------------------------------------------------------------------
    """
    def __init__(self):
        print ("initializing LSTM architecture")

    def size(self, model):
        """This function is used to get the size of te model"""
        return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

    def one_layer_lstm(self, num_neurons, num_epochs,
                       dropout, X_train, Y_train,
                       X_val, Y_val):
        """This function is used to construct a 1-layer LSTM"""

        model = Sequential()
	    # layer: 1
        model.add(LSTM(num_neurons, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        print(model.summary())
        size = self.size(model)
        
        hist=model.fit(X_train, Y_train, epochs=num_epochs,
                       batch_size=1, shuffle=False, validation_data = (X_val, Y_val),
                       verbose = 0)
        return model, size

    def two_layer_lstm(self, num_neurons_layer_1, num_neurons_layer_2,
                       num_epochs, dropout, X_train,
                       Y_train, X_val, Y_val):
        """This function is used to construct a 2-layer LSTM"""
        model = Sequential()
        # layer: 1
        model.add(LSTM(num_neurons_layer_1, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        # layer: 2
        model.add(LSTM(num_neurons_layer_2, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        print(model.summary())
        size = self.size(model)
        hist=model.fit(X_train, Y_train, epochs=num_epochs,
                       batch_size=1, shuffle=False, validation_data = (X_val, Y_val),
                       verbose = 0)
        return model, size

    def three_layer_lstm(self, num_neurons_layer_1, num_neurons_layer_2,
                       num_neurons_layer_3, num_epochs, dropout,
                       X_train, Y_train, X_val, Y_val):
        """This function is used to construct a 3-layer LSTM"""
        model = Sequential()
        # layer: 1
        model.add(LSTM(num_neurons_layer_1, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        # layer: 2
        model.add(LSTM(num_neurons_layer_2, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        # layer: 3
        model.add(LSTM(num_neurons_layer_3, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        print(model.summary())
        size = self.size(model)
        hist=model.fit(X_train, Y_train, epochs=num_epochs,
                       batch_size=1, shuffle=False, validation_data = (X_val, Y_val),
                       verbose = 0)
        return model, size

    def four_layer_lstm(self, num_neurons_layer_1, num_neurons_layer_2,
                       num_neurons_layer_3, num_neurons_layer_4, num_epochs,
                       dropout, X_train, Y_train,
                       X_val, Y_val):
        """This function is used to construct a 4-layer LSTM"""
        model = Sequential()

        # layer: 1
        model.add(LSTM(num_neurons_layer_1, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))

        model.add(Dropout(dropout))
        # layer: 2
        model.add(LSTM(num_neurons_layer_2, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        # layer: 3
        model.add(LSTM(num_neurons_layer_3, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
         # layer: 4
        model.add(LSTM(num_neurons_layer_4, activation='tanh', return_sequences=True,
                       batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        print(model.summary())
        size = self.size(model)
        hist=model.fit(X_train, Y_train, epochs=num_epochs,
                       batch_size=1, shuffle=False, validation_data = (X_val, Y_val),
                       verbose = 0)
        return model, size

    def evaluate(self, model, X_test,
                 Y_test, scaler):
        """This function is used to evaluate the LSTM model"""

        start = time.time()
        yhat = model.predict(X_test,1)
        it = time.time() - start
        # invert scaling for predictions
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
        inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        # invert scaling for actual
        Y_test = Y_test.reshape((len(Y_test), 1))
        inv_y = concatenate((Y_test, X_test[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        # calculate RMSE
        err = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: {0}'.format(err))
        print ('End Time: {0}'.format(it))
        return err, it
