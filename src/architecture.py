"""-----------------------------------------------------------------------------
-------------------------------- LSTM Architecture -----------------------------
--------------------------------------------------------------------------------
"""
import os
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
        return -err, it
