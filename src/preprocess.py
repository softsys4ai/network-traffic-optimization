"""-----------------------------------------------------------------------------
--------------------------- Preprocess LSTM Input Data -------------------------
--------------------------------------------------------------------------------
"""
import re
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
from sklearn.preprocessing import MinMaxScaler

class Preprocess(object):
    """-------------------------------------------------------------------------
    This class is used to perform operations on data
    ----------------------------------------------------------------------------
    """
    def __init__(self, config):
        print ("initializing preprocess class")
        self.data_dim=24
        self.batch_size=1
        self.timesteps=8
        self.n_features=1
        self.conf=config
        self.finput=os.path.join(os.getcwd(), self.conf["fnames"]["input"])
        self.fcur_proc_input=os.path.join(os.getcwd(), self.conf["fnames"]["cur_proc_input"])
        self.bounds={"L1": [[1, 100], [5, 100], [0, 1]],
                     "L2":  [[1, 100], [1, 100], [5, 100], [0, 1]],
                     "L3":  [[1, 100], [1, 100], [1, 100], [5, 100], [0, 1]],
                     "L4":  [[1, 100], [1, 100], [1, 100], [1, 100], [5, 100], [0, 1]]
        }

    def get_data(self, link):
        """This function is used to get data"""
        # get data
        df = pd.read_csv(self.finput, header=None)
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
        np.savetxt(self.fcur_proc_input, c, delimiter=',')
        X = X.reshape(X.shape[0], 1, 1)
        Y = Y.reshape(Y.shape[0], 1)

        # train-test split
        X_train = X[:self.conf["size"]["train"]["ub"]]
        Y_train = Y[:self.conf["size"]["train"]["ub"]]
        X_val = X[self.conf["size"]["val"]["lb"]:self.conf["size"]["val"]["ub"]]
        Y_val = Y[self.conf["size"]["val"]["lb"]:self.conf["size"]["val"]["ub"]]
        X_test=X[self.conf["size"]["test"]["lb"]:]
        Y_test=Y[self.conf["size"]["test"]["lb"]:]
        return (scaler, traffic_scaler, X_train,
               Y_train, X_val, Y_val,
               X_test, Y_test)

