from __future__ import division
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from src.preprocess import Preprocess
from src.architecture import Architecture
from src.visualize import Visualize

def get_Y(cur_arch, ARCH, X_train,
          Y_train, X_val, Y_val,
          X_test, Y_test, config,
          scaler):
    """Thus function is used for getting the measurements"""
    if cur_arch == "L1":
        (num_neurons, num_epochs, dropout) = config
        model, size = ARCH.one_layer_lstm(num_neurons, num_epochs, dropout,
                                    X_train, Y_train, X_val,
                                    Y_val)
    elif cur_arch == "L2":
        (num_neurons_layer_1, num_neurons_layer_2, num_epochs,
         dropout) = config
        model, size = ARCH.two_layer_lstm(num_neurons_layer_1, num_neurons_layer_2, num_epochs,
                                    dropout, X_train, Y_train,
                                    X_val, Y_val)
    elif cur_arch == "L3":
        (num_neurons_layer_1, num_neurons_layer_2, num_neurons_layer_3,
         num_epochs, dropout) = config
        model, size = ARCH.three_layer_lstm(num_neurons_layer_1, num_neurons_layer_2, num_neurons_layer_2,
                                      num_epochs, dropout, X_train,
                                      Y_train, X_val, Y_val)
    elif cur_arch == "L4":
        (num_neurons_layer_1, num_neurons_layer_2, num_neurons_layer_3,
         num_neurons_layer_4, num_epochs, dropout) = config
        model, size = ARCH.four_layer_lstm(num_neurons_layer_1, num_neurons_layer_2, num_neurons_layer_3,
                                     num_neurons_layer_4, num_epochs, dropout,
                                     X_train, Y_train, X_val,
                                     Y_val)

    else:
        print ("[ERROR]: LSTM architecture not supported")

    err, it = ARCH.evaluate(model, X_test, Y_test,
                           scaler)
    return err, it, size

def round_config(config):
    """This function is used to round config options"""
    config[-1]=round(config[-1],1)
    for option in range(len(config)-1):
        config[option]=int(round(config[option]))
    return config

def standardize_config(X, mode):
    """This function is used to standardize config
    """

    if mode=="forward":
        for i in range(len(X)):
            for j in range(len(X[i])-2,-1,-1):
                X[i][j]=X[i][j]/100

    elif mode=="backward":
            for i in range(len(X)-2,-1,-1):
                X[i]=X[i]*100
    else:
        print ("[ERROR]: mode not supported")
    return X

def initialize(bounds):
    """This function is used to initialize"""
    X=list()
    for conf in range(20):
        temp=np.random.uniform(bounds[:,0],bounds[:,1],size=(1,bounds.shape[0]))
        temp=temp.tolist()[0]
        X.append(round_config(temp))
    return X

def hypervolume_improvement(X, X_sample, Y_sample,
                            gpr, xi=0.01):
    """computes the HI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model."""
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        hi = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        hi[sigma == 0.0] = 0.0

    return hi

def propose_location(acquisition, X_sample,
                     Y_sample, gpr, bounds,
                     n_restarts=25):
    """Proposes the next sampling point by optimizing the acquisition function."""
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    def min_obj(X):
        # minimization of objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)

if __name__=="__main__":
    with open("config.yaml","r") as fp:
        conf=yaml.load(fp)
        conf=conf["config"]

    PREP = Preprocess(conf)
    # get bounds
    cur_link, cur_arch = sys.argv[1], sys.argv[2]
    # output file
    fpath = os.path.join(os.getcwd(), conf["fnames"]["output"])
    fname = str(cur_link)+"_"+str(cur_arch)+".csv"
    foutput = os.path.join(fpath, fname)

    bounds = np.array(PREP.bounds[cur_arch])
    # get data
    (df, scaler, traffic_scaler) = PREP.get_data(cur_link)
    # create train, validation and test data
    (scaler, traffic_scaler, X_train,
    Y_train, X_val, Y_val,
    X_test, Y_test) = PREP.process_data(df[['Time', 'Out']], scaler, traffic_scaler)
    output=[]
    ARCH = Architecture()
    # get initial configs
    X_init = initialize(bounds)
    Y_init = []
    # get initial measurements
    
    for config in X_init:
        cur_err, cur_inf_time, cur_size = get_Y(cur_arch, ARCH, X_train,
                                                Y_train, X_val, Y_val,
                                                X_test, Y_test, config,
                                                scaler)

        Y_init.append([cur_err])
        output.append([config, cur_err, cur_inf_time,
                       cur_size, cur_arch, cur_link])


    # store initial data
    df = pd.DataFrame(output)
    df.columns= ["config","err","inf_time",
                 "size", "arch", "link"]
    df.to_csv(foutput, index=False)

    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel = m52, n_restarts_optimizer=20)
    # initialize samples
    X_sample = np.array(X_init[:])
    Y_sample = np.array(Y_init[:])
    # BO loop
    n_iter = 200
    bo_start = time.time()
    for i in range(n_iter):
        # update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(hypervolume_improvement, X_sample, Y_sample,
                                  gpr, bounds)
        # Obtain next sample from the objective function
        X_next = [x[0] for x in X_next]
        X_next = round_config(X_next)
        Y_next, next_inf_time, next_size = get_Y(cur_arch, ARCH, X_train,
                                                 Y_train, X_val, Y_val,
                                                 X_test, Y_test, X_next,
                                                 scaler)

        cur_df = pd.DataFrame([[X_next, Y_next, next_inf_time,
                              next_size, cur_arch, cur_link]])
        Y_next = [Y_next]
        # add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
        # add current config and measurement to csv file
        with open(foutput,"a") as f:
            cur_df.to_csv(f, header=False, index=False)
    bo_end = time.time()-bo_start
    print ("Time : {0}".format(bo_end))
    # visualize
    #Visualize(foutput)
    
