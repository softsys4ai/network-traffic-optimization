from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import entropy 
import scipy.stats 
import scipy

def kl(p,q):
    p=np.asarray(p,dtype=np.float)
    q=np.asarray(q,dtype=np.float)
    return np.sum(np.where(p!=0,p*np.log(p/q),0))

