"""
template for generating data to fool learners (c) 2018 T. Ruzmetov
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273, n_samples=100, n_features=5):
    """
    This function generates data that is suppose to give better performance
    for LinRegLearner than DTLearner. 
    """
    
    np.random.seed(seed)
    Xval_range = 100.0
    noise_std = Xval_range / 50.0  # std for random noise, lower the better for LinReg

    X = np.random.random(size = (n_samples,n_features))*2*Xval_range - Xval_range  #generated random data as ndarray
    bias_unit = np.ones(shape=(n_samples,1))         # generate bias unit(column of ones) 
    noise = np.random.normal(size = n_samples, scale=noise_std) # generate rand noise to add
    coefficients = np.random.random(size = (n_features + 1,))*2*2.0 - 2.0 #generate rand coefficients 
    X = np.append(bias_unit, X, axis=1) # append bias unit as 1st column
    Y = np.dot(X,coefficients)  + noise # targer is X dot coeff + noise or
                                        # Y=a + b*x1 + c*x2 + .... + rand_noise
    return X, Y


def best4DT(seed=1489683273, n_samples=1000, n_features=2):
    """
    This function generates data that is suppose to give better performance
    for DTLearner than LinRegLearner. 
    """
    np.random.seed(seed)    
    Xval_range = 10.0
    noise_std = Xval_range / 50
    X = np.random.random(size = (n_samples,n_features))*2*Xval_range - Xval_range
    noise = np.random.normal(size = n_samples, scale = noise_std)
    coefficients = np.random.random(size = (n_features,))*2*2.0 - 2.0

    Y = np.dot(5*np.cos(X) + X,coefficients)  + noise
    #Y = np.dot(np.log(X),coefficients) + noise
    #Y = np.dot(X**2,coefficients)  + noise

    return X, Y

def author():
    return 'truzmetov3'

if __name__=="__main__":
    print "they call me Ta."
