# support.Bootstrap.py
import numpy as np

def bootstrap(data, train = 0.6, validation = 0.2, test = 0.2, replacement = True): 
    data_train = data.sample(frac=train, replace=replacement, axis=0)
    data_validation = data.sample(frac=validation, replace=replacement, axis=0)
    data_test = data.sample(frac=test, replace=replacement, axis=0)
    return data_train, data_validation, data_test


def standardize(data):
    _,c = data.shape
    data_standardized = data.copy(deep=True)
    for j in range(c):
        x = data_standardized.iloc[:, j]
        avg = x.mean()
        std = x.std()
        x_standardized = (x - avg)/ std
        data_standardized.iloc[:, j] = x_standardized
        
def standardize_image(data):
    r,_ = data.shape
    data_standardized = data.copy(deep=True)
    for j in range(r):
        x = data_standardized.iloc[j,:]
        avg = x.mean()
        std = x.std()
        x_standardized = (x - avg)/ std
        data_standardized.iloc[j,:] = x_standardized
        
    return data_standardized

def transformY(data, output):
    n,_ = data.shape
    y = np.zeros((n,output))+0.01
    for i in range(n):
        j = data.iloc[i,0] 
        if j == 2:
             y[i,1] = 0.99
        if j == 1: 
            y[i,0] = 0.99
    return y

def transformY_mnist(data, output):
    n,_ = data.shape
    y = np.zeros((n,output))+0.01
    for i in range(n):
        j = int(data.iloc[i,0])
        y[i,j] = 0.99
    return y