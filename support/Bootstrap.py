# support.Bootstrap.py

import numpy as np
import pandas as pd

def bootstrap(data, fracIS = 0.7, replacement = True): 
    fracOS = 1-fracIS
    data_train = data.sample(frac=fracIS, replace=replacement, axis=0)
    data_test = data.sample(frac=fracOS, replace=replacement, axis=0)
    return data_train, data_test
