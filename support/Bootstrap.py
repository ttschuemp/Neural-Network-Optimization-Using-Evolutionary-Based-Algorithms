# support.Bootstrap.py


def bootstrap(data, train = 0.6, validation = 0.2, test = 0.2, replacement = True): 
    data_train = data.sample(frac=train, replace=replacement, axis=0)
    data_validation = data.sample(frac=validation, replace=replacement, axis=0)
    data_test = data.sample(frac=test, replace=replacement, axis=0)
    return data_train, data_validation, data_test


