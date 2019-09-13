# support/BatchGenerator.py

import numpy as np


# Batch Generator

class BatchGenerator: # mini-Batch gradient descent # could also be bootsrapped

    def __init__(self, x, y, batch_size):
        self.x_train = x
        self.y_train = y
        self.batch_size = batch_size
        self.num_examples = self.x_train.shape[0]
        self.k = 0  # current batch index

    def next(self):
        first_index = self.k * self.batch_size
        last_index = (self.k + 1) * self.batch_size
        if last_index <= self.num_examples:
            batch_x = self.x_train[first_index:last_index]
            batch_y = self.y_train[first_index:last_index]
            if last_index == self.num_examples:
                self.k = 0
            else:
                self.k += 1
        else:
            batch_x = self.x_train[first_index:]
            batch_y = self.y_train[first_index:]
            self.k = 0
        return batch_x, batch_y
