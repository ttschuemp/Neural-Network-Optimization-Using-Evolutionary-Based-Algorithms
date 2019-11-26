#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:56:18 2019

@author: tobiastschuemperlin
"""

WeightsHat_0 = nnHat.layers[0].weights
WeightsHat_1 = nnHat.layers[2].weights
WeightsHat_2 = nnHat.layers[4].weights
Weights_0 = nn.layers[0].weights
Weights_1 = nn.layers[2].weights
Weights_2 = nn.layers[4].weights
print(WeightsHat_0)
print(Weights_0)



rowsumHat_0 = np.sum(WeightsHat_0, axis = 1)
print(WeightsHat_0.shape)
print(rowsumHat_0)
rowsum_0 = np.sum(Weights_0, axis = 1)
print(Weights_0.shape)
print(rowsum_0)

rowsumHat_2 = np.sum(WeightsHat_2, axis = 0)
print(WeightsHat_2.shape)
print(rowsumHat_2)
rowsum_2 = np.sum(Weights_2, axis = 0)
print(Weights_2.shape)
print(rowsum_2)