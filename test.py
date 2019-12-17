import numpy as np




#foreward Step 1
# NN_hat
Input_test = np.array([0.5, 0.87])
Input_test=Input_test.reshape(1,2)
W_hat_IH = Populations[8].neuralNetworks[4].layers[0].weights
B_hat_IH = Populations[8].neuralNetworks[4].layers[0].bias

H1_hat = Input_test @ W_hat_IH + B_hat_IH
print('H1_hat=',H1_hat)
# NN_dgp
W_IH = nn.layers[0].weights
B_IH = nn.layers[0].bias

H1 = Input_test @ W_IH + B_IH
print('H1=',H1)

#H1_hat= [[ 1.16383462 -0.20432631  0.01122737 -0.24754152 -0.21662631 -0.17889987]]
#H1= [[-0.22549924 -0.29255339 -0.27280007 -0.11190819 -0.18719476 -0.07241586]]

#foreward Step 2
# NN_hat

W_hat_HO = Populations[8].neuralNetworks[4].layers[2].weights
B_hat_HO = Populations[8].neuralNetworks[4].layers[2].bias

O_hat = H1_hat @ W_hat_HO + B_hat_HO
print('O_hat=',O_hat)

# NN_dgp
W_HO = nn.layers[2].weights
B_HO = nn.layers[2].bias

O = H1_hat @ W_HO + B_HO
print('O=',O)

#O_hat= [[-2.7816225  2.8591216]]
#O= [[ 2.17761575 -2.19372341]]


