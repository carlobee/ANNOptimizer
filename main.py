"""
A program that implements a basic feedforward multilayer Artificial Neural Network

Written By Carl Bettosi and Mamta Sofat
16/11/2020

"""

import numpy as np
import random
import pandas as pd

from PSO import PSO
from particle import particle

#--- INPUT DATA --------------------------------------------------------------------------

input_values_x = []
input_values_y = []

#--- ANN PROPERTIES ----------------------------------------------------------------------

input_neurons = 1
hidden_neurons = 4
output_neurons = 1

#--- PARTICLE SWARM OPTIMIZATION HYPERPARAMS ---------------------------------------------
    
swarmSize = 50
iterations = 100
numberOfInformants = 10
dimensions = (input_neurons * hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons) + output_neurons
cog_constant = 0.5
soc_constant = 0.3

# ranges
bounds=[(-10,10),(-10,10)]
weight_range = (0.0, 1.0)
learning_rate_range = (0.0, 1.0)
inertia_range = (0.9, 0.9)

#--- ACTIVATION FUNCTIONS ---------------------------------------------------------------    

#reference this https://medium.com/towards-artificial-intelligence/building-neural-networks-with-python-code-and-math-in-detail-ii-bbe8accbf3d1
def sigmoid(x):
    return 1/(1+np.exp(-x))

def hyperbolicTangent(x):
    return np.tanh(x)
    
def cosine(x):
    return np.cos(x)
    
def Gaussian(x):
    return np.exp((-x**2)/2)    #check

#--- FEED FORWARD FUNCTIONALITY ----------------------------------------------------------

def feed_forward_train(input_values_x, target_val, weights):

    if isinstance(weights, particle):

        # set weights equal to the current postiion if this particle
        weights = weights.currentPosition
        # define the architecture of the ANN based on its properties
        # reference https://medium.com/@zeeshanahmad10809/train-neural-network-numpy-particle-swarm-optimization-pso-93f289fc8a8e
        layer1_weights  = weights[0 : input_neurons * hidden_neurons].reshape((input_neurons, hidden_neurons))
        layer1_bias     = weights[input_neurons * hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons].reshape((hidden_neurons, ))
        layer2_weights  = weights[(input_neurons * hidden_neurons) + hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons +\
        (hidden_neurons * output_neurons)].reshape((hidden_neurons, output_neurons))
        layer2_bias     = weights[(input_neurons * hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons): (input_neurons *\
        hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons) + output_neurons].reshape((output_neurons, ))
        
        # forward pass calculations
        layer1_output = np.dot(input_values_x, layer1_weights) + layer1_bias
        AF_output = sigmoid(layer1_output)
        layer2_output = np.dot(AF_output, layer2_weights) + layer2_bias
        final = layer2_output
        
        # get the probablilty of the final output
        predicted_val = hyperbolicTangent(final)
        #print(MSE(target_val, predicted_val))
        
        return MSE(target_val, predicted_val)

#--- PREDICT FINAL VALUE ---------------------------------------------------------------

def final_prediction(input_values_x, weights):
        
    # define the architecture of the ANN based on its properties
    # reference https://medium.com/@zeeshanahmad10809/train-neural-network-numpy-particle-swarm-optimization-pso-93f289fc8a8e
    layer1_weights  = weights[0 : input_neurons * hidden_neurons].reshape((input_neurons, hidden_neurons))
    layer1_bias     = weights[input_neurons * hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons].reshape((hidden_neurons, ))
    layer2_weights  = weights[(input_neurons * hidden_neurons) + hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons +\
    (hidden_neurons * output_neurons)].reshape((hidden_neurons, output_neurons))
    layer2_bias     = weights[(input_neurons * hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons): (input_neurons *\
    hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons) + output_neurons].reshape((output_neurons, ))
        
    # forward pass calculations
    layer1_output = np.dot(input_values_x, layer1_weights) + layer1_bias
    AF_output = np.tanh(layer1_output)
    layer2_output = np.dot(AF_output, layer2_weights) + layer2_bias
    final = layer2_output
    

    # get the probablilty of the final output
    predicted_val = sigmoid(final)
        
    return np.argmax(predicted_val, axis=1)

#--- COST FUNCTION ---------------------------------------------------------------------------

#def MSE(target_val, predicted_val):
    #return np.square(np.subtract(target_val, predicted_val)).mean()

def MSE(target_val, predicted_val):
    sum_square_error = 0.0
    for i in range(len(target_val)):
        sum_square_error += (target_val[i] - predicted_val[i])**2.0
    mean_square_error = 1.0 / len(target_val) * sum_square_error
    return mean_square_error


#--- CALCULATE ACCURACY ----------------------------------------------------------------------

def accuracyCalc(true_value, value_prediction):
    return (true_value == value_prediction).mean()

'''
# create weights for input > first hidden layer
print("-------------- INPUT ---------------")
iArray = np.random.random((numberOfInputs, numberOfNeurons))
HiddenLayers[0] = iArray
print(iArray)

# create the weights between the hidden layers
i=-1
for x in range(numberOfLayers-1):
    print("------------- HIDDEN ---------------")
    hArray = np.random.random((numberOfNeurons, numberOfNeurons))
    HiddenLayers[x+1] = hArray
    print(hArray)
    i=x+1

# create the weights from the last hidden layer > output layer
print("------------- OUTPUT ---------------")
oArray = np.random.random((numberOfOutputs, numberOfNeurons))
HiddenLayers[i+1] = oArray
print(oArray)
'''

#--- RUN MAIN ---------------------------------------------------------------------------- 

if __name__ == '__main__':
    
    # read values from file and append to arrays
    data = pd.read_csv("1in_cubic.txt", header=None, delim_whitespace=True)
        
    input_values_x = data[0].to_numpy().reshape(101,1)
    input_values_y = data[1].to_numpy().reshape(101,1)
    
    # initialise swarm
    swarm = PSO(swarmSize, dimensions, weight_range, learning_rate_range, inertia_range, cog_constant, soc_constant, numberOfInformants)
    
    swarm.optimize(feed_forward_train, input_values_x, input_values_y, iterations)




