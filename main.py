"""
main.py

Contains functionality for generating architecture of neural net with its parameters. Contains hyperparameters for initialising PSO algorithm.

Written By Carl Bettosi and Mamta Sofat
23/11/2020

"""

import numpy as np
import random
import pandas as pd

from PSO import PSO
from particle import particle

import matplotlib.pyplot as plt
import math


#--- INPUT DATA --------------------------------------------------------------------------

input_values_x = []
input_values_y = []

#--- ANN PROPERTIES ----------------------------------------------------------------------

input_neurons = 1
hidden_neurons = 4
output_neurons = 1

#--- PARTICLE SWARM OPTIMIZATION HYPERPARAMS ---------------------------------------------
    
swarmSize = 50
iterations = 400
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

'''
Title: Activation methods for neural nets
Author: Towards AI
Title of program/source code: Buidling Neural Networks with Python Code and Math in Detail
Type: Source code snippet
Availability: https://medium.com/towards-artificial-intelligence/building-neural-networks-with-python-code-and-math-in-detail-ii-bbe8accbf3d1
'''

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
        
        
        '''
        Title: Train Neural Network (Numpy)— Particle Swarm Optimization(PSO)
        Author: Zeeshan Ahmad
        Title of program/source code: Generating the node connections of an ANN given values (adapted)
        Type: Source code snippet
        Availability: https://medium.com/@zeeshanahmad10809/train-neural-network-numpy-particle-swarm-optimization-pso-93f289fc8a8e
        Licence: https://github.com/zeeshanahmad10809/neural-net-pso/blob/master/LICENSE
        '''
        # define the architecture of the ANN based on its properties  
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
        
        predicted_val = hyperbolicTangent(final)
        return MSE(target_val, predicted_val)

#--- PREDICT FINAL VALUE ---------------------------------------------------------------

def final_prediction(input_values_x, settings):
        
    # define the architecture of the ANN based on its properties
    # reference https://medium.com/@zeeshanahmad10809/train-neural-network-numpy-particle-swarm-optimization-pso-93f289fc8a8e
    layer1_weights  = settings[0 : input_neurons * hidden_neurons].reshape((input_neurons, hidden_neurons))
    layer1_bias     = settings[input_neurons * hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons].reshape((hidden_neurons, ))
    layer2_weights  = settings[(input_neurons * hidden_neurons) + hidden_neurons:(input_neurons * hidden_neurons) + hidden_neurons +\
    (hidden_neurons * output_neurons)].reshape((hidden_neurons, output_neurons))
    layer2_bias     = settings[(input_neurons * hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons): (input_neurons *\
    hidden_neurons) + hidden_neurons + (hidden_neurons * output_neurons) + output_neurons].reshape((output_neurons, ))
        
    # forward pass calculations
    layer1_output = np.dot(input_values_x, layer1_weights) + layer1_bias
    AF_output = np.tanh(layer1_output)
    layer2_output = np.dot(AF_output, layer2_weights) + layer2_bias
    final = layer2_output
    

    # get the probablilty of the final output
    predicted_val = hyperbolicTangent(final)
    
    # return predicted value
    return predicted_val
    

#--- COST FUNCTION ---------------------------------------------------------------------------

'''
Title: Loss and Loss Functions for Training Deep Learning Neural Networks
Author: Jason Brownlee
Title of program/source code: Function for calculating Mean-Squared Error (MSE)
Type: Source code snippet
Availability: https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
'''

def MSE(target_val, predicted_val):
    sum_square_error = 0.0
    for i in range(len(target_val)):
        sum_square_error += (target_val[i] - predicted_val[i])**2.0
    mean_square_error = 1.0 / len(target_val) * sum_square_error
    return mean_square_error

#---------------------------------------------------------------------------------------------

'''
(Part of initial implementation - no longer in use)

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

#--- GRAPHS -----------------------------------------------------------------------------

# tanh
def plot_tanh(predicted_values):
    #tanh points
    x=[]
    y = []
    i=-10
    while (i<=10):
        x.append(i)
        y.append(math.tanh(i))
        i=i+0.1
        
    pred_x = []
    pred_y = []
    j=-1
    q=0
    while (j<=1):
        pred_x.append(j)
        pred_y.append(predicted_values[q])
        j=j+(2/101)
        q=q+1
    
    plt.plot(x,y)
    plt.plot(pred_x, pred_y)
    
    plt.axvline(x=0.00,linewidth=2, color='#f1f1f1')
    plt.axhline(y=0.00,linewidth=2, color='#f1f1f1')
    plt.grid(linestyle='-',
        linewidth=0.5,color='#f1f1f1')
    plt.show()

# x cubed
def plot_cubic(predicted_values):
    #cubic points
    x=[]
    y = []
    i=-1
    while (i<=1):
        x.append(i)
        y.append(np.power(i,3))
        i=i+0.1
    
    pred_x = []
    pred_y = []
    j=-1
    q=0
    while (j<=1):
        pred_x.append(j)
        pred_y.append(predicted_values[q])
        j=j+(2/101)
        q=q+1
        
    plt.plot(x, y)
    plt.plot(pred_x, pred_y)
    
    plt.axvline(x=0.00,linewidth=2, color='#f1f1f1')
    plt.axhline(y=0.00,linewidth=2, color='#f1f1f1')
    plt.grid(linestyle='-',
        linewidth=0.5,color='#f1f1f1')
    plt.show()

#--- RUN MAIN ---------------------------------------------------------------------------- 

if __name__ == '__main__':

    # read values from file and append to arrays
    data = pd.read_csv("1in_cubic.txt", header=None, delim_whitespace=True)
        
    input_values_x = data[0].to_numpy().reshape(101,1)
    input_values_y = data[1].to_numpy().reshape(101,1)
    
    # initialise swarm
    swarm = PSO(swarmSize, dimensions, weight_range, learning_rate_range, inertia_range, cog_constant, soc_constant, numberOfInformants)
    
    # call optimization on swarm object
    swarm.optimize(feed_forward_train, input_values_x, input_values_y, iterations)
    
    # print final prediction values
    predicted_values = final_prediction(input_values_x, swarm.GLOBAL_positionBest)
    
    # compare predicted value to actual value using MSE
    ACCURACY = MSE(input_values_y, predicted_values)
    
    print("\nPREDICTED VALUES: " + str(predicted_values))
    #print("\nACTUAL VALUES: " + str(input_values_y))
    print("\nACCURACY: " + str(ACCURACY))
    
    plot_cubic(predicted_values)
    



