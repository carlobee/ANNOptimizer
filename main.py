"""
A program that implements a basic feedforward multilayer Artificial Neural Network

Written By Carl Bettosi and Mamta Sofat
16/11/202

"""

import numpy as np
import random

# create ANN architecture
numberOfInputs = 2 #HARDCODED - CHANGE

# get some properties from user
numberOfNeurons = int(input("How many neurons? "))

# give restraints to number of neurons that can be generated
if(numberOfNeurons >= (2*numberOfInputs) or numberOfNeurons < 1):
    print("Error: Number of neurons must be between 1 and twice the number of inputs.")
    numberOfNeurons = int(input("How many neurons? "))

numberOfLayers = int(input("How many hidden layers? "))

print(numberOfNeurons)
print(numberOfLayers)

# add bias


# assign random weights to hidden and output layer

# generate randomly assigned weights (edges)
HiddenLayers = {};

print("----------------------------------------")

# create weights for input > first hidden layer
arr = np.random.random((numberOfInputs, numberOfNeurons))
HiddenLayers[0] = arr
print(arr)

for x in range(numberOfLayers-1):
    print("----------------------------------------")
    arr = np.random.random((numberOfNeurons, numberOfNeurons))
    HiddenLayers[x] = arr
    print(arr)
    
    
# define activation functions


#reference this https://medium.com/towards-artificial-intelligence/building-neural-networks-with-python-code-and-math-in-detail-ii-bbe8accbf3d1
def sigmoid(x):
    return 1/(1+np.exp(-x))

def hyperbolicTangent(x):
    return np.tanh(x)
    
def cosine(x):
    return np.cos(x)
    
#def Gaussian(x):
    #return exp    

# run training code (do this last - PSO)



# Find error in predication